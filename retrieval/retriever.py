"""
retrieval/retriever.py

Hybrid retrieval pipeline:

  query ─► query understanding (deterministic; query_understanding.py)
   ├─ dense retrieval (ChromaDB, ORIGINAL query)      → ranked list A
   ├─ lexical retrieval (BM25, ORIGINAL query)         → ranked list B
   └─ entity-focused BM25 (method names + property)    → ranked list C
                                 (only when a method name is detected)
           ↓
       RRF fusion (k=60)
           ↓
       optional CrossEncoder reranking  (text relevance only; it never sees
                                         figure pixels)
           ↓
       entity-aware re-ranking (ranking.py; rank-based, tiered, documented)
           ↓
       modality-aware diversification  (optional)
           ↓
       evidence expansion (neighbouring text chunks)
           ↓
       evidence validation (evidence_validator.py) [+ optional document check]
           ↓
       RetrievalResponse (results + query_analysis + validation)

The retrieval layer returns EVIDENCE, not answers.  It never calls an LLM.
The answer-generation layer consumes RetrievalResponse.

Separate module (corpus_similarity.py) handles document-level
similarity search, which is a conceptually different operation.

Usage
-----
    python retrieval/retriever.py            # interactive CLI
    python retrieval/retriever.py --diagnose # diagnostic mode (top-5 verbose)
"""

import argparse
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from bm25_index import BM25Index
from evidence_validator import (
    EvidenceAssessment, apply_document_check, assess_evidence,
    document_findings, validate,
)
from query_understanding import QueryAnalysis, analyze_query
from ranking import rank_candidates
from schema import UID_SEPARATOR, Evidence, RetrievalResponse, RetrievalResult

logger = logging.getLogger("retrieval.retriever")


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DIR = "data/chroma_db"
# v4: vector-store keys are doc-qualified uids ('<doc_id>::<id>').  In v3,
# parser table/figure ids collided across papers and silently overwrote
# each other (148 evidence items -> 137 stored).  v3 is left untouched.
COLLECTION_NAME = "academic_documents_v4"
BM25_PATH = "data/bm25_index_v4.json"
RRF_K = 60          # Standard Reciprocal Rank Fusion constant


# ---------------------------------------------------------------------------
# RRF helper
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Each candidate receives score = Σ 1 / (rank + k) across all lists.
    Lists need not be the same length.

    Returns a sorted list of (evidence_id, fused_score) in descending order.
    """
    scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (eid, _) in enumerate(ranked, start=1):
            scores[eid] += 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# AcademicRetriever
# ---------------------------------------------------------------------------

class AcademicRetriever:
    """
    Hybrid academic document retriever.

    Steps:
    1. Query understanding (entities, information need, relation, modality)
    2. Dense candidate retrieval from ChromaDB (semantic, original query)
    3. Lexical candidate retrieval from BM25 (original query)
       + entity-focused BM25 list when the query names a method
    4. RRF fusion of the ranked lists
    5. Optional CrossEncoder reranking
    6. Entity-aware re-ranking (see ranking.py)
    7. Optional modality-aware diversification
    8. Evidence expansion (neighbouring text chunks)
    9. Evidence validation (+ optional document-level check)
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: Optional[str] = RERANKER_MODEL,
        persist_directory: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        bm25_path: str = BM25_PATH,
    ):
        # Imported here so model-free callers (tests, from_components) do not
        # pay the import cost.
        from sentence_transformers import CrossEncoder, SentenceTransformer

        print(f"[Retriever] Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        self.reranker = None
        if reranker_model:
            print(f"[Retriever] Loading reranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)

        self.client = chromadb.PersistentClient(path=persist_directory)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            raise RuntimeError(
                f"Collection '{collection_name}' not found in "
                f"'{persist_directory}'. Run the indexer first: "
                "python retrieval/indexer.py"
            )

        self.bm25 = BM25Index(index_path=bm25_path)
        self._check_index_consistency()
        print(
            f"[Retriever] Ready. "
            f"ChromaDB: {self.collection.count()} items, "
            f"BM25: {len(self.bm25)} items."
        )

    @classmethod
    def from_components(
        cls,
        embedding_model: Any,
        collection: Any,
        bm25: BM25Index,
        reranker: Any = None,
    ) -> "AcademicRetriever":
        """
        Build a retriever from already-constructed components (no model
        downloads).  Used by the deterministic unit tests and by callers
        that share models between components.
        """
        obj = object.__new__(cls)
        obj.embedding_model = embedding_model
        obj.reranker = reranker
        obj.client = None
        obj.collection = collection
        obj.bm25 = bm25
        return obj

    def _check_index_consistency(self) -> None:
        """Warn when the index uses legacy (pre-uid) keys or BM25 is out of sync."""
        try:
            ids = self.collection.peek(limit=1).get("ids", [])
            if len(ids) and UID_SEPARATOR not in ids[0]:
                print(
                    "[Retriever] [WARN] Collection uses legacy evidence keys "
                    "(no doc prefix): table/figure ids may collide across "
                    "documents. Re-index: python retrieval/indexer.py --reset"
                )
        except Exception:  # diagnostics only
            pass
        if self.collection.count() != len(self.bm25):
            print(
                f"[Retriever] [WARN] ChromaDB ({self.collection.count()}) and "
                f"BM25 ({len(self.bm25)}) item counts differ; re-index both "
                "together."
            )

    # -------------------------------------------------------------------
    # Stage 1: Dense retrieval
    # -------------------------------------------------------------------

    def _dense_retrieve(
        self,
        query: str,
        candidate_k: int,
        doc_id: Optional[str],
        types: Optional[List[str]],
    ) -> List[Tuple[str, float]]:
        """
        Query ChromaDB and return (evidence_uid, pseudo_score) pairs,
        ordered by ascending distance (lower = more similar).
        """
        query_embedding = (
            self.embedding_model.encode(
                [query], normalize_embeddings=True
            )[0]
            .tolist()
        )

        # Build ChromaDB where clause
        where = None
        conditions = []
        if doc_id:
            conditions.append({"doc_id": {"$eq": doc_id}})
        if types and len(types) == 1:
            conditions.append({"type": {"$eq": types[0]}})
        elif types:
            conditions.append({"type": {"$in": types}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(candidate_k, count),
            where=where,
        )

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Convert distance to a pseudo-score (lower distance = higher score)
        max_d = max(distances) if distances else 1.0
        ranked = [
            (eid, 1.0 - (d / (max_d + 1e-9)))
            for eid, d in zip(ids, distances)
        ]
        return ranked  # already sorted by ascending distance from ChromaDB

    # -------------------------------------------------------------------
    # Stage 2: Lexical retrieval
    # -------------------------------------------------------------------

    def _lexical_retrieve(
        self,
        query: str,
        candidate_k: int,
        doc_id: Optional[str],
        types: Optional[List[str]],
    ) -> List[Tuple[str, float]]:
        """Return BM25-ranked (evidence_uid, score) pairs."""
        return self.bm25.search(
            query=query,
            top_k=candidate_k,
            doc_id=doc_id,
            types=types,
            return_uids=True,
        )

    # -------------------------------------------------------------------
    # Stage 3: RRF fusion
    # -------------------------------------------------------------------

    def _fuse(
        self,
        dense: List[Tuple[str, float]],
        lexical: List[Tuple[str, float]],
        *extra: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        return reciprocal_rank_fusion([dense, lexical, *extra], k=RRF_K)

    # -------------------------------------------------------------------
    # Stage 4: Fetch full Evidence from ChromaDB
    # -------------------------------------------------------------------

    def _fetch_evidence(
        self, ids: List[str]
    ) -> Dict[str, Evidence]:
        """
        Retrieve full evidence records from ChromaDB by uid.
        Returns a dict keyed by evidence uid.
        """
        if not ids:
            return {}
        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        ev_map: Dict[str, Evidence] = {}
        for eid, doc, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            ev_map[eid] = Evidence.from_chroma_result(eid, doc, meta)
        return ev_map

    def _fetch_document_evidence(self, doc_id: str) -> List[Evidence]:
        """All indexed evidence of one document (for document-level checks)."""
        got = self.collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["documents", "metadatas"],
        )
        return [
            Evidence.from_chroma_result(eid, doc, meta)
            for eid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
        ]

    # -------------------------------------------------------------------
    # Stage 5: Reranking
    # -------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        ev_map: Dict[str, Evidence],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        CrossEncoder reranking over the fused candidate set.
        Falls back to RRF scores if reranker is not loaded.

        Returned scores are raw CrossEncoder logits (unbounded; not
        probabilities).  Only candidates present in ev_map are scored.
        """
        if self.reranker is None or not candidates:
            return candidates[:top_k]

        ids = [eid for eid, _ in candidates if eid in ev_map]
        texts = [ev_map[eid].content for eid in ids]
        pairs = [(query, t) for t in texts]
        scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(ids, [float(s) for s in scores]),
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked[:top_k]

    # -------------------------------------------------------------------
    # Stage 6: Modality-aware diversification
    # -------------------------------------------------------------------

    def _diversify(
        self,
        results: List[RetrievalResult],
        top_k: int,
        max_per_type: Optional[Dict[str, int]],
    ) -> List[RetrievalResult]:
        """
        Enforce per-type caps on the ranked result list.

        Walks the ranked list in order, admitting each item if its type
        has not yet hit its cap.  Overflow items (type cap reached) are
        collected and appended to fill any remaining top_k slots after
        the primary pass.

        This prevents text chunks from crowding out relevant tables or
        figures when cross-modal evidence exists in the candidate pool.

        Parameters
        ----------
        results:
            Ranked RetrievalResult list (descending score).
        top_k:
            Maximum final result count.
        max_per_type:
            Dict mapping type → max allowed occurrences in output.
            Types not mentioned are treated as uncapped.
            Example: {"text": 3, "table": 2, "figure": 2}

        Returns
        -------
        Diversified list of at most top_k RetrievalResult objects.
        """
        if not max_per_type:
            return results[:top_k]

        counts: Dict[str, int] = defaultdict(int)
        chosen: List[RetrievalResult] = []
        overflow: List[RetrievalResult] = []

        for r in results:
            cap = max_per_type.get(r.type, top_k)  # uncapped if not in dict
            if counts[r.type] < cap:
                chosen.append(r)
                counts[r.type] += 1
            else:
                overflow.append(r)
            if len(chosen) >= top_k:
                break

        # Fill remaining slots from overflow (preserving score order)
        if len(chosen) < top_k:
            chosen.extend(overflow[: top_k - len(chosen)])

        return chosen

    # -------------------------------------------------------------------
    # Stage 7: Evidence expansion
    # -------------------------------------------------------------------

    def _expand_evidence(
        self,
        evidence: Evidence,
        ev_map: Dict[str, Evidence],
    ) -> List[str]:
        """
        Attach neighbouring text chunks to an evidence item (context).
        Currently returns at most 1 neighbouring chunk from the same doc/page.
        """
        if evidence.type != "text":
            return []

        neighbours: List[str] = []
        for other in ev_map.values():
            if (
                other.uid != evidence.uid
                and other.doc_id == evidence.doc_id
                and other.type == "text"
                and other.page == evidence.page
            ):
                neighbours.append(other.content[:300])
                if len(neighbours) >= 1:
                    break
        return neighbours

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _to_result(
        ev: Evidence,
        score: float,
        context: List[str],
        details: Dict[str, Any],
        assessment: Optional[EvidenceAssessment],
    ) -> RetrievalResult:
        return RetrievalResult(
            id=ev.id,
            doc_id=ev.doc_id,
            type=ev.type,
            score=float(score),
            content=ev.content,
            page=ev.page,
            section=ev.section,
            bbox=ev.bbox,
            caption=ev.caption,
            artifact_path=ev.artifact_path,
            context_chunks=context,
            uid=ev.uid,
            score_details=details,
            assessment=assessment.to_dict() if assessment else None,
            metadata=dict(ev.metadata or {}),
        )

    @staticmethod
    def _rank_positions(ranked: List[Tuple[str, float]]) -> Dict[str, int]:
        return {eid: i for i, (eid, _) in enumerate(ranked, start=1)}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def search(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        types: Optional[List[str]] = None,
        use_reranker: bool = True,
        expand_evidence: bool = True,
        diversify: bool = False,
        max_per_type: Optional[Dict[str, int]] = None,
        entity_aware: bool = True,
        validate_evidence: bool = True,
        document_check: bool = False,
    ) -> RetrievalResponse:
        """
        Run the full hybrid retrieval pipeline.

        Parameters
        ----------
        query:
            Natural-language question.
        candidate_k:
            Number of candidates from each retrieval stage before fusion.
        top_k:
            Final number of evidence items to return.
        doc_id:
            If set, restrict retrieval to this document only.
        types:
            If set, restrict retrieval to these evidence types.
            E.g. ['table'] to retrieve only tables.
        use_reranker:
            Whether to apply CrossEncoder reranking after fusion.
        expand_evidence:
            Whether to attach neighbouring chunks to text results.
        diversify:
            Whether to apply modality-aware diversification after reranking.
            When True, uses max_per_type to cap each evidence type so visual
            evidence (tables/figures) is not crowded out by text chunks.
        max_per_type:
            Per-type cap applied during diversification.
            Only used when diversify=True.
            Defaults to {"text": top_k-2, "table": 2, "figure": 1} when None
            and diversify=True.
            Example: {"text": 3, "table": 2, "figure": 2}
        entity_aware:
            Apply query understanding (entity-focused BM25 list) and the
            entity-aware re-ranking of ranking.py.  False reproduces the
            previous ordering (useful for ablation / before-after runs);
            ``score`` then keeps its legacy meaning (raw CrossEncoder logit
            or RRF score).
        validate_evidence:
            Attach a ValidationResult (evidence_validator.py).
        document_check:
            Additionally scan ALL indexed evidence of the filtered document
            (``doc_id``) — or of every document in the results when no
            filter is given — before reporting a requested method as absent.
            Supporting evidence found this way is appended to the results
            (score_details['added_by'] = 'document_check').

        Returns
        -------
        RetrievalResponse with query, ranked evidence, query_analysis and
        validation.  ``score`` is a ranking score: with entity_aware=True it
        is the rank-based final score of ranking.py — never a probability.
        """
        analysis = analyze_query(query, doc_id=doc_id)
        logger.info("query analysis: %s", analysis.to_dict())

        # 1. Retrieval: dense + BM25 on the ORIGINAL query
        dense = self._dense_retrieve(query, candidate_k, doc_id, types)
        lexical = self._lexical_retrieve(query, candidate_k, doc_id, types)
        lists = [dense, lexical]

        # 1b. Entity-focused lexical list (supplements, never replaces)
        focused: List[Tuple[str, float]] = []
        if entity_aware and analysis.is_method_specific:
            focused = self._lexical_retrieve(
                analysis.focused_query(), candidate_k, doc_id, types
            )
            lists.append(focused)

        # 2. RRF fusion
        fused = self._fuse(*lists)

        # 3. Fetch top candidates from ChromaDB
        fetch_ids = [eid for eid, _ in fused[: candidate_k * 2]]
        ev_map = self._fetch_evidence(fetch_ids)

        # Corpus isolation guard (defence in depth: both retrievers already
        # filter; this guarantees no cross-document leakage downstream).
        if doc_id:
            leaked = [u for u, ev in ev_map.items() if ev.doc_id != doc_id]
            if leaked:
                logger.warning("Dropping %d evidence items outside doc_id=%s",
                               len(leaked), doc_id)
                for u in leaked:
                    ev_map.pop(u)
        if types:
            ev_map = {u: ev for u, ev in ev_map.items() if ev.type in types}

        pool = [(eid, s) for eid, s in fused if eid in ev_map]
        rrf_scores = dict(fused)
        positions = {
            "dense_rank": self._rank_positions(dense),
            "bm25_rank": self._rank_positions(lexical),
            "focused_bm25_rank": self._rank_positions(focused),
        }

        # 4. Rerank (or fall back to RRF order)
        rerank_scores: Dict[str, float] = {}
        if use_reranker and self.reranker:
            ranked = self._rerank(query, pool, ev_map, len(pool))
            rerank_scores = dict(ranked)
        else:
            ranked = pool
        if not entity_aware:
            ranked = ranked[: top_k * 3]

        # 5. Assess every candidate; entity-aware re-ranking
        assessments: Dict[str, EvidenceAssessment] = {
            eid: assess_evidence(ev_map[eid], analysis) for eid, _ in ranked
        }

        result_pool: List[RetrievalResult] = []
        if entity_aware:
            ordered = rank_candidates(
                [(eid, ev_map[eid].type) for eid, _ in ranked],
                analysis, assessments,
            )
            for c in ordered:
                ev = ev_map[c.uid]
                details = {
                    "dense_rank": positions["dense_rank"].get(c.uid),
                    "bm25_rank": positions["bm25_rank"].get(c.uid),
                    "focused_bm25_rank": positions["focused_bm25_rank"].get(c.uid),
                    "rrf_score": round(rrf_scores.get(c.uid, 0.0), 6),
                    "rerank_score": (round(rerank_scores[c.uid], 4)
                                     if c.uid in rerank_scores else None),
                    "base_order": "cross_encoder" if rerank_scores else "rrf",
                    "base_rank": c.base_rank,
                    "base_score": c.base_score,
                    "tier": c.tier,
                    "adjustments": c.adjustments,
                    "final_score": c.final_score,
                    "score_semantics": "ordinal ranking score; not a probability",
                }
                ctx = self._expand_evidence(ev, ev_map) if expand_evidence else []
                result_pool.append(self._to_result(
                    ev, c.final_score, ctx, details, assessments.get(c.uid)))
        else:
            for eid, score in ranked:
                ev = ev_map[eid]
                ctx = self._expand_evidence(ev, ev_map) if expand_evidence else []
                details = {
                    "rrf_score": round(rrf_scores.get(eid, 0.0), 6),
                    "rerank_score": (round(rerank_scores[eid], 4)
                                     if eid in rerank_scores else None),
                    "score_semantics": "legacy: raw CrossEncoder logit or RRF score",
                }
                result_pool.append(self._to_result(
                    ev, score, ctx, details, assessments.get(eid)))

        # 6. Modality-aware diversification (optional)
        if diversify:
            type_caps = max_per_type or {
                "text": max(1, top_k - 2),
                "table": min(2, top_k),
                "figure": min(1, top_k),
            }
            results = self._diversify(result_pool, top_k, type_caps)
        else:
            results = result_pool[:top_k]

        response = RetrievalResponse(
            query=query, results=results, query_analysis=analysis,
        )
        if not validate_evidence:
            return response

        # 7. Evidence validation over the RETURNED results
        validation = validate(analysis, results, assessments)

        # 8. Optional document-level check before claiming absence
        if document_check and analysis.is_method_specific:
            docs = [doc_id] if doc_id else sorted({r.doc_id for r in results})
            by_doc = {d: self._fetch_document_evidence(d) for d in docs}
            findings = document_findings(analysis, by_doc)
            present = {r.uid for r in results}
            added = False
            for d, f in findings.items():
                ev_by_uid = {e.uid: e for e in by_doc[d]}
                for r_name in analysis.entities:
                    entry = f.get(r_name, {})
                    if entry.get("status") != "associated":
                        continue
                    for u in entry["evidence_ids"]:
                        if u in present or u not in ev_by_uid:
                            continue
                        ev = ev_by_uid[u]
                        a = assess_evidence(ev, analysis)
                        assessments[u] = a
                        results.append(self._to_result(
                            ev, 0.0, [], {"added_by": "document_check",
                                          "score_semantics": "not ranked"}, a))
                        present.add(u)
                        added = True
            if added:
                validation = validate(analysis, results, assessments)
            validation = apply_document_check(validation, analysis, findings)

        response.validation = validation
        return response

    # -------------------------------------------------------------------
    # Dense-only convenience method (for testing / ablation)
    # -------------------------------------------------------------------

    def search_dense_only(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        types: Optional[List[str]] = None,
    ) -> RetrievalResponse:
        dense = self._dense_retrieve(query, top_k, doc_id, types)
        ev_map = self._fetch_evidence([eid for eid, _ in dense])
        results = [
            self._to_result(ev_map[eid], score, [], {"dense_score": score}, None)
            for eid, score in dense
            if eid in ev_map and (not doc_id or ev_map[eid].doc_id == doc_id)
        ]
        return RetrievalResponse(query=query, results=results[:top_k])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_diagnostic(response: RetrievalResponse) -> None:
    print(f"\nQuery: {response.query}")
    print(f"\nTop {len(response.results)} results:\n")
    for i, r in enumerate(response.results, 1):
        print(f"{i}.")
        print(f"   ID        : {r.id}")
        print(f"   UID       : {r.uid}")
        print(f"   Type      : {r.type}")
        print(f"   Score     : {r.score:.4f}  (ranking score, not a probability)")
        print(f"   Page      : {r.page}")
        print(f"   Document  : {r.doc_id}")
        if r.caption:
            print(f"   Caption   : {r.caption[:80]}")
        if r.artifact_path:
            print(f"   Artifact  : {r.artifact_path}")
        if r.section:
            print(f"   Section   : {r.section[:60]}")
        if r.assessment:
            print(f"   Evidence  : {r.assessment['label']}")
        print(f"   Content   : {r.content[:120]}…")
        print()
    if response.validation is not None:
        v = response.validation
        print(f"Validation: {v.status}  (entity_mismatch={v.entity_mismatch})")
        print(f"  {v.explanation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid academic retriever.")
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Restrict retrieval to a specific document ID.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print verbose diagnostic output.",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_DIR,
    )
    parser.add_argument(
        "--bm25-path",
        default=BM25_PATH,
    )
    args = parser.parse_args()

    retriever = AcademicRetriever(
        persist_directory=args.chroma_dir,
        bm25_path=args.bm25_path,
    )

    print("\nEnter queries (type 'exit' to quit):\n")
    while True:
        query = input("Query: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        response = retriever.search(
            query=query,
            top_k=args.top_k,
            doc_id=args.doc_id,
        )

        if args.diagnose:
            _print_diagnostic(response)
        else:
            print(json.dumps(response.to_dict(), indent=2))
