"""
retrieval/retriever.py

Hybrid retrieval pipeline:

  query
   ├─ dense retrieval (ChromaDB)     → ranked list A
   └─ lexical retrieval (BM25)       → ranked list B
           ↓
       RRF fusion (k=60)
           ↓
       optional CrossEncoder reranking
           ↓
       evidence expansion (neighbouring text chunks)
           ↓
       RetrievalResponse

The retrieval layer returns EVIDENCE, not answers.
The answer-generation layer consumes RetrievalResponse.

Usage
-----
    python retrieval/retriever.py            # interactive CLI
    python retrieval/retriever.py --diagnose # diagnostic mode (top-5 verbose)
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from bm25_index import BM25Index
from schema import Evidence, RetrievalResponse, RetrievalResult


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "academic_documents_v3"
BM25_PATH = "data/bm25_index.json"
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
    1. Dense candidate retrieval from ChromaDB (semantic)
    2. Lexical candidate retrieval from BM25
    3. RRF fusion of both ranked lists
    4. Optional CrossEncoder reranking
    5. Evidence expansion (neighbouring text chunks)
    6. Return RetrievalResponse
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: Optional[str] = RERANKER_MODEL,
        persist_directory: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        bm25_path: str = BM25_PATH,
    ):
        print(f"[Retriever] Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        self.reranker: Optional[CrossEncoder] = None
        if reranker_model:
            print(f"[Retriever] Loading reranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)

        self.client = chromadb.PersistentClient(path=persist_directory)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            raise RuntimeError(
                f"Collection '{collection_name}' not found. "
                "Run the indexer first."
            )

        self.bm25 = BM25Index(index_path=bm25_path)
        print(
            f"[Retriever] Ready. "
            f"ChromaDB: {self.collection.count()} items, "
            f"BM25: {len(self.bm25)} items."
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
        Query ChromaDB and return (evidence_id, distance) pairs,
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

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(candidate_k, self.collection.count()),
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
        """Return BM25-ranked (evidence_id, score) pairs."""
        return self.bm25.search(
            query=query,
            top_k=candidate_k,
            doc_id=doc_id,
            types=types,
        )

    # -------------------------------------------------------------------
    # Stage 3: RRF fusion
    # -------------------------------------------------------------------

    def _fuse(
        self,
        dense: List[Tuple[str, float]],
        lexical: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        return reciprocal_rank_fusion([dense, lexical], k=RRF_K)

    # -------------------------------------------------------------------
    # Stage 4: Fetch full Evidence from ChromaDB
    # -------------------------------------------------------------------

    def _fetch_evidence(
        self, ids: List[str]
    ) -> Dict[str, Evidence]:
        """
        Retrieve full evidence records from ChromaDB by ID.
        Returns a dict keyed by evidence ID.
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
        """
        if self.reranker is None or not candidates:
            return candidates[:top_k]

        ids = [eid for eid, _ in candidates]
        texts = [
            ev_map[eid].content if eid in ev_map else ""
            for eid in ids
        ]
        pairs = [(query, t) for t in texts]
        scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return reranked[:top_k]

    # -------------------------------------------------------------------
    # Stage 6: Evidence expansion
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
                other.id != evidence.id
                and other.doc_id == evidence.doc_id
                and other.type == "text"
                and other.page == evidence.page
            ):
                neighbours.append(other.content[:300])
                if len(neighbours) >= 1:
                    break
        return neighbours

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

        Returns
        -------
        RetrievalResponse with query and ranked evidence.
        """
        # 1. Dual retrieval
        dense = self._dense_retrieve(query, candidate_k, doc_id, types)
        lexical = self._lexical_retrieve(query, candidate_k, doc_id, types)

        # 2. RRF fusion
        fused = self._fuse(dense, lexical)

        # 3. Fetch top candidates from ChromaDB
        fetch_ids = [eid for eid, _ in fused[: candidate_k * 2]]
        ev_map = self._fetch_evidence(fetch_ids)

        # 4. Rerank (or fall back to RRF scores)
        if use_reranker and self.reranker:
            ranked = self._rerank(query, fused, ev_map, top_k)
        else:
            ranked = fused[:top_k]

        # 5. Build RetrievalResult objects
        results: List[RetrievalResult] = []
        for eid, score in ranked:
            if eid not in ev_map:
                continue
            ev = ev_map[eid]
            ctx = self._expand_evidence(ev, ev_map) if expand_evidence else []
            results.append(
                RetrievalResult(
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
                    context_chunks=ctx,
                )
            )

        return RetrievalResponse(query=query, results=results)

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
            RetrievalResult(
                id=ev.id,
                doc_id=ev.doc_id,
                type=ev.type,
                score=score,
                content=ev.content,
                page=ev.page,
                section=ev.section,
                bbox=ev.bbox,
                caption=ev.caption,
                artifact_path=ev.artifact_path,
            )
            for (eid, score), ev in (
                (item, ev_map[item[0]])
                for item in dense
                if item[0] in ev_map
            )
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
        print(f"   Type      : {r.type}")
        print(f"   Score     : {r.score:.4f}")
        print(f"   Page      : {r.page}")
        print(f"   Document  : {r.doc_id}")
        if r.caption:
            print(f"   Caption   : {r.caption[:80]}")
        if r.artifact_path:
            print(f"   Artifact  : {r.artifact_path}")
        if r.section:
            print(f"   Section   : {r.section[:60]}")
        print(f"   Content   : {r.content[:120]}…")
        print()


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