"""
retrieval/corpus_similarity.py

Document-level corpus similarity search.

This module is CONCEPTUALLY SEPARATE from question-answer retrieval
(retriever.py).  It answers a different kind of query:

    Q&A retrieval:       "What does the literature say about ramp metering?"
                         → returns specific text/table/figure evidence

    Corpus similarity:   "Which papers are most similar to 2012.12104v1?"
                         → returns document-level similarity scores

Architecture
------------
- Each indexed document is represented as the centroid (mean) of all its
  chunk embeddings stored in ChromaDB.
- Document-level similarity is cosine similarity between centroids.
- No BM25, no reranking — this is purely embedding-space comparison.
- The ChromaDB collection is accessed read-only.

Usage
-----
    from corpus_similarity import CorpusSimilarity
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient("data/chroma_db")
    collection = client.get_collection("academic_documents_v3")

    cs = CorpusSimilarity(collection, model)

    # Closest papers to a target
    similar = cs.find_similar_documents("2012.12104v1", top_k=3)
    for s in similar:
        print(s["doc_id"], f"{s['similarity']:.3f}")

    # Pairwise comparison
    score = cs.compare_documents("2012.12104v1", "2012.13026v1")
    print(f"Similarity: {score:.3f}")

    # Cluster all indexed documents
    matrix = cs.similarity_matrix()

CLI
---
    python retrieval/corpus_similarity.py --source 2012.12104v1 --top-k 3
"""

import argparse
import sys
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# CorpusSimilarity
# ---------------------------------------------------------------------------

class CorpusSimilarity:
    """
    Document-level embedding similarity over the indexed corpus.

    Parameters
    ----------
    collection:
        ChromaDB collection that contains per-evidence embeddings.
    embedding_model:
        The same SentenceTransformer used during indexing.
        Required only for embed_and_compare(); not needed for find_similar_documents.
    """

    def __init__(self, collection, embedding_model=None):
        self.collection = collection
        self.model = embedding_model

    # -------------------------------------------------------------------
    # Document centroid
    # -------------------------------------------------------------------

    def get_document_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Compute the centroid embedding for a document.

        Retrieves all item embeddings for the given doc_id from ChromaDB
        and returns their mean vector.  Returns None if the document is
        not indexed.
        """
        results = self.collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["embeddings"],
        )
        embeddings = results.get("embeddings")
        if embeddings is None or (hasattr(embeddings, '__len__') and len(embeddings) == 0):
            return None
        arr = np.array(embeddings, dtype=np.float32)
        centroid = arr.mean(axis=0)
        # L2-normalise so dot-product = cosine similarity
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
        return centroid

    def _all_doc_ids(self) -> List[str]:
        """Return unique doc_ids present in the collection."""
        results = self.collection.get(include=["metadatas"])
        metas = results.get("metadatas") or []
        seen = set()
        ordered = []
        for m in metas:
            did = m.get("doc_id", "")
            if did and did not in seen:
                seen.add(did)
                ordered.append(did)
        return ordered

    # -------------------------------------------------------------------
    # find_similar_documents
    # -------------------------------------------------------------------

    def find_similar_documents(
        self,
        doc_id: str,
        top_k: int = 5,
        text_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Find the documents most similar to doc_id.

        Parameters
        ----------
        doc_id:
            Source document to compare against.
        top_k:
            Maximum number of similar documents to return.
        text_only:
            If True, only considers text-type evidence for the centroid
            (excludes table/figure embeddings which can skew the vector).

        Returns
        -------
        List of dicts with keys: doc_id, similarity.
        Sorted descending by similarity.
        """
        if text_only:
            src_results = self.collection.get(
                where={"$and": [
                    {"doc_id": {"$eq": doc_id}},
                    {"type": {"$eq": "text"}},
                ]},
                include=["embeddings"],
            )
        else:
            src_results = self.collection.get(
                where={"doc_id": {"$eq": doc_id}},
                include=["embeddings"],
            )

        src_embeddings = src_results.get("embeddings")
        if src_embeddings is None or (hasattr(src_embeddings, '__len__') and len(src_embeddings) == 0):
            return []

        src_vec = np.array(src_embeddings, dtype=np.float32).mean(axis=0)
        norm = np.linalg.norm(src_vec)
        if norm > 1e-9:
            src_vec = src_vec / norm

        # Query ChromaDB for nearest neighbours, excluding source doc
        n_results = min(self.collection.count(), 100)
        if n_results == 0:
            return []

        try:
            query_results = self.collection.query(
                query_embeddings=[src_vec.tolist()],
                n_results=n_results,
                where={"doc_id": {"$ne": doc_id}},
                include=["metadatas", "distances"],
            )
        except Exception:
            # Fallback: no $ne filter (ChromaDB version compatibility)
            query_results = self.collection.query(
                query_embeddings=[src_vec.tolist()],
                n_results=n_results,
                include=["metadatas", "distances"],
            )

        ids_list = query_results.get("ids", [[]])[0]
        metas_list = query_results.get("metadatas", [[]])[0]
        dist_list = query_results.get("distances", [[]])[0]

        # Aggregate per document
        doc_scores: Dict[str, List[float]] = defaultdict(list)
        for meta, dist in zip(metas_list, dist_list):
            other_id = meta.get("doc_id", "")
            if other_id and other_id != doc_id:
                # Convert L2 distance to a [0,1] similarity proxy.
                # For normalised embeddings, distance ∈ [0, 2].
                similarity = max(0.0, 1.0 - dist / 2.0)
                doc_scores[other_id].append(similarity)

        aggregated = [
            {
                "doc_id": did,
                "similarity": float(np.mean(scores)),
                "n_items": len(scores),
            }
            for did, scores in doc_scores.items()
        ]
        aggregated.sort(key=lambda x: x["similarity"], reverse=True)
        return aggregated[:top_k]

    # -------------------------------------------------------------------
    # compare_documents
    # -------------------------------------------------------------------

    def compare_documents(self, doc_id_1: str, doc_id_2: str) -> float:
        """
        Return cosine similarity in [-1, 1] between two documents.

        Uses the centroid embeddings of each document.
        Returns 0.0 if either document is not indexed.
        """
        vec1 = self.get_document_vector(doc_id_1)
        vec2 = self.get_document_vector(doc_id_2)
        if vec1 is None or vec2 is None:
            return 0.0
        # Both are already L2-normalised; dot product = cosine similarity
        return float(np.dot(vec1, vec2))

    # -------------------------------------------------------------------
    # similarity_matrix
    # -------------------------------------------------------------------

    def similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise cosine similarity for all indexed documents.

        Returns a nested dict: matrix[doc_id_1][doc_id_2] = score.
        Useful for clustering or visualisation.
        """
        doc_ids = self._all_doc_ids()
        vectors: Dict[str, np.ndarray] = {}
        for did in doc_ids:
            v = self.get_document_vector(did)
            if v is not None:
                vectors[did] = v

        matrix: Dict[str, Dict[str, float]] = {}
        for did1, v1 in vectors.items():
            matrix[did1] = {}
            for did2, v2 in vectors.items():
                matrix[did1][did2] = float(np.dot(v1, v2))

        return matrix

    # -------------------------------------------------------------------
    # embed_and_compare (query-text vs corpus)
    # -------------------------------------------------------------------

    def embed_and_compare(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find documents whose centroid is closest to an arbitrary text query.

        Useful for: "Find papers about experience replay in deep RL."

        Requires an embedding model to be provided at construction time.
        """
        if self.model is None:
            raise RuntimeError(
                "embedding_model must be provided to use embed_and_compare()."
            )
        query_vec = self.model.encode(
            [query_text], normalize_embeddings=True
        )[0].tolist()

        n_results = min(self.collection.count(), 200)
        if n_results == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metas_list = results.get("metadatas", [[]])[0]
        dist_list = results.get("distances", [[]])[0]

        doc_scores: Dict[str, List[float]] = defaultdict(list)
        for meta, dist in zip(metas_list, dist_list):
            did = meta.get("doc_id", "")
            if did:
                similarity = max(0.0, 1.0 - dist / 2.0)
                doc_scores[did].append(similarity)

        aggregated = [
            {"doc_id": did, "similarity": float(np.mean(scores))}
            for did, scores in doc_scores.items()
        ]
        aggregated.sort(key=lambda x: x["similarity"], reverse=True)
        return aggregated[:top_k]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Document-level corpus similarity search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source document ID to find similar papers for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar documents to return.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Compare --source against a specific second document.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Print pairwise similarity matrix for all indexed documents.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use only text-type evidence for document centroids.",
    )
    parser.add_argument("--chroma-dir", default="data/chroma_db")
    parser.add_argument(
        "--collection", default="academic_documents_v3"
    )
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=args.chroma_dir)
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        print(
            f"[ERROR] Collection '{args.collection}' not found. "
            "Run the indexer first."
        )
        sys.exit(1)

    cs = CorpusSimilarity(collection)

    if args.matrix:
        matrix = cs.similarity_matrix()
        doc_ids = list(matrix.keys())
        header = f"{'':25}" + "  ".join(f"{d[:15]:15}" for d in doc_ids)
        print(header)
        for did1 in doc_ids:
            row = f"{did1[:25]:25}  "
            row += "  ".join(f"{matrix[did1][did2]:7.4f}        " for did2 in doc_ids)
            print(row)
        return

    if args.compare:
        score = cs.compare_documents(args.source, args.compare)
        print(
            f"\nCosine similarity  {args.source} ↔ {args.compare}: {score:.4f}"
        )
        return

    print(f"\nDocuments most similar to: {args.source}\n")
    similar = cs.find_similar_documents(
        args.source, top_k=args.top_k, text_only=args.text_only
    )
    if not similar:
        print("  No other documents indexed, or document not found.")
    for i, s in enumerate(similar, 1):
        print(
            f"  {i}.  {s['doc_id']:30}  "
            f"similarity={s['similarity']:.4f}  "
            f"({s['n_items']} evidence items compared)"
        )


if __name__ == "__main__":
    main()
