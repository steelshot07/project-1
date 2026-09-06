"""
retrieval/bm25_index.py

Lexical BM25 index over Evidence content.

Uses rank_bm25.BM25Okapi.  The index is persisted to a JSON file so
it does not need to be rebuilt on every run.

Design:
- Index stores (tokenized corpus, evidence IDs, lightweight metadata)
- Supports filtering by doc_id and/or type at query time
- No torch / image-processing dependencies
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from schema import Evidence


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser, lowercase."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Persistent BM25 index for lexical retrieval over Evidence items.

    The index is backed by a JSON file containing the tokenised corpus,
    evidence IDs, and per-item metadata needed for filtering.
    On first use the index is built in memory and then saved to disk.
    On subsequent runs it is loaded from disk.
    """

    DEFAULT_PATH = "data/bm25_index.json"

    def __init__(self, index_path: str = DEFAULT_PATH):
        self.index_path = index_path
        self._corpus: List[List[str]] = []   # tokenised documents
        self._ids: List[str] = []
        self._meta: List[Dict[str, Any]] = []  # {doc_id, type, page}
        self._bm25: Optional[BM25Okapi] = None

        if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
            self._load()

    # -------------------------------------------------------------------
    # Build / update
    # -------------------------------------------------------------------

    def add_evidence(self, evidence_list: List[Evidence]) -> None:
        """
        Add evidence items to the index.

        Items whose IDs already exist are replaced (upsert semantics).
        Call save() after adding to persist.
        """
        existing_ids = set(self._ids)

        for ev in evidence_list:
            if ev.id in existing_ids:
                # Replace in-place
                idx = self._ids.index(ev.id)
                self._corpus[idx] = _tokenize(ev.content)
                self._meta[idx] = {
                    "doc_id": ev.doc_id,
                    "type": ev.type,
                    "page": ev.page if ev.page is not None else -1,
                }
            else:
                self._corpus.append(_tokenize(ev.content))
                self._ids.append(ev.id)
                self._meta.append(
                    {
                        "doc_id": ev.doc_id,
                        "type": ev.type,
                        "page": ev.page if ev.page is not None else -1,
                    }
                )
                existing_ids.add(ev.id)

        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def save(self) -> None:
        """Persist the index to disk as JSON."""
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "corpus": self._corpus,
                    "ids": self._ids,
                    "meta": self._meta,
                },
                f,
            )
        print(
            f"[BM25Index] Saved {len(self._ids)} entries "
            f"→ {self.index_path}"
        )

    def _load(self) -> None:
        """Load a previously saved index from disk."""
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._corpus = data.get("corpus", [])
        self._ids = data.get("ids", [])
        self._meta = data.get("meta", [])
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        print(
            f"[BM25Index] Loaded {len(self._ids)} entries "
            f"← {self.index_path}"
        )

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_id: Optional[str] = None,
        types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return a ranked list of (evidence_id, bm25_score) tuples.

        Parameters
        ----------
        query:
            Free-text query.
        top_k:
            Maximum number of results to return.
        doc_id:
            If provided, restrict results to this document.
        types:
            If provided, restrict results to these evidence types.
            e.g. ['table', 'figure']

        Returns
        -------
        Sorted list of (evidence_id, score) in descending score order.
        """
        if self._bm25 is None or not self._ids:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        candidates: List[Tuple[str, float]] = []
        for i, (eid, meta) in enumerate(zip(self._ids, self._meta)):
            score = float(scores[i])
            if score <= 0:
                continue
            if doc_id and meta.get("doc_id") != doc_id:
                continue
            if types and meta.get("type") not in types:
                continue
            candidates.append((eid, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # -------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    def stats(self) -> Dict[str, Any]:
        from collections import Counter
        type_counts = Counter(m.get("type") for m in self._meta)
        doc_counts = Counter(m.get("doc_id") for m in self._meta)
        return {
            "total": len(self._ids),
            "by_type": dict(type_counts),
            "by_doc": dict(doc_counts),
        }
