"""
retrieval/schema.py

Common evidence model for the entire retrieval pipeline.
Every retrievable item — text chunk, table, or figure — is
represented as an Evidence instance.  RetrievalResult wraps
one Evidence item with a retrieval score for the caller.
RetrievalResponse is the final contract returned by the
retrieval layer (query → evidence, NOT query → answer).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

UID_SEPARATOR = "::"


def make_uid(doc_id: str, evidence_id: str) -> str:
    """Doc-qualified, globally unique evidence key."""
    return f"{doc_id}{UID_SEPARATOR}{evidence_id}"


@dataclass
class Evidence:
    """A single retrievable item from any modality."""

    id: str
    """Stable identifier as produced by the parser / chunker.
    Examples: 'table_p10_1', 'fig_p3_0', '2012.12104v1_chunk_4'

    NOTE: parser table/figure IDs are only unique WITHIN a document
    ('table_p8_1' exists in several papers).  Use ``uid`` (doc-qualified)
    wherever global uniqueness is required — e.g. as the vector-store key.
    """

    doc_id: str
    """Originating document identifier, e.g. '2012.12104v1'."""

    type: str
    """'text' | 'table' | 'figure'"""

    content: str
    """Searchable text representation of this evidence item.
    For text chunks: the chunk text.
    For tables: caption + flattened cell content.
    For figures: caption (+ OCR if available).
    """

    page: Optional[int] = None
    """1-indexed page number in the source PDF."""

    section: Optional[str] = None
    """Title of the containing section (text evidence only)."""

    bbox: Optional[List[float]] = None
    """[x0, y0, x1, y1] bounding box in PDF points."""

    caption: Optional[str] = None
    """Full caption string (tables and figures)."""

    artifact_path: Optional[str] = None
    """Relative path to the on-disk visual artifact.
    Tables  → .txt extraction file  (e.g. 'data/processed/.../tables/table_p10_1.txt')
    Figures → .png crop             (e.g. 'data/processed/.../figures/fig_p3_0.png')
    Text    → None
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Any additional fields from the parser output."""

    @property
    def uid(self) -> str:
        """Globally unique key: '<doc_id>::<id>'.  Used by ChromaDB and BM25."""
        return make_uid(self.doc_id, self.id)

    # -------------------------------------------------------------------
    # ChromaDB serialisation
    # -------------------------------------------------------------------

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """
        Flatten to a ChromaDB-compatible metadata dict.
        ChromaDB only accepts str / int / float / bool values.
        bbox is serialised as a JSON string.

        Extra ``metadata`` entries are included when they do not clash with
        the core fields: primitives as-is, lists/dicts as JSON strings,
        None values dropped (ChromaDB rejects None).
        """
        core = {
            "id": self.id,
            "doc_id": self.doc_id,
            "type": self.type,
            "page": self.page if self.page is not None else -1,
            "section": self.section or "",
            "caption": self.caption or "",
            "bbox": json.dumps(self.bbox) if self.bbox else "",
            "artifact_path": self.artifact_path or "",
        }
        extras: Dict[str, Any] = {}
        for key, value in (self.metadata or {}).items():
            if key in core or value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                extras[key] = value
            else:
                extras[key] = json.dumps(value, ensure_ascii=False)
        extras.update(core)
        return extras

    @classmethod
    def from_chroma_result(
        cls,
        id_: str,
        document: str,
        metadata: Dict[str, Any],
    ) -> "Evidence":
        """Reconstruct an Evidence from a raw ChromaDB query hit."""
        bbox: Optional[List[float]] = None
        if metadata.get("bbox"):
            try:
                bbox = json.loads(metadata["bbox"])
            except (ValueError, TypeError):
                pass

        page = metadata.get("page")
        if page == -1:
            page = None

        return cls(
            # The vector-store key is the uid; the parser id lives in metadata.
            id=metadata.get("id") or id_,
            doc_id=metadata.get("doc_id", ""),
            type=metadata.get("type", "text"),
            content=document,
            page=page,
            section=metadata.get("section") or None,
            bbox=bbox,
            caption=metadata.get("caption") or None,
            artifact_path=metadata.get("artifact_path") or None,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single ranked result returned to the caller.

    The retrieval layer returns evidence, NOT a generated answer.
    The answer-generation layer consumes these results.
    """

    id: str
    doc_id: str
    type: str
    score: float

    content: str
    page: Optional[int] = None
    section: Optional[str] = None
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    artifact_path: Optional[str] = None

    # Neighbouring text chunks attached during evidence expansion.
    context_chunks: List[str] = field(default_factory=list)

    # ---- fields added for entity-aware retrieval (all optional) ----------
    uid: str = ""
    """Doc-qualified unique key ('<doc_id>::<id>').  Empty for legacy callers."""

    score_details: Dict[str, Any] = field(default_factory=dict)
    """Per-stage ranking information: dense_rank, bm25_rank, rrf_score,
    rerank_score (raw CrossEncoder logit), base_rank, tier, adjustments.
    None of these values is a probability of correctness."""

    assessment: Optional[Dict[str, Any]] = None
    """Method-aware evidence assessment (see evidence_validator.py)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Derived / structured metadata (table headers & cells, figure label,
    detected entities, evidence role …)."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "uid": self.uid or make_uid(self.doc_id, self.id),
            "doc_id": self.doc_id,
            "type": self.type,
            "score": round(self.score, 6),
            "content": self.content,
            "page": self.page,
            "section": self.section,
            "bbox": self.bbox,
            "caption": self.caption,
            "artifact_path": self.artifact_path,
            "context_chunks": self.context_chunks,
            "score_details": self.score_details,
            "assessment": self.assessment,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# RetrievalResponse
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResponse:
    """
    The public contract of the retrieval layer.

    query → relevant evidence

    The answer-generation layer receives this object and builds
    a natural-language answer; the UI layer uses artifact_path
    to display the actual visual evidence.
    """

    query: str
    results: List[RetrievalResult]

    query_analysis: Optional[Any] = None
    """query_understanding.QueryAnalysis (or None for legacy callers)."""

    validation: Optional[Any] = None
    """evidence_validator.ValidationResult: whether the retrieved evidence
    establishes what was asked.  This is an EVIDENCE check, not a verified
    answer and not an answer-confidence score."""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
        }
        if self.query_analysis is not None:
            d["query_analysis"] = self.query_analysis.to_dict()
        if self.validation is not None:
            d["validation"] = self.validation.to_dict()
        return d
