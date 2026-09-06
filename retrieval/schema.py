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

@dataclass
class Evidence:
    """A single retrievable item from any modality."""

    id: str
    """Stable, globally unique identifier.
    Examples: 'table_p10_1', 'fig_p3_0', '2012.12104v1_chunk_4'
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

    # -------------------------------------------------------------------
    # ChromaDB serialisation
    # -------------------------------------------------------------------

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """
        Flatten to a ChromaDB-compatible metadata dict.
        ChromaDB only accepts str / int / float / bool values.
        bbox is serialised as a JSON string.
        """
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "type": self.type,
            "page": self.page if self.page is not None else -1,
            "section": self.section or "",
            "caption": self.caption or "",
            "bbox": json.dumps(self.bbox) if self.bbox else "",
            "artifact_path": self.artifact_path or "",
        }

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
            id=id_,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
        }
