"""
retrieval/normalizer.py

Converts a parsed document JSON (produced by parser/main.py) into a
flat list of Evidence objects ready for embedding and indexing.

Responsibilities:
- Extract text chunks from sections/pages (with sliding-window overlap)
- Extract one Evidence per table  (content = caption + cell text)
- Extract one Evidence per figure (content = caption)
- Preserve all provenance fields: doc_id, page, bbox, artifact_path, etc.
- Never import parser code; consumes its JSON output schema only.

NOT responsible for:
- Embedding
- Storing to a vector database
- Generating answers
"""

import os
import re
from typing import Any, Dict, List, Optional

from schema import Evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_PAGE_RE = re.compile(r"^p(\d+)_")


def _page_from_block_id(block_id: str) -> Optional[int]:
    """Derive page number from the parser's block_id convention 'p{N}_b{M}'."""
    m = _BLOCK_PAGE_RE.match(block_id)
    return int(m.group(1)) if m else None


def _normalize_path(path: Optional[str]) -> Optional[str]:
    """Convert Windows backslash paths to forward slashes."""
    if not path:
        return None
    return path.replace("\\", "/")


def _estimate_tokens(text: str) -> int:
    """Rough word-count proxy for token length."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _collect_ordered_blocks(
    doc: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Walk the parser JSON and return blocks in document order,
    each annotated with 'page' and 'section'.

    Strategy:
    1. Build block_id → section_title mapping from sections.
    2. Walk pages in order; emit each block with its section context.
    """
    # Map block_id → section title
    block_to_section: Dict[str, str] = {}
    for section in doc.get("sections", []):
        title = section.get("title", "")
        for block in section.get("blocks", []):
            bid = block.get("block_id", "")
            if bid:
                block_to_section[bid] = title

    ordered: List[Dict[str, Any]] = []
    for page in doc.get("pages", []):
        page_num = page.get("page_number")
        for block in page.get("blocks", []):
            text = block.get("text", "").strip()
            if not text or len(text) < 5:
                continue
            bid = block.get("block_id", "")
            ordered.append(
                {
                    "block_id": bid,
                    "text": text,
                    "page": page_num,
                    "section": block_to_section.get(bid, ""),
                    "type": block.get("type", "paragraph"),
                    "bbox": block.get("bbox"),
                }
            )
    return ordered


def _sliding_window_chunks(
    doc_id: str,
    blocks: List[Dict[str, Any]],
    chunk_size: int = 450,
    overlap: int = 100,
) -> List[Evidence]:
    """
    Produce overlapping text chunks from ordered blocks.
    Each chunk records the section of its first constituent block.
    """
    chunks: List[Evidence] = []
    chunk_number = 0
    current: List[Dict[str, Any]] = []
    current_tokens = 0

    def _flush(block_list: List[Dict[str, Any]], n: int) -> Evidence:
        texts = [b["text"] for b in block_list]
        pages = [b["page"] for b in block_list if b.get("page") is not None]
        section = block_list[0].get("section") if block_list else None
        page = min(pages) if pages else None
        chunk_id = f"{doc_id}_chunk_{n}"
        return Evidence(
            id=chunk_id,
            doc_id=doc_id,
            type="text",
            content="\n\n".join(texts),
            page=page,
            section=section or None,
            bbox=None,
            caption=None,
            artifact_path=None,
            metadata={
                "chunk_id": chunk_id,
                "block_ids": ",".join(b["block_id"] for b in block_list),
                "start_page": min(pages) if pages else None,
                "end_page": max(pages) if pages else None,
            },
        )

    for block in blocks:
        tok = _estimate_tokens(block["text"])
        if current and current_tokens + tok > chunk_size:
            chunks.append(_flush(current, chunk_number))
            chunk_number += 1
            # Overlap: keep trailing blocks whose tokens fit within overlap budget
            overlap_blocks: List[Dict[str, Any]] = []
            overlap_tokens = 0
            for prev in reversed(current):
                pt = _estimate_tokens(prev["text"])
                if overlap_tokens + pt > overlap:
                    break
                overlap_blocks.insert(0, prev)
                overlap_tokens += pt
            current = overlap_blocks
            current_tokens = overlap_tokens

        current.append(block)
        current_tokens += tok

    if current:
        chunks.append(_flush(current, chunk_number))

    return chunks


# ---------------------------------------------------------------------------
# Table normalisation
# ---------------------------------------------------------------------------

def _table_content(table: Dict[str, Any]) -> str:
    """
    Build a searchable text representation of a table.
    Includes caption, then rows joined as pipe-separated lines.
    Empty cells are dropped from each row.
    """
    parts: List[str] = []
    caption = table.get("caption", "")
    if caption:
        parts.append(caption)

    data = table.get("data", [])
    for row in data:
        cells = [str(c).strip() for c in row if str(c).strip()]
        if cells:
            parts.append(" | ".join(cells))

    return "\n".join(parts)


def _normalize_tables(
    doc_id: str,
    tables: List[Dict[str, Any]],
) -> List[Evidence]:
    evidence = []
    for table in tables:
        tid = table.get("table_id", "")
        if not tid:
            continue
        content = _table_content(table)
        if not content.strip():
            continue
        evidence.append(
            Evidence(
                id=tid,
                doc_id=doc_id,
                type="table",
                content=content,
                page=table.get("page"),
                section=None,
                bbox=table.get("bbox"),
                caption=table.get("caption"),
                artifact_path=_normalize_path(table.get("table_path")),
                metadata={
                    "detection_confidence": table.get(
                        "detection_confidence", 0.0
                    ),
                },
            )
        )
    return evidence


# ---------------------------------------------------------------------------
# Figure normalisation
# ---------------------------------------------------------------------------

def _figure_content(figure: Dict[str, Any]) -> str:
    """
    Build a searchable text representation of a figure.
    Currently: caption text only.
    Can be extended with OCR or surrounding-text context.
    """
    caption = figure.get("caption", "")
    return caption.strip() if caption else ""


def _normalize_figures(
    doc_id: str,
    figures: List[Dict[str, Any]],
) -> List[Evidence]:
    evidence = []
    for fig in figures:
        fid = fig.get("figure_id", "")
        if not fid:
            continue
        content = _figure_content(fig)
        if not content.strip():
            content = f"Figure on page {fig.get('page', '?')}"
        evidence.append(
            Evidence(
                id=fid,
                doc_id=doc_id,
                type="figure",
                content=content,
                page=fig.get("page"),
                section=None,
                bbox=fig.get("bbox"),
                caption=fig.get("caption"),
                artifact_path=_normalize_path(fig.get("image_path")),
                metadata={
                    "detection_confidence": fig.get(
                        "detection_confidence", 0.0
                    ),
                },
            )
        )
    return evidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_document(
    doc: Dict[str, Any],
    chunk_size: int = 450,
    overlap: int = 100,
) -> List[Evidence]:
    """
    Convert a parsed document dict into a flat list of Evidence objects.

    Parameters
    ----------
    doc:
        Parsed document loaded from a *_parsed.json file.
    chunk_size:
        Target chunk size in word-count tokens.
    overlap:
        Overlap budget in word-count tokens between consecutive chunks.

    Returns
    -------
    List of Evidence objects covering text chunks, tables, and figures.
    """
    doc_id: str = doc.get("doc_id", "unknown")

    # Text
    blocks = _collect_ordered_blocks(doc)
    text_evidence = _sliding_window_chunks(
        doc_id, blocks, chunk_size=chunk_size, overlap=overlap
    )

    # Tables
    table_evidence = _normalize_tables(doc_id, doc.get("tables", []))

    # Figures
    figure_evidence = _normalize_figures(doc_id, doc.get("figures", []))

    all_evidence = text_evidence + table_evidence + figure_evidence

    print(
        f"[Normalizer] {doc_id}: "
        f"{len(text_evidence)} text chunks, "
        f"{len(table_evidence)} tables, "
        f"{len(figure_evidence)} figures "
        f"-> {len(all_evidence)} total evidence items"
    )

    return all_evidence
