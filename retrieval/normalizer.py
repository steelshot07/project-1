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

import json
import os
import re
from typing import Any, Dict, List, Optional

from citation_detector import FIGURE_PATTERN, TABLE_PATTERN
from evidence_signals import derive_metadata, is_backmatter_heading
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
       Duplicate block_ids are emitted once.
    """
    # Map block_id → section title
    block_to_section: Dict[str, str] = {}
    for section in doc.get("sections", []):
        title = section.get("title", "")
        for block in section.get("blocks", []):
            bid = block.get("block_id", "")
            if bid:
                block_to_section[bid] = title

    # Text is taken from `pages` ONLY; `sections` contribute titles, never
    # text, so a block present in both representations is indexed once.
    # The seen-set additionally guards against a parser emitting the same
    # block twice within `pages`.
    ordered: List[Dict[str, Any]] = []
    seen: set = set()
    for page in doc.get("pages", []):
        page_num = page.get("page_number")
        for block in page.get("blocks", []):
            text = block.get("text", "").strip()
            if not text or len(text) < 5:
                continue
            bid = block.get("block_id", "")
            key = bid or (page_num, text)
            if key in seen:
                continue
            seen.add(key)
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
        content = "\n\n".join(texts)
        metadata: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "block_ids": ",".join(b["block_id"] for b in block_list),
            "start_page": min(pages) if pages else None,
            "end_page": max(pages) if pages else None,
            "source_type": "parser_text_blocks",
        }
        metadata.update(derive_metadata("text", content, None, section))
        return Evidence(
            id=chunk_id,
            doc_id=doc_id,
            type="text",
            content=content,
            page=page,
            section=section or None,
            bbox=None,
            caption=None,
            artifact_path=None,
            metadata=metadata,
        )

    for block in blocks:
        tok = _estimate_tokens(block["text"])
        # Never merge prose with the bibliography: a References /
        # Bibliography / Appendix heading always opens a fresh chunk
        # (no overlap), so reference lists form their own chunks.
        if current and is_backmatter_heading(block["text"]):
            chunks.append(_flush(current, chunk_number))
            chunk_number += 1
            current, current_tokens = [], 0
        elif current and current_tokens + tok > chunk_size:
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


def _table_structure(table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structured table fields kept alongside the flattened content so that
    downstream consumers can render the table and keep each value attached
    to its parameter name / column header.

    table_headers : cells of the first non-empty row (the parser does not
                    mark header rows, so this is a best-effort guess and is
                    labelled as such by `table_header_source`)
    row_labels    : first non-empty cell of every following row
    table_data    : all rows, cells stripped (JSON string)
    """
    rows = [[str(c).strip() for c in row] for row in table.get("data", []) or []]
    non_empty = [r for r in rows if any(r)]
    headers = [c for c in non_empty[0] if c] if non_empty else []
    row_labels = [next(c for c in r if c) for r in non_empty[1:]]
    return {
        "table_headers": json.dumps(headers, ensure_ascii=False),
        "table_header_source": "first_non_empty_row",
        "row_labels": json.dumps(row_labels, ensure_ascii=False),
        "table_data": json.dumps(rows, ensure_ascii=False),
        "n_rows": len(non_empty),
        "n_cols": max((len(r) for r in rows), default=0),
    }


def _label_number(caption: Optional[str], pattern: re.Pattern) -> Optional[str]:
    """'Figure 8: Vehicle travel times…' -> '8'  (only at caption start)."""
    if not caption:
        return None
    m = pattern.match(caption.strip())
    return m.group(1) if m else None


def _citing_blocks(
    number: Optional[str],
    pattern: re.Pattern,
    blocks: List[Dict[str, Any]],
    max_blocks: int = 2,
    max_chars: int = 300,
) -> List[Dict[str, Any]]:
    """
    Text blocks that refer to 'Figure N' / 'Table N', excluding the caption
    block itself ('Figure N: …').  Used as surrounding-text context.
    """
    if not number:
        return []
    caption_start = re.compile(
        pattern.pattern + r"\s*[:.]", re.IGNORECASE
    )
    found: List[Dict[str, Any]] = []
    for b in blocks:
        text = b["text"]
        if number not in pattern.findall(text):
            continue
        m = caption_start.match(text.strip())
        if m and m.group(1) == number:
            continue  # this block IS the caption
        found.append({"block_id": b["block_id"], "page": b["page"],
                      "text": text[:max_chars]})
        if len(found) >= max_blocks:
            break
    return found


def _normalize_tables(
    doc_id: str,
    tables: List[Dict[str, Any]],
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> List[Evidence]:
    evidence = []
    for table in tables:
        tid = table.get("table_id", "")
        if not tid:
            continue
        content = _table_content(table)
        if not content.strip():
            continue
        caption = table.get("caption")
        metadata: Dict[str, Any] = {
            "detection_confidence": table.get("detection_confidence", 0.0),
            "source_id": tid,
            "source_type": "parser_table",
        }
        metadata.update(_table_structure(table))
        number = _label_number(caption, TABLE_PATTERN)
        if number:
            metadata["label"] = f"Table {number}"
            citing = _citing_blocks(number, TABLE_PATTERN, blocks or [])
            if citing:
                metadata["referenced_by"] = ",".join(c["block_id"] for c in citing)
                metadata["context_text"] = " … ".join(c["text"] for c in citing)
        metadata.update(derive_metadata("table", content, caption))
        evidence.append(
            Evidence(
                id=tid,
                doc_id=doc_id,
                type="table",
                content=content,
                page=table.get("page"),
                section=None,
                bbox=table.get("bbox"),
                caption=caption,
                artifact_path=_normalize_path(table.get("table_path")),
                metadata=metadata,
            )
        )
    return evidence


# ---------------------------------------------------------------------------
# Figure normalisation
# ---------------------------------------------------------------------------

def _figure_content(
    figure: Dict[str, Any],
    citing: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build a searchable text representation of a figure.

    caption + (optional) the body-text sentences that cite the figure.
    Image pixels are NOT embedded: figure retrieval is text-based.
    """
    caption = (figure.get("caption") or "").strip()
    if citing:
        ctx = " … ".join(c["text"] for c in citing)
        return f"{caption}\nReferenced in text: {ctx}" if caption else ctx
    return caption


def _normalize_figures(
    doc_id: str,
    figures: List[Dict[str, Any]],
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> List[Evidence]:
    evidence = []
    for fig in figures:
        fid = fig.get("figure_id", "")
        if not fid:
            continue
        caption = fig.get("caption")
        number = _label_number(caption, FIGURE_PATTERN)
        citing = _citing_blocks(number, FIGURE_PATTERN, blocks or [])
        content = _figure_content(fig, citing)
        if not content.strip():
            content = f"Figure on page {fig.get('page', '?')}"
        metadata: Dict[str, Any] = {
            "detection_confidence": fig.get("detection_confidence", 0.0),
            "source_id": fid,
            "source_type": "parser_figure",
            "content_basis": "caption_and_citing_text" if citing else "caption_only",
        }
        if number:
            metadata["label"] = f"Figure {number}"
        if citing:
            metadata["referenced_by"] = ",".join(c["block_id"] for c in citing)
            metadata["context_text"] = " … ".join(c["text"] for c in citing)
        metadata.update(derive_metadata("figure", content, caption))
        evidence.append(
            Evidence(
                id=fid,
                doc_id=doc_id,
                type="figure",
                content=content,
                page=fig.get("page"),
                section=None,
                bbox=fig.get("bbox"),
                caption=caption,
                artifact_path=_normalize_path(fig.get("image_path")),
                metadata=metadata,
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
    table_evidence = _normalize_tables(doc_id, doc.get("tables", []), blocks)

    # Figures
    figure_evidence = _normalize_figures(doc_id, doc.get("figures", []), blocks)

    all_evidence = text_evidence + table_evidence + figure_evidence

    print(
        f"[Normalizer] {doc_id}: "
        f"{len(text_evidence)} text chunks, "
        f"{len(table_evidence)} tables, "
        f"{len(figure_evidence)} figures "
        f"-> {len(all_evidence)} total evidence items"
    )

    return all_evidence
