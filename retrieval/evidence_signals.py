"""
retrieval/evidence_signals.py

Deterministic, traceable signals derived from an evidence item's own text.

- reference_density / is_reference_list : bibliography detection
- evidence_role                          : hyperparameters | performance_results |
                                           architecture | algorithm | references |
                                           unknown
- derive_metadata                        : the derived fields written by the
                                           normaliser into Evidence.metadata

Every derived value is computed from the evidence's caption / content /
section title only, so it can be traced back to the source document.
When no rule fires, the role is 'unknown' — nothing is guessed.
"""

import json
import re
from typing import Any, Dict, List, Optional

from entities import entity_names, kind_of


# ---------------------------------------------------------------------------
# Bibliography detection
# ---------------------------------------------------------------------------

_REF_MARKERS = [
    re.compile(r"\[\d{1,3}\]"),                                    # [12]
    re.compile(r"(?:^|\s)\d{1,3}\.\s+[A-Z][a-zA-Z'\-]+,\s+[A-Z]\."),  # 1. Smith, A.
    re.compile(r"\bet\s+al\.?"),
    re.compile(r"\b(?:19|20)\d{2}[a-z]?\b"),                       # years
    re.compile(r"\b(?:vol|no|pp)\.\s*\d+", re.IGNORECASE),
    re.compile(r"\bp\.\s*\d+\s*[-–]\s*\d+"),                  # p. 58-67
    re.compile(r"\b(?:Proc\.|Proceedings|Conference|Journal|Transactions|"
               r"arXiv|Press|Workshop|Symposium|Research\s+Record)\b"),
    re.compile(r"\b\d+\s*\(\d+\)\s*:"),                            # 1320 (1):
]
_REF_SECTION_RE = re.compile(r"^\s*(?:\d+\.?\s*)?(r\s?eferences|bibliography|works\s+cited)\b",
                             re.IGNORECASE)
# Author names as they appear in bibliographies: 'J. Duan', 'A. K. Jain',
# 'Srivastava, A.'.  Distinguishes a reference list from a short technical
# table that merely cites a few works ('FaceNet [30] | 96.3 %').
_AUTHOR_RE = re.compile(
    r"\b[A-Z]\.\s?(?:[A-Z]\.\s?)*[A-Z][a-z]{2,}|\b[A-Z][a-z]+,\s+[A-Z]\.(?:\s?[A-Z]\.)*"
)


_BACKMATTER_RE = re.compile(r"^\s*(?:[A-Z]\.?\s*|\d+\.?\s*)?(appendix|appendices)\b\s*:?",
                            re.IGNORECASE)


def is_reference_heading(text: str) -> bool:
    """True for a short block that is a References / Bibliography heading."""
    return len(text.split()) <= 3 and bool(_REF_SECTION_RE.match(text.strip()))


def is_backmatter_heading(text: str) -> bool:
    """References / Bibliography / Appendix heading (short block)."""
    t = text.strip()
    return len(t.split()) <= 3 and bool(_REF_SECTION_RE.match(t) or _BACKMATTER_RE.match(t))

REFERENCE_DENSITY_THRESHOLD = 6.0
"""Bibliographic markers per 100 words above which a chunk is treated as a
reference list.  Calibrated on the indexed papers: ordinary prose with
inline citations scores < 3, reference sections score > 10."""


def reference_density(text: str) -> float:
    words = len(text.split())
    if words == 0:
        return 0.0
    hits = sum(len(p.findall(text)) for p in _REF_MARKERS)
    return 100.0 * hits / words


def is_reference_list(text: str, section: Optional[str] = None) -> bool:
    """
    True when the item is dominated by bibliography entries.

    Rules:
    - inside a References section: marker density >= threshold / 2
    - elsewhere: density >= threshold AND at least 2 author-name patterns
    Ordinary prose with inline citations and short tables that cite a few
    works are NOT flagged.
    """
    if len(text.split()) < 12:
        return False
    density = reference_density(text)
    if section and _REF_SECTION_RE.match(section):
        return density >= REFERENCE_DENSITY_THRESHOLD / 2
    return (density >= REFERENCE_DENSITY_THRESHOLD
            and len(_AUTHOR_RE.findall(text)) >= 2)


# ---------------------------------------------------------------------------
# Evidence role
# ---------------------------------------------------------------------------

_ROLE_RULES = [
    ("hyperparameters", re.compile(
        r"hyper[\s-]?param\w*|key\s+parameters|training\s+parameters|"
        r"parameter\s+settings?|^\s*(?:hyper)?parameters?\s*\|", re.IGNORECASE | re.MULTILINE)),
    ("algorithm", re.compile(r"\balgorithm\s+\d+|pseudo-?code", re.IGNORECASE)),
    ("architecture", re.compile(
        r"architecture|structure\s+of\s+the\s+(?:deep\s+)?(?:neural\s+)?network|network\s+structure",
        re.IGNORECASE)),
    ("performance_results", re.compile(
        r"performance|results?\b|travel\s+times?|accuracy|comparison|reward|"
        r"\bmean\b|error|queue\s+length|improvement", re.IGNORECASE)),
]


def evidence_role(
    ev_type: str,
    caption: Optional[str],
    content: str,
    section: Optional[str] = None,
    reference_list: bool = False,
) -> str:
    """
    Classify what the evidence item *is about*, from its caption (tables /
    figures) or its reference-list status (text).  Text chunks are only
    labelled 'references'; otherwise 'unknown', because a multi-paragraph
    chunk rarely has one role.
    """
    if reference_list:
        return "references"
    if ev_type in ("table", "figure"):
        probe = caption or ""
        if ev_type == "table" and not probe:
            # Fall back to the header row when the parser found no caption.
            probe = content.split("\n", 1)[0]
        for role, pat in _ROLE_RULES:
            if pat.search(probe):
                return role
    return "unknown"


# ---------------------------------------------------------------------------
# Metadata derivation (used by the normaliser)
# ---------------------------------------------------------------------------

def _join(values: List[str]) -> str:
    return "; ".join(values)


def derive_metadata(
    ev_type: str,
    content: str,
    caption: Optional[str] = None,
    section: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Derived, traceable metadata.  Values are ChromaDB-safe primitives
    (lists are '; '-joined strings).

    Keys
    ----
    entities          : registry method names detected anywhere in the content
    candidate_entities: method-SHAPED names not in the registry (lower
                        confidence; may include tools, venues, datasets)
    caption_entities  : registry method names detected in the caption only
    method            : set only when the caption names exactly ONE entity
                        and it is a specific method (not a category such as
                        DRL); omitted when uncertain
    evidence_role     : see evidence_role()
    is_reference_list : bool
    reference_density : float (markers per 100 words)
    metadata_source   : provenance of these derived fields
    """
    ref_list = is_reference_list(content, section)
    meta: Dict[str, Any] = {
        "entities": _join(entity_names(content, use_shape=False)),
        "candidate_entities": _join(
            [e for e in entity_names(content) if e not in entity_names(content, use_shape=False)]
        ),
        "evidence_role": evidence_role(ev_type, caption, content, section, ref_list),
        "is_reference_list": ref_list,
        "reference_density": round(reference_density(content), 2),
        "metadata_source": "rule-based-v1 (derived from caption/content/section)",
    }
    if caption:
        cap_ents = entity_names(caption, use_shape=False)
        meta["caption_entities"] = _join(cap_ents)
        if len(cap_ents) == 1 and kind_of(cap_ents[0]) == "method":
            meta["method"] = cap_ents[0]
    return meta


def split_list(value: Any) -> List[str]:
    """Inverse of _join for metadata read back from ChromaDB."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not value:
        return []
    return [v.strip() for v in str(value).split(";") if v.strip()]


def to_json_str(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
