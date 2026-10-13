"""
retrieval/query_understanding.py

Lightweight, deterministic query understanding.  No LLM calls.

Given a natural-language question, produce a ``QueryAnalysis`` that records:

- information_need : hyperparameters | performance_results | architecture |
                     training_procedure | dataset | general
- entities         : method / algorithm names (see entities.py)
- relation         : property_of | comparison | mention | general
- property_terms   : stemmed content words describing WHAT is asked
                     (e.g. ['travel', 'time'])
- requested_modalities : ['figure'] / ['table'] when explicitly asked for
- doc_id           : document filter supplied by the caller

The ORIGINAL query is always preserved and still used for dense and BM25
retrieval.  The analysis only supplements retrieval (an extra
entity-focused lexical list) and drives evidence validation.

Unknown queries degrade gracefully: no entities -> relation 'general',
information_need 'general', and validation falls back to topical
term coverage.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from entities import find_entities, kind_of


# ---------------------------------------------------------------------------
# Shared text helpers (also used by evidence_validator.py)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\[])|\n+")

STOPWORDS = {
    "a", "an", "the", "of", "for", "in", "on", "at", "to", "by", "with",
    "from", "and", "or", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "how", "why", "when", "where", "does",
    "do", "did", "this", "that", "these", "those", "it", "its", "their",
    "there", "they", "them", "as", "than", "into", "about", "any", "all",
    "some", "such", "can", "could", "would", "should", "will", "has", "have",
    "had", "used", "use", "using", "me", "i", "we", "our", "you", "your",
    "paper", "study", "work", "author", "authors", "give", "tell", "list",
    "describe", "explain", "find", "show", "shown", "display", "please",
    "value", "values", "reported", "report", "provide", "provided", "each",
    "between", "versus", "vs", "compare", "compared", "comparing",
    "comparison", "against", "difference", "differences", "better", "worse",
    "figure", "fig", "figures", "table", "tables", "plot", "graph", "chart",
    "diagram", "image", "picture", "method", "methods", "approach",
    "approaches", "algorithm", "algorithms", "agent", "agents", "model",
    "models", "proposed",
}


def stem(word: str) -> str:
    """Very light, deterministic plural stripping (no external stemmer)."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        return w[:-1]
    return w


def terms(text: str) -> List[str]:
    """Stemmed, lower-cased word tokens (stopwords kept)."""
    return [stem(w) for w in _WORD_RE.findall(text or "")]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


# ---------------------------------------------------------------------------
# Information-need lexicon
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NeedSpec:
    name: str
    query_cues: str
    """Regex that identifies this need in a query."""

    evidence_cues: str
    """Regex that identifies this need in evidence text."""


NEEDS: Sequence[NeedSpec] = (
    NeedSpec(
        "hyperparameters",
        r"hyper[\s-]?param\w*|(?:key|training)\s+parameters|parameter\s+settings?"
        r"|learning\s+rates?|batch\s+sizes?|discount\s+factor",
        r"hyper[\s-]?param\w*|\bparameters?\b|learning\s+rate|\blr\b|batch[\s_]size"
        r"|replay\s+buffer|discount\s+factor|\bgamma\b|exploration\s+rate|epsilon"
        r"|optimi[sz]er|\bepochs?\b|freez\w*\s+interval|target_update|\btau\b",
    ),
    NeedSpec(
        "architecture",
        r"architecture|network\s+structure|structure\s+of|layers?\b|topology",
        r"architecture|structure|layers?\b|neurons|fully\s+connected|convolution\w*",
    ),
    NeedSpec(
        "training_procedure",
        r"training\s+(?:procedure|process|steps?|loop)|how\s+(?:was|is|were)\s+.*trained"
        r"|pseudo-?code",
        r"train\w*|algorithm|procedure|replay|episode",
    ),
    NeedSpec(
        "dataset",
        r"\bdata\s*sets?\b|benchmarks?|corpus",
        r"\bdata\s*sets?\b|benchmarks?|corpus|samples|cases",
    ),
    NeedSpec(
        "performance_results",
        r"performance|results?\b|accuracy|travel\s+times?|rewards?|scores?|errors?"
        r"|outperform\w*|improve\w*|reduc\w*|delay|throughput|queue",
        r"",  # matched through property terms instead
    ),
)

_COMPARISON_RE = re.compile(
    r"\b(compar\w*|versus|vs\.?|against|differ\w*|better\s+than|worse\s+than"
    r"|outperform\w*|between\b.+\band\b)",
    re.IGNORECASE,
)
_FIGURE_RE = re.compile(
    r"\b(figures?|fig\.?|plots?|graphs?|charts?|diagrams?|images?|"
    r"visuali[sz]ations?|pictures?|illustrat\w*|curves?)\b",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(r"\b(tables?|tabular|tabulated)\b", re.IGNORECASE)
_REFERENCE_QUERY_RE = re.compile(
    r"\b(references?|citations?|cited|cite|bibliograph\w*|related\s+work)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# QueryAnalysis
# ---------------------------------------------------------------------------

@dataclass
class QueryAnalysis:
    original_query: str
    information_need: str = "general"
    relation: str = "general"
    entities: List[str] = field(default_factory=list)
    """Canonical method names explicitly present in the query."""

    entity_kinds: Dict[str, str] = field(default_factory=dict)
    """canonical -> 'method' | 'category'."""

    entity_sources: Dict[str, str] = field(default_factory=dict)
    """canonical -> 'registry' | 'pattern'."""

    property_terms: List[str] = field(default_factory=list)
    """Stemmed content terms describing the requested property/topic."""

    need_evidence_cues: str = ""
    """Regex describing the information need in evidence ('' = none)."""

    requested_modalities: List[str] = field(default_factory=list)
    asks_about_references: bool = False
    doc_id: Optional[str] = None

    @property
    def is_method_specific(self) -> bool:
        return bool(self.entities)

    @property
    def is_comparison(self) -> bool:
        return self.relation == "comparison"

    def focused_query(self) -> str:
        """
        Entity-focused lexical query: requested names (with explicit
        expansions) + property terms.  Used as an ADDITIONAL BM25 list;
        never replaces the original query.
        """
        parts: List[str] = []
        for ent in self.entities:
            parts.append(ent)
        parts.extend(self.property_terms)
        if self.information_need == "hyperparameters":
            parts.extend(["hyperparameters", "parameters"])
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_method_specific"] = self.is_method_specific
        return d


def analyze_query(query: str, doc_id: Optional[str] = None) -> QueryAnalysis:
    """Deterministically analyse a query.  Never raises on unknown input."""
    q = query or ""
    mentions = find_entities(q)
    entities = []
    sources: Dict[str, str] = {}
    for m in mentions:
        if m.canonical not in entities:
            entities.append(m.canonical)
            sources[m.canonical] = m.source

    # Information need: first matching spec in priority order.
    need = "general"
    need_cues = ""
    for spec in NEEDS:
        if re.search(spec.query_cues, q, re.IGNORECASE):
            need = spec.name
            need_cues = spec.evidence_cues
            break

    # Property terms: content words that are not entity tokens.
    entity_spans = [(m.start, m.end) for m in mentions]
    prop: List[str] = []
    for w in _WORD_RE.finditer(q):
        if any(s <= w.start() < e for s, e in entity_spans):
            continue
        raw = w.group(0).lower()
        t = stem(raw)
        if raw in STOPWORDS or t in STOPWORDS or len(t) < 2 or t in prop:
            continue
        prop.append(t)

    is_comparison = bool(_COMPARISON_RE.search(q))
    if is_comparison and (len(entities) >= 2 or not entities):
        relation = "comparison" if len(entities) >= 2 else "general"
    elif entities and need != "general":
        relation = "property_of"
    elif entities and prop:
        relation = "property_of"
    elif entities:
        relation = "mention"
    else:
        relation = "general"
    # A comparison with one named entity ("how does SAC compare?") is a
    # property_of query about that entity's results.
    if is_comparison and len(entities) == 1:
        relation = "property_of"

    modalities: List[str] = []
    if _FIGURE_RE.search(q):
        modalities.append("figure")
    if _TABLE_RE.search(q):
        modalities.append("table")

    return QueryAnalysis(
        original_query=query,
        information_need=need,
        relation=relation,
        entities=entities,
        entity_kinds={e: kind_of(e) for e in entities},
        entity_sources=sources,
        property_terms=prop,
        need_evidence_cues=need_cues,
        requested_modalities=modalities,
        asks_about_references=bool(_REFERENCE_QUERY_RE.search(q)),
        doc_id=doc_id,
    )
