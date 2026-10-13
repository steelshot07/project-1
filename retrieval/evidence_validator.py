"""
retrieval/evidence_validator.py

Method-aware evidence validation.  Deterministic; no LLM calls.

Question answered
-----------------
"Does the RETRIEVED evidence establish what the query asks — for the
method(s) the query names?"

It does NOT generate or verify an answer, and its status is not an
answer-confidence score.  It tells the generation layer which evidence it
may attribute to which method, and what it must NOT claim.

Per evidence item -> EvidenceAssessment.label
---------------------------------------------
SUPPORTS        the item explicitly associates the requested method(s) with
                the requested property (same sentence / caption / table
                header / table row-within-a-property-table)
PARTIAL         association only via the caption (body does not confirm),
                or only some of the compared methods are present
MENTIONS_ONLY   the requested method is named but not tied to the property
OTHER_METHOD    the item ties the property to a DIFFERENT method
                (e.g. 'Hyperparameters used for training SAC agents' for a
                DQN question).  Useful context; values must not be
                attributed to the requested method.
RELATED_CONTEXT topically related (property terms present), no method link
REFERENCE_LIST  bibliography entries (unless the query asks about references)
UNRELATED       none of the above

Aggregate -> ValidationResult.status
-----------------------------------
SUPPORTED              all requested methods are explicitly associated with
                       the property (for comparisons: in one evidence item)
PARTIALLY_SUPPORTED    some, caption-only, or spread across items
ENTITY_MISMATCH        property evidence exists, but for other methods;
                       at least one requested method is not established
INSUFFICIENT_EVIDENCE  retrieved evidence does not establish the answer
NOT_FOUND_IN_DOCUMENT  a document-level scan (all indexed evidence of the
                       document(s)) also found no association.  Only set by
                       apply_document_check(); never inferred from top-k.

Absence from the top-k results is reported as "not found in retrieved
evidence", never as absence from the paper.
"""

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from entities import find_entities, related, same_entity
from evidence_signals import is_reference_list
from query_understanding import QueryAnalysis, split_sentences, terms
from schema import make_uid


# Status values
SUPPORTED = "SUPPORTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
ENTITY_MISMATCH = "ENTITY_MISMATCH"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
NOT_FOUND_IN_DOCUMENT = "NOT_FOUND_IN_DOCUMENT"

# Per-evidence labels
SUPPORTS = "SUPPORTS"
PARTIAL = "PARTIAL"
MENTIONS_ONLY = "MENTIONS_ONLY"
OTHER_METHOD = "OTHER_METHOD"
RELATED_CONTEXT = "RELATED_CONTEXT"
REFERENCE_LIST = "REFERENCE_LIST"
UNRELATED = "UNRELATED"

_SCOPED_UNITS = ("caption", "caption+header")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvidenceAssessment:
    uid: str
    evidence_id: str
    doc_id: str
    type: str
    page: Optional[int]
    label: str
    requested_mentioned: List[str] = field(default_factory=list)
    requested_associated: Dict[str, str] = field(default_factory=dict)
    """requested canonical name -> basis ('sentence', 'caption', 'caption+header', 'row')"""

    other_associated: Dict[str, str] = field(default_factory=dict)
    """other method -> basis"""

    entities_detected: List[str] = field(default_factory=list)
    property_match: bool = False
    property_coverage: float = 0.0
    caption_only: bool = False
    is_reference_list: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    query: str
    status: str
    scope: str = "retrieved_evidence"
    relation: str = "general"
    information_need: str = "general"
    requested_entities: List[str] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)
    supported_entities: List[str] = field(default_factory=list)
    partially_supported_entities: List[str] = field(default_factory=list)
    mentioned_only_entities: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    """Requested names not even mentioned in the retrieved evidence."""

    mismatched_entities: Dict[str, List[str]] = field(default_factory=dict)
    """Other method -> uids of evidence that ties the property to it."""

    related_entity_notes: List[str] = field(default_factory=list)
    established_comparisons: List[List[str]] = field(default_factory=list)
    entity_mismatch: bool = False
    supporting_evidence_ids: List[str] = field(default_factory=list)
    partial_evidence_ids: List[str] = field(default_factory=list)
    mismatch_evidence_ids: List[str] = field(default_factory=list)
    mention_evidence_ids: List[str] = field(default_factory=list)
    reference_list_ids: List[str] = field(default_factory=list)
    documents_checked: List[str] = field(default_factory=list)
    document_findings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    explanation: str = ""
    caveats: List[str] = field(default_factory=list)
    per_evidence: List[EvidenceAssessment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_evidence"] = [a.to_dict() for a in self.per_evidence]
        return d


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _uid(ev: Any) -> str:
    uid = getattr(ev, "uid", "") or ""
    return uid or make_uid(ev.doc_id, ev.id)


def _required_matches(n: int) -> int:
    return n if n <= 2 else math.ceil(2 * n / 3)


def property_hit(text: str, analysis: QueryAnalysis) -> Tuple[bool, float]:
    """
    Does ``text`` talk about the requested property?

    - information-need cue (e.g. hyperparameter vocabulary) -> hit
    - otherwise property-term coverage: all terms when <= 2 terms,
      at least 2/3 of them otherwise.
    Returns (hit, coverage in [0, 1]).
    """
    if not text:
        return False, 0.0
    if analysis.need_evidence_cues and re.search(
        analysis.need_evidence_cues, text, re.IGNORECASE
    ):
        return True, 1.0
    pt = analysis.property_terms
    if not pt:
        return False, 0.0
    toks = set(terms(text))
    matched = sum(1 for t in pt if t in toks)
    return matched >= _required_matches(len(pt)), matched / len(pt)


def _units(ev: Any) -> Tuple[List[Tuple[str, str]], str]:
    """
    Split an evidence item into association units.

    Returns (units, body_text) where units are (kind, text) pairs and
    body_text is the content WITHOUT the caption (used for the
    caption-only check).
    """
    caption = (getattr(ev, "caption", None) or "").strip()
    content = ev.content or ""
    units: List[Tuple[str, str]] = []

    if ev.type == "table":
        lines = [ln for ln in content.split("\n") if ln.strip()]
        if caption and lines and lines[0].strip() == caption:
            lines = lines[1:]
        header = " | ".join(lines[:3])
        if caption:
            units.append(("caption", caption))
        units.append(("caption+header", f"{caption}\n{header}".strip()))
        units.extend(("row", ln) for ln in lines)
        return units, "\n".join(lines)

    body = content
    if caption:
        units.append(("caption", caption))
        if content.startswith(caption):
            body = content[len(caption):]
    units.extend(("sentence", s) for s in split_sentences(content))
    return units, body


def _unit_entities(text: str, kind: str, extra: Sequence[str]) -> List[str]:
    """
    Registry + query-supplied names everywhere; method-SHAPED (unregistered)
    names only in captions / table headers, where they are far less noisy.
    """
    use_shape = kind in _SCOPED_UNITS
    out: List[str] = []
    for m in find_entities(text, extra_names=extra, use_shape=use_shape):
        if m.canonical not in out:
            out.append(m.canonical)
    return out


def _match_requested(name: str, requested: Sequence[str]) -> Optional[str]:
    for r in requested:
        if same_entity(name, r):
            return r
    return None


# ---------------------------------------------------------------------------
# Per-evidence assessment
# ---------------------------------------------------------------------------

def assess_evidence(ev: Any, analysis: QueryAnalysis) -> EvidenceAssessment:
    requested = analysis.entities
    extra = [e for e in requested if analysis.entity_sources.get(e) != "registry"]

    units, body = _units(ev)
    full_text = "\n".join(
        [getattr(ev, "caption", None) or "", ev.content or ""]
    )
    meta = getattr(ev, "metadata", None) or {}
    ref_flag = meta.get("is_reference_list")
    if isinstance(ref_flag, str):
        ref_flag = ref_flag.lower() == "true"
    if ref_flag is None:
        ref_flag = is_reference_list(ev.content or "", getattr(ev, "section", None))

    prop_hit, coverage = property_hit(full_text, analysis)
    table_topic_hit = False
    if ev.type == "table":
        scoped = [t for k, t in units if k in _SCOPED_UNITS]
        table_topic_hit = any(property_hit(t, analysis)[0] for t in scoped)

    detected: List[str] = []
    req_mentioned: List[str] = []
    req_assoc: Dict[str, str] = {}
    other_assoc: Dict[str, str] = {}
    full_unit_coverage = False

    for kind, text in units:
        ents = _unit_entities(text, kind, extra)
        hit, cov = property_hit(text, analysis)
        if cov >= 1.0 and kind != "caption+header":
            full_unit_coverage = True
        for e in ents:
            if e not in detected:
                detected.append(e)
            r = _match_requested(e, requested)
            if r and r not in req_mentioned:
                req_mentioned.append(r)
        unit_hit = hit or (kind == "row" and table_topic_hit)
        if not unit_hit:
            continue
        for e in ents:
            r = _match_requested(e, requested)
            if r:
                req_assoc.setdefault(r, kind)
            else:
                other_assoc.setdefault(e, kind)

    # Caption-only association: a table caption claims the property but the
    # table body does not corroborate it (e.g. a mis-attached caption).
    # Only applicable when the information need has body vocabulary (e.g.
    # hyperparameter names).  For measured quantities ('travel times') a
    # results table body is numbers under the caption, so the caption is
    # the legitimate label of those numbers.
    caption_only = False
    if ev.type == "table" and req_assoc and analysis.need_evidence_cues:
        body_hit = property_hit(body, analysis)[0]
        caption_only = not body_hit

    notes: List[str] = []
    if ev.type == "figure":
        notes.append("Figure assessed from caption/citing text only; image pixels are not analysed.")
    if caption_only:
        notes.append("Association rests on the caption only; table body does not confirm the property.")
    for other in other_assoc:
        for r in requested:
            rel = related(r, other)
            if rel:
                notes.append(f"{other} is not treated as {r}: {rel}.")

    label = _label(analysis, ev, ref_flag, req_mentioned, req_assoc, other_assoc,
                   prop_hit, caption_only, full_unit_coverage)

    return EvidenceAssessment(
        uid=_uid(ev),
        evidence_id=ev.id,
        doc_id=ev.doc_id,
        type=ev.type,
        page=getattr(ev, "page", None),
        label=label,
        requested_mentioned=req_mentioned,
        requested_associated=req_assoc,
        other_associated=other_assoc,
        entities_detected=detected,
        property_match=prop_hit,
        property_coverage=round(coverage, 3),
        caption_only=caption_only,
        is_reference_list=bool(ref_flag),
        notes=notes,
    )


def _label(
    analysis: QueryAnalysis,
    ev: Any,
    ref_flag: bool,
    req_mentioned: List[str],
    req_assoc: Dict[str, str],
    other_assoc: Dict[str, str],
    prop_hit: bool,
    caption_only: bool,
    full_unit_coverage: bool,
) -> str:
    if ref_flag and not analysis.asks_about_references:
        return REFERENCE_LIST

    requested = analysis.entities
    if not requested:
        if full_unit_coverage and (analysis.property_terms or analysis.need_evidence_cues):
            return SUPPORTS
        return RELATED_CONTEXT if prop_hit else UNRELATED

    if analysis.relation == "mention":
        return MENTIONS_ONLY if req_mentioned else UNRELATED

    if req_assoc:
        all_req = all(r in req_assoc for r in requested)
        if analysis.relation == "comparison":
            return SUPPORTS if all_req and not caption_only else PARTIAL
        return SUPPORTS if all_req and not caption_only else PARTIAL
    if other_assoc and prop_hit:
        return OTHER_METHOD
    if req_mentioned:
        return MENTIONS_ONLY
    return RELATED_CONTEXT if prop_hit else UNRELATED


# ---------------------------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------------------------

def validate(
    analysis: QueryAnalysis,
    evidence: Iterable[Any],
    assessments: Optional[Dict[str, EvidenceAssessment]] = None,
) -> ValidationResult:
    """
    Validate a set of retrieved evidence items (Evidence or RetrievalResult)
    against the analysed query.  ``assessments`` (uid -> assessment) may be
    supplied to avoid recomputation.
    """
    items = list(evidence)
    per: List[EvidenceAssessment] = []
    for ev in items:
        uid = _uid(ev)
        a = (assessments or {}).get(uid) or assess_evidence(ev, analysis)
        per.append(a)

    requested = analysis.entities
    supported: List[str] = []
    partial: List[str] = []
    mentioned: Set[str] = set()
    alternatives: Dict[str, List[str]] = {}
    comparisons: List[List[str]] = []
    detected: List[str] = []
    notes: List[str] = []
    ids: Dict[str, List[str]] = {k: [] for k in (SUPPORTS, PARTIAL, OTHER_METHOD,
                                                 MENTIONS_ONLY, REFERENCE_LIST)}

    for a in per:
        if a.label in ids:
            ids[a.label].append(a.uid)
        for e in a.entities_detected:
            if e not in detected:
                detected.append(e)
        if a.label == REFERENCE_LIST:
            continue
        mentioned.update(a.requested_mentioned)
        for r in a.requested_associated:
            target = partial if a.caption_only else supported
            if r not in target:
                target.append(r)
        if a.property_match or a.requested_associated:
            for o in a.other_associated:
                alternatives.setdefault(o, [])
                if a.uid not in alternatives[o]:
                    alternatives[o].append(a.uid)
        linked = list(a.requested_associated) + list(a.other_associated)
        if len(linked) >= 2:
            pair = sorted(linked)
            if pair not in comparisons:
                comparisons.append(pair)
        for n in a.notes:
            if " is not treated as " in n and n not in notes:
                notes.append(n)

    partial = [p for p in partial if p not in supported]
    unassociated = [r for r in requested if r not in supported and r not in partial]
    mentioned_only = [r for r in unassociated if r in mentioned]
    missing = [r for r in unassociated if r not in mentioned]

    status = _status(analysis, per, supported, partial, unassociated, alternatives)
    entity_mismatch = bool(requested and unassociated and alternatives)

    result = ValidationResult(
        query=analysis.original_query,
        status=status,
        relation=analysis.relation,
        information_need=analysis.information_need,
        requested_entities=list(requested),
        detected_entities=detected,
        supported_entities=supported,
        partially_supported_entities=partial,
        mentioned_only_entities=mentioned_only,
        missing_entities=missing,
        mismatched_entities=alternatives if requested else {},
        related_entity_notes=notes,
        established_comparisons=comparisons if requested else [],
        entity_mismatch=entity_mismatch,
        supporting_evidence_ids=ids[SUPPORTS],
        partial_evidence_ids=ids[PARTIAL],
        mismatch_evidence_ids=ids[OTHER_METHOD],
        mention_evidence_ids=ids[MENTIONS_ONLY],
        reference_list_ids=ids[REFERENCE_LIST],
        caveats=[
            "Status describes the retrieved evidence only; it is not a "
            "generated or verified answer and not an answer-confidence score.",
        ],
        per_evidence=per,
    )
    if status != NOT_FOUND_IN_DOCUMENT:
        result.caveats.append(
            "A method not found in the retrieved evidence may still appear "
            "elsewhere in the document; run a document-level check to confirm absence."
        )
    result.explanation = _explain(result, analysis)
    return result


def _status(
    analysis: QueryAnalysis,
    per: List[EvidenceAssessment],
    supported: List[str],
    partial: List[str],
    unassociated: List[str],
    alternatives: Dict[str, List[str]],
) -> str:
    labels = {a.label for a in per}
    requested = analysis.entities

    if not requested:
        if SUPPORTS in labels:
            return SUPPORTED
        if RELATED_CONTEXT in labels:
            return PARTIALLY_SUPPORTED
        return INSUFFICIENT_EVIDENCE

    if analysis.relation == "mention":
        return PARTIALLY_SUPPORTED if MENTIONS_ONLY in labels else INSUFFICIENT_EVIDENCE

    if analysis.relation == "comparison":
        if SUPPORTS in labels:
            return SUPPORTED
        if not unassociated:
            return PARTIALLY_SUPPORTED     # all present, but across items
        if alternatives:
            return ENTITY_MISMATCH
        return PARTIALLY_SUPPORTED if (supported or partial) else INSUFFICIENT_EVIDENCE

    # property_of
    if not unassociated and not partial:
        return SUPPORTED
    if supported or partial:
        return PARTIALLY_SUPPORTED
    if alternatives:
        return ENTITY_MISMATCH
    return INSUFFICIENT_EVIDENCE


def _fmt(names: Iterable[str]) -> str:
    names = list(names)
    return ", ".join(names) if names else "none"


def _explain(v: ValidationResult, analysis: QueryAnalysis) -> str:
    if analysis.information_need == "hyperparameters":
        prop = "hyperparameters"
    else:
        prop = " ".join(analysis.property_terms) or "the requested information"
    scope =("the indexed content of " + _fmt(v.documents_checked)) \
        if v.scope == "document" else "the retrieved evidence"
    parts: List[str] = []

    if not v.requested_entities:
        if v.status == SUPPORTED:
            parts.append(f"Retrieved evidence contains all key query terms ({prop}) "
                         f"in a single passage/caption/row: {_fmt(v.supporting_evidence_ids)}. "
                         "No specific method was requested.")
        elif v.status == PARTIALLY_SUPPORTED:
            parts.append(f"Retrieved evidence is topically related to '{prop}' but no single "
                         "passage contains all key terms.")
        else:
            parts.append(f"No retrieved evidence addresses '{prop}'.")
        return " ".join(parts)

    req = _fmt(v.requested_entities)
    if v.supported_entities:
        strong = v.supporting_evidence_ids or v.partial_evidence_ids
        parts.append(f"Evidence explicitly associates {prop} with {_fmt(v.supported_entities)} "
                     f"({_fmt(strong)}).")
        if v.supporting_evidence_ids and v.partial_evidence_ids:
            parts.append(f"Partially supporting (caption-only or incomplete) items: "
                         f"{_fmt(v.partial_evidence_ids)}.")
    if v.partially_supported_entities:
        parts.append(f"For {_fmt(v.partially_supported_entities)} the association rests on a "
                     "caption that the evidence body does not confirm.")
    if v.mentioned_only_entities:
        parts.append(f"{_fmt(v.mentioned_only_entities)} is mentioned "
                     f"({_fmt(v.mention_evidence_ids) if v.mention_evidence_ids else 'in retrieved evidence'}) "
                     f"but no passage ties it to {prop}.")
    if v.missing_entities:
        parts.append(f"{_fmt(v.missing_entities)} is not found in {scope}.")
    if v.mismatched_entities and (v.mentioned_only_entities or v.missing_entities):
        alts = "; ".join(f"{k} ({', '.join(u)})" for k, u in v.mismatched_entities.items())
        parts.append(f"The {prop} evidence that was found concerns other methods: {alts}. "
                     f"Those values must not be attributed to "
                     f"{_fmt(v.mentioned_only_entities + v.missing_entities)}.")
    if v.relation == "comparison":
        if v.status == SUPPORTED:
            parts.append(f"A direct comparison of {req} is established.")
        else:
            parts.append(f"The requested comparison ({' vs '.join(v.requested_entities)}) is "
                         f"NOT established by {scope}.")
            if v.established_comparisons:
                comps = "; ".join(" vs ".join(c) for c in v.established_comparisons)
                parts.append(f"The evidence instead compares: {comps}.")
    for n in v.related_entity_notes:
        parts.append(n)
    if v.status == NOT_FOUND_IN_DOCUMENT:
        parts.append("A document-level scan of all indexed evidence did not find the "
                     "requested association (absence holds for the parsed/indexed text; "
                     "parser extraction errors cannot be ruled out).")
    if not parts:
        parts.append(f"The retrieved evidence does not establish {prop} for {req}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Document-level check
# ---------------------------------------------------------------------------

def document_findings(
    analysis: QueryAnalysis,
    evidence_by_doc: Dict[str, List[Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Scan ALL indexed evidence of each document.  For every requested method:
      'associated'     + uids — the property is tied to it somewhere
      'mentioned_only' + uids — named, never tied to the property
      'not_found'             — never named in the indexed content
    Reference-list items are ignored.
    """
    findings: Dict[str, Dict[str, Any]] = {}
    for doc_id, items in evidence_by_doc.items():
        per = [assess_evidence(ev, analysis) for ev in items]
        doc: Dict[str, Any] = {}
        for r in analysis.entities:
            assoc = [a.uid for a in per if a.label != REFERENCE_LIST
                     and r in a.requested_associated and not a.caption_only]
            weak = [a.uid for a in per if a.label != REFERENCE_LIST
                    and r in a.requested_associated and a.caption_only]
            ment = [a.uid for a in per if a.label != REFERENCE_LIST and r in a.requested_mentioned]
            if assoc:
                doc[r] = {"status": "associated", "evidence_ids": assoc}
            elif weak:
                doc[r] = {"status": "caption_only", "evidence_ids": weak}
            elif ment:
                doc[r] = {"status": "mentioned_only", "evidence_ids": ment}
            else:
                doc[r] = {"status": "not_found", "evidence_ids": []}
        doc["_items_scanned"] = len(items)
        findings[doc_id] = doc
    return findings


def apply_document_check(
    result: ValidationResult,
    analysis: QueryAnalysis,
    findings: Dict[str, Dict[str, Any]],
) -> ValidationResult:
    """
    Fold document-level findings into a validation result.

    Upgrades the status to NOT_FOUND_IN_DOCUMENT only when the scan of the
    checked document(s) establishes that a required requested method is
    never associated with the property.  Other fields (entity_mismatch,
    mismatched_entities …) are kept so the mismatch remains visible.
    """
    result.scope = "document"
    result.documents_checked = sorted(findings)
    result.document_findings = findings
    if not analysis.entities or result.status == SUPPORTED:
        result.explanation = _explain(result, analysis)
        return result

    def associated_anywhere(r: str) -> bool:
        return any(d.get(r, {}).get("status") in ("associated", "caption_only")
                   for d in findings.values())

    unestablished = [r for r in analysis.entities
                     if r not in result.supported_entities
                     and r not in result.partially_supported_entities]
    absent = [r for r in unestablished if not associated_anywhere(r)]

    if analysis.relation == "comparison":
        if absent:
            result.status = NOT_FOUND_IN_DOCUMENT
    elif analysis.relation == "property_of":
        if absent and len(absent) == len(analysis.entities):
            result.status = NOT_FOUND_IN_DOCUMENT
    elif analysis.relation == "mention":
        if all(d.get(r, {}).get("status") == "not_found"
               for r in analysis.entities for d in findings.values()):
            result.status = NOT_FOUND_IN_DOCUMENT

    result.caveats = [c for c in result.caveats if "document-level check" not in c]
    result.explanation = _explain(result, analysis)
    return result
