"""
retrieval/ranking.py

Entity-aware re-ranking applied AFTER the established relevance pipeline
(dense + BM25 -> RRF -> optional CrossEncoder).

Why rank-based
--------------
RRF scores (~0.01-0.03) and CrossEncoder logits (unbounded, e.g. -11..+9)
live on incomparable scales and neither is a probability.  We therefore
never add raw scores.  The relevance pipeline contributes only its ORDER,
converted to a scale-free base score:

    base = 1 - (rank - 1) / (N - 1)          in [0, 1], rank 1 = best

Evidence signals then adjust this base in the same units ("fraction of the
candidate list"), and a small number of hard TIERS encode the guarantees:

    tier 0  evidence that explicitly associates a requested method with the
            requested property (SUPPORTS / PARTIAL)  — method-specific
            questions only
    tier 1  everything else
    tier 2  bibliography / reference-list items (unless the query asks
            about references)

Final order: (tier ASC, base + Σ adjustments DESC, base rank ASC).

Consequences
------------
- A hyperparameter table for the WRONG algorithm can never outrank evidence
  that ties the property to the requested algorithm (tier 0 > tier 1).
- It is demoted, not hidden: it can still appear as related context.
- Queries without a recognised method name are ordered by relevance (+
  modality preference); no hard tiers except the reference-list tier.

The resulting ``final`` value is an ordinal ranking score.  It is NOT a
probability, NOT a percentage, and NOT answer confidence.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from entities import related
from evidence_validator import (
    MENTIONS_ONLY, OTHER_METHOD, PARTIAL, REFERENCE_LIST, SUPPORTS, UNRELATED,
    EvidenceAssessment,
)
from query_understanding import QueryAnalysis


TIER_METHOD_SUPPORT = 0
TIER_DEFAULT = 1
TIER_REFERENCE = 2

WEIGHTS: Dict[str, float] = {
    # method-specific queries.  The tiers already guarantee that wrong-method
    # evidence never outranks evidence for the requested method; these
    # weights only order tier 1:
    #   mentions requested method > property context > wrong-method property
    #   evidence > neither method nor property
    "full_support": 0.25,          # SUPPORTS above PARTIAL inside tier 0
    "mentions_requested": 0.15,    # names the method, no property link
    "other_method": -0.10,         # property tied to a different method
    "related_method": -0.05,       # ... a closely related family (e.g. DRL vs DQN)
    "no_signal": -0.20,            # neither the method nor the property
    # queries without a method name
    "topic_support": 0.10,         # all key terms in one passage/caption/row
    # any query
    "requested_modality": 0.20,    # 'show the figure ...' and item is a figure
}


@dataclass
class RankedCandidate:
    uid: str
    base_rank: int
    base_score: float
    tier: int
    final_score: float
    adjustments: Dict[str, float] = field(default_factory=dict)


def base_score(rank: int, n: int) -> float:
    """Rank (1-based) -> [0, 1].  A single candidate scores 1."""
    if n <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (n - 1)


def _tier(analysis: QueryAnalysis, a: EvidenceAssessment) -> int:
    if a.label == REFERENCE_LIST:
        return TIER_REFERENCE
    if analysis.entities and a.label in (SUPPORTS, PARTIAL):
        return TIER_METHOD_SUPPORT
    return TIER_DEFAULT


def _adjustments(
    analysis: QueryAnalysis,
    a: EvidenceAssessment,
    ev_type: str,
    weights: Dict[str, float],
) -> Dict[str, float]:
    adj: Dict[str, float] = {}
    if analysis.entities:
        if a.label == SUPPORTS:
            adj["full_support"] = weights["full_support"]
        elif a.label == MENTIONS_ONLY:
            adj["mentions_requested"] = weights["mentions_requested"]
        elif a.label == OTHER_METHOD:
            all_related = all(
                any(related(r, o) for r in analysis.entities)
                for o in a.other_associated
            )
            key = "related_method" if all_related else "other_method"
            adj[key] = weights[key]
        elif a.label == UNRELATED:
            adj["no_signal"] = weights["no_signal"]
    elif a.label == SUPPORTS:
        adj["topic_support"] = weights["topic_support"]

    if analysis.requested_modalities and ev_type in analysis.requested_modalities:
        adj["requested_modality"] = weights["requested_modality"]
    return adj


def rank_candidates(
    ordered: Sequence[Tuple[str, str]],
    analysis: QueryAnalysis,
    assessments: Dict[str, EvidenceAssessment],
    weights: Optional[Dict[str, float]] = None,
) -> List[RankedCandidate]:
    """
    Parameters
    ----------
    ordered:
        (uid, evidence_type) pairs in relevance order (best first) as
        produced by RRF or the CrossEncoder.
    analysis:
        Query analysis.
    assessments:
        uid -> EvidenceAssessment for every uid in ``ordered``.

    Returns
    -------
    RankedCandidate list in final order.
    """
    w = dict(WEIGHTS)
    if weights:
        w.update(weights)
    n = len(ordered)
    out: List[RankedCandidate] = []
    for rank, (uid, ev_type) in enumerate(ordered, start=1):
        b = base_score(rank, n)
        a = assessments.get(uid)
        if a is None:
            out.append(RankedCandidate(uid, rank, b, TIER_DEFAULT, b))
            continue
        adj = _adjustments(analysis, a, ev_type, w)
        out.append(RankedCandidate(
            uid=uid,
            base_rank=rank,
            base_score=round(b, 6),
            tier=_tier(analysis, a),
            final_score=round(b + sum(adj.values()), 6),
            adjustments=adj,
        ))
    out.sort(key=lambda c: (c.tier, -c.final_score, c.base_rank))
    return out
