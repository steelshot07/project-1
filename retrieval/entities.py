"""
retrieval/entities.py

Deterministic detection of technical method / algorithm names in text.

Shared by query understanding (query side), the normaliser (evidence
metadata) and the evidence validator (evidence side), so both sides of
the comparison use exactly the same rules.

Two detection mechanisms
------------------------
1. A SMALL, extensible registry of well-known method names
   (``METHOD_REGISTRY``).  Each entry lists only *explicit* surface forms
   of the same name: the acronym and its literal expansion
   (e.g. DQN <-> "deep Q-network").  No two different methods are ever
   declared synonyms.

2. A shape-based fallback (``_SHAPE_RE``) that recognises method-like
   tokens not present in the registry: ALL-CAPS acronyms (``FOO``),
   hyphenated acronyms (``PI-FOO``), acronyms with digits (``A3C``) and
   internal-capital names (``ResNet``).  Ordinary acronyms (venues,
   units, metrics, file formats ...) are excluded via ``NON_METHOD_ACRONYMS``.
   Shape-detected entities carry ``source="pattern"`` so callers can treat
   them with less confidence than registry entities.

Relatedness is NOT equivalence
------------------------------
``RELATED_METHODS`` records explicit, defensible "broader family"
relations (DQN is a DRL method; DQN is a deep Q-learning algorithm).
These relations are used ONLY to explain a mismatch to the downstream
generator ("the evidence concerns DRL, a broader category that includes
DQN").  They never make one method count as evidence for another.

To add a method, append one ``MethodEntry`` to ``METHOD_REGISTRY``.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodEntry:
    canonical: str
    """Display / canonical name, e.g. 'DQN'."""

    patterns: Tuple[str, ...]
    """Regex fragments for explicit surface forms of THIS name only."""

    kind: str = "method"
    """'method' (a specific algorithm) or 'category' (a family, e.g. DRL)."""


# Word boundary that also treats '-' as part of a name, so 'ALINEA' does
# not match inside 'PI-ALINEA' and 'Q-learning' does not match inside
# 'deep Q-learning' (handled separately via longest-match resolution).
_L = r"(?<![\w-])"
_R = r"(?![\w-])"

METHOD_REGISTRY: Tuple[MethodEntry, ...] = (
    # --- reinforcement learning ------------------------------------------
    MethodEntry("DQN", (r"DQNs?", r"(?i:deep[\s-]+q[\s-]*networks?)")),
    MethodEntry("Double DQN", (r"DDQN", r"(?i:double[\s-]+dqn)")),
    MethodEntry("Deep Q-learning", (r"(?i:deep[\s-]+q[\s-]*learning)",)),
    MethodEntry("Q-learning", (r"(?i:q[\s-]*learning)",)),
    MethodEntry("SAC", (r"SAC", r"(?i:soft[\s-]+actor[\s-]*critic)")),
    MethodEntry("DDPG", (r"DDPG", r"(?i:deep[\s-]+deterministic[\s-]+policy[\s-]+gradients?)")),
    MethodEntry("TD3", (r"TD3",)),
    MethodEntry("PPO", (r"PPO", r"(?i:proximal[\s-]+policy[\s-]+optimi[sz]ation)")),
    MethodEntry("TRPO", (r"TRPO", r"(?i:trust[\s-]+region[\s-]+policy[\s-]+optimi[sz]ation)")),
    MethodEntry("A2C", (r"A2C",)),
    MethodEntry("A3C", (r"A3C",)),
    MethodEntry("Imitation learning", (r"(?i:imitation[\s-]+learning)",)),
    MethodEntry("DRL", (r"DRL", r"(?i:deep[\s-]+reinforcement[\s-]+learning)"), kind="category"),
    MethodEntry("RL", (r"RL", r"(?i:(?<!deep )reinforcement[\s-]+learning)"), kind="category"),
    # --- traffic control ----------------------------------------------------
    MethodEntry("PI-ALINEA", (r"PI[\s-]ALINEA",)),
    MethodEntry("ALINEA", (r"ALINEA",)),
    # --- common deep-learning architectures ----------------------------------
    MethodEntry("CNN", (r"CNNs?", r"(?i:convolutional[\s-]+neural[\s-]+networks?)")),
    MethodEntry("LSTM", (r"LSTMs?", r"(?i:long[\s-]+short[\s-]+term[\s-]+memory)")),
    MethodEntry("Transformer", (r"Transformers?",)),
    MethodEntry("BERT", (r"BERT",)),
)

# Explicit "A is a (kind of) B" relations, used for explanations only.
# Keys and values are canonical names.  Never used to substitute evidence.
RELATED_METHODS: Dict[str, Tuple[str, ...]] = {
    "DQN": ("Deep Q-learning", "Q-learning", "DRL", "RL"),
    "Double DQN": ("DQN", "Deep Q-learning", "DRL", "RL"),
    "Deep Q-learning": ("Q-learning", "DRL", "RL"),
    "Q-learning": ("RL",),
    "SAC": ("DRL", "RL"),
    "DDPG": ("DRL", "RL"),
    "TD3": ("DRL", "RL"),
    "PPO": ("DRL", "RL"),
    "TRPO": ("DRL", "RL"),
    "A2C": ("DRL", "RL"),
    "A3C": ("DRL", "RL"),
    "DRL": ("RL",),
    "PI-ALINEA": ("ALINEA",),
}

# Ordinary acronyms that are NOT method names (venues, units, metrics,
# formats, generic technology words, document furniture).
NON_METHOD_ACRONYMS: Set[str] = {
    "AI", "ML", "NLP", "CV", "PDF", "PNG", "URL", "API", "GPU", "CPU", "RAM",
    "USA", "UK", "EU", "IEEE", "ACM", "ICML", "ICLR", "NIPS", "NEURIPS",
    "CVPR", "ICCV", "ECCV", "AAAI", "IJCAI", "KDD", "ACL", "EMNLP", "TR",
    "MW", "MVAR", "KW", "KM", "KMH", "HZ", "MS", "SEC", "PU", "DB",
    "RMSE", "MSE", "MAE", "MAPE", "AUC", "ROC", "FAR", "FRR", "EER", "TPR",
    "FPR", "IOU", "BLEU", "ROUGE", "SD", "STD", "CI",
    "FIG", "TABLE", "EQ", "SEC", "APPENDIX", "ID", "IDS", "OK", "NO", "VS",
    "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
    "PHD", "MSC", "BSC", "DOI", "ISBN", "ARXIV", "PP", "VOL",
}

# Method-shaped tokens: FOO, FOO-BAR, A3C, ResNet, ResNet-50.
_SHAPE_RE = re.compile(
    r"(?<![\w-])("
    r"[A-Z][A-Z0-9]*[A-Z][A-Z0-9]*(?:-[A-Z0-9][A-Za-z0-9]*)*"  # ALLCAPS (2+ caps), opt -suffix
    r"|[A-Z]+-[A-Z][A-Z0-9]+"                                    # PI-ALINEA style
    r"|[A-Z][a-z]+[A-Z][A-Za-z0-9]*(?:-\d+)?"                    # ResNet, ResNet-50
    r")(?![\w-])"
)


# ---------------------------------------------------------------------------
# Compiled registry
# ---------------------------------------------------------------------------

def _compile(entry: MethodEntry) -> re.Pattern:
    return re.compile(_L + "(?:" + "|".join(entry.patterns) + ")" + _R)


_COMPILED: List[Tuple[MethodEntry, re.Pattern]] = [
    (e, _compile(e)) for e in METHOD_REGISTRY
]
_BY_CANONICAL: Dict[str, MethodEntry] = {e.canonical: e for e in METHOD_REGISTRY}
_REGISTRY_TOKENS: Set[str] = {e.canonical.upper() for e in METHOD_REGISTRY}
_NON_METHOD_KEYS: Set[str] = {re.sub(r"[\s\-_]+", "", t).upper() for t in NON_METHOD_ACRONYMS}


@dataclass
class EntityMention:
    canonical: str
    surface: str
    start: int
    end: int
    source: str = "registry"      # 'registry' | 'pattern' | 'query'
    kind: str = "method"


def _canon_key(name: str) -> str:
    """Normalise a surface form for comparison (case / hyphen / space)."""
    return re.sub(r"[\s\-_]+", "", name).upper()


def lookup(name: str) -> Optional[MethodEntry]:
    """Return the registry entry whose surface forms match ``name`` exactly."""
    for entry, pat in _COMPILED:
        m = pat.fullmatch(name.strip())
        if m:
            return entry
    return None


def kind_of(canonical: str) -> str:
    entry = _BY_CANONICAL.get(canonical)
    return entry.kind if entry else "method"


def related(a: str, b: str) -> Optional[str]:
    """
    Describe an explicit relation between two canonical names, or None.

    Returns e.g. 'DRL is a broader category that includes DQN'.
    Relatedness is informational only — never equivalence.
    """
    if b in RELATED_METHODS.get(a, ()):
        return f"{b} is a broader category/family that includes {a}"
    if a in RELATED_METHODS.get(b, ()):
        return f"{b} is a specific instance within the broader {a} family"
    return None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def find_entities(
    text: str,
    extra_names: Iterable[str] = (),
    use_shape: bool = True,
) -> List[EntityMention]:
    """
    Detect method / algorithm names in ``text``.

    Parameters
    ----------
    text:
        Any text (query, caption, chunk, table row).
    extra_names:
        Additional canonical names to look for verbatim (e.g. unregistered
        names extracted from the query, such as 'FooNet').  Matched with
        case/hyphen-insensitive exact-token rules.
    use_shape:
        Also report method-shaped tokens not in the registry.

    Returns
    -------
    Non-overlapping mentions, longest match wins, in text order.
    """
    if not text:
        return []

    found: List[EntityMention] = []
    for entry, pat in _COMPILED:
        for m in pat.finditer(text):
            found.append(EntityMention(entry.canonical, m.group(0), m.start(),
                                       m.end(), "registry", entry.kind))

    for name in extra_names:
        if lookup(name):
            continue  # already covered by the registry
        parts = [re.escape(p) for p in re.split(r"[\s\-_]+", name) if p]
        if not parts:
            continue
        pat = re.compile(_L + r"[\s\-_]?".join(parts) + _R, re.IGNORECASE)
        for m in pat.finditer(text):
            found.append(EntityMention(name, m.group(0), m.start(), m.end(), "query"))

    if use_shape:
        for m in _SHAPE_RE.finditer(text):
            tok = m.group(1)
            if _canon_key(tok) in _NON_METHOD_KEYS or tok.upper() in _REGISTRY_TOKENS:
                continue
            found.append(EntityMention(tok, tok, m.start(1), m.end(1), "pattern"))

    # Resolve overlaps: longest span first, then registry > query > pattern.
    priority = {"registry": 0, "query": 1, "pattern": 2}
    found.sort(key=lambda e: (-(e.end - e.start), priority[e.source], e.start))
    taken: List[Tuple[int, int]] = []
    result: List[EntityMention] = []
    for e in found:
        if any(e.start < t_end and t_start < e.end for t_start, t_end in taken):
            continue
        taken.append((e.start, e.end))
        result.append(e)
    result.sort(key=lambda e: e.start)
    return result


def entity_names(
    text: str,
    extra_names: Iterable[str] = (),
    use_shape: bool = True,
) -> List[str]:
    """Unique canonical names in text order."""
    seen: List[str] = []
    for e in find_entities(text, extra_names, use_shape):
        if e.canonical not in seen:
            seen.append(e.canonical)
    return seen


def same_entity(a: str, b: str) -> bool:
    """True when two names refer to the same method (registry-explicit only)."""
    if _canon_key(a) == _canon_key(b):
        return True
    ea, eb = lookup(a), lookup(b)
    return ea is not None and ea is eb
