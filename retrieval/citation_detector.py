import re
from typing import Dict, List


FIGURE_PATTERN = re.compile(
    r"\b(?:Figure|Fig\.?)\s*\.?\s*(\d+)",
    re.IGNORECASE,
)

TABLE_PATTERN = re.compile(
    r"\bTable\s*\.?\s*(\d+)",
    re.IGNORECASE,
)

EQUATION_PATTERN = re.compile(
    r"(?:Equation|Eq\.?)\s*\.?\s*\(?(\d+)\)?",
    re.IGNORECASE,
)


def detect_citations(text: str) -> Dict[str, List[str]]:

    figures = FIGURE_PATTERN.findall(text)

    tables = TABLE_PATTERN.findall(text)

    equations = EQUATION_PATTERN.findall(text)

    return {
        "figures": sorted(set(f"Figure {number}" for number in figures)),
        "tables": sorted(set(f"Table {number}" for number in tables)),
        "equations": sorted(set(f"Equation {number}" for number in equations)),
    }


def detect_citations_in_results(results: List[Dict]) -> List[Dict]:

    for result in results:

        citations = detect_citations(result["text"])

        result["citations"] = citations

    return results
