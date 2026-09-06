"""
retrieval/smoke_test.py

Quick end-to-end smoke test that runs after indexing.
Not part of the unit test suite — this requires a live ChromaDB + BM25 index.

Usage:
    python retrieval/smoke_test.py
    python retrieval/smoke_test.py --verbose
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))

from retriever import AcademicRetriever

QUERIES = [
    {
        "query": "What hyperparameters were used in the deep Q-learning agent?",
        "expect_types": ["table"],      # table must appear
        "expect_keyword": "buffer",
        "label": "Table retrieval (hyperparameters)",
    },
    {
        "query": "mean travel time comparison between DRL and PI-ALINEA",
        "expect_types": ["table", "figure"],  # either visual modality is valid
        "expect_keyword": "travel",
        "label": "Visual evidence retrieval (results comparison)",
    },
    {
        "query": "architecture diagram of the ramp metering control scheme",
        "expect_types": ["figure"],     # figure must appear
        "expect_keyword": "Figure",
        "label": "Figure retrieval (architecture diagram)",
    },
    {
        "query": "experience replay buffer training procedure",
        "expect_types": ["text"],       # text passage must appear
        "expect_keyword": "replay",
        "label": "Text retrieval (training procedure)",
    },
]

def run_smoke_test(verbose: bool = False) -> int:
    """Returns number of failures."""
    print("\n[Smoke Test] Loading retriever...")
    retriever = AcademicRetriever()
    print(f"[Smoke Test] ChromaDB items: {retriever.collection.count()}")
    print(f"[Smoke Test] BM25 items:     {len(retriever.bm25)}\n")

    failures = 0

    for case in QUERIES:
        response = retriever.search(
            query=case["query"],
            top_k=5,
            use_reranker=False,   # fast smoke test
            expand_evidence=False,
            diversify=True,
        )

        if not response.results:
            print(f"  [FAIL] {case['label']}: no results returned")
            failures += 1
            continue

        top = response.results[0]

        # Check provenance fields on top result
        provenance_ok = (
            top.doc_id is not None and top.doc_id != ""
            and top.page is not None
            and top.type in ("text", "table", "figure")
        )

        # Check if any expected type appears in top-5
        type_present = any(
            r.type in case["expect_types"]
            for r in response.results
        )

        # Check keyword in any result content
        keyword_present = any(
            case["expect_keyword"].lower() in r.content.lower()
            for r in response.results
        )

        # Check artifact path is present for visual evidence
        visual_results = [r for r in response.results if r.type in ("table", "figure")]
        artifact_ok = all(r.artifact_path is not None for r in visual_results)

        ok = provenance_ok and type_present and keyword_present and artifact_ok

        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status} {case['label']}")

        if verbose or not ok:
            print(f"         Top result: {top.id}  type={top.type}  page={top.page}  doc={top.doc_id}")
            print(f"         Provenance OK: {provenance_ok}")
            print(f"         Type {case['expect_types']} in top-5: {type_present}")
            print(f"         Keyword '{case['expect_keyword']}' found: {keyword_present}")
            print(f"         Artifact paths OK: {artifact_ok}")
            if verbose:
                for r in response.results:
                    print(f"           {r.type:6}  {r.id:30}  score={r.score:.4f}  artifact={r.artifact_path}")
            print()

        if not ok:
            failures += 1

    print(f"\n[Smoke Test] Results: {len(QUERIES) - failures}/{len(QUERIES)} passed")
    return failures


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    failures = run_smoke_test(verbose=verbose)
    sys.exit(0 if failures == 0 else 1)
