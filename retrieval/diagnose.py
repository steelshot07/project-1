"""
retrieval/diagnose.py

Diagnostic CLI for evaluating retrieval quality before connecting an LLM.

Prints a human-readable breakdown of each result:
  - Evidence ID, type, score
  - Page, document, section
  - Caption (for tables/figures)
  - Artifact path (for visual evidence)
  - Content snippet
  - Evidence assessment (method-aware validation label)
and the query analysis and aggregate validation status.

Usage
-----
    # All documents
    python retrieval/diagnose.py "Which method achieved the lowest travel time?"

    # Restrict to one document
    python retrieval/diagnose.py "freezing interval" --doc-id 2012.12104v1

    # Tables only
    python retrieval/diagnose.py "mean travel times" --types table

    # Skip reranker for speed
    python retrieval/diagnose.py "architecture of the network" --no-rerank

    # Output as JSON (for piping into other tools)
    python retrieval/diagnose.py "ramp metering" --json

    # Scan the whole document before reporting a method as absent
    python retrieval/diagnose.py "hyperparameters of DQN" --doc-id 2012.13026v1 --document-check

    # Ablation: previous ordering (no entity-aware re-ranking)
    python retrieval/diagnose.py "hyperparameters of DQN" --no-entity-aware

    # Legacy v3 index (pre-uid keys) for before/after comparisons
    python retrieval/diagnose.py "hyperparameters of DQN" --legacy-index --no-entity-aware

    # Per-stage ranking details
    python retrieval/diagnose.py "hyperparameters of DQN" -v
"""

import argparse
import json
import sys
import os

# Allow running from the project root: python retrieval/diagnose.py
sys.path.insert(0, os.path.dirname(__file__))

from retriever import BM25_PATH, CHROMA_DIR, COLLECTION_NAME, AcademicRetriever
from schema import RetrievalResponse

LEGACY_COLLECTION = "academic_documents_v3"
LEGACY_BM25_PATH = "data/bm25_index.json"

BAR = "=" * 72
SEP = "-" * 72


def print_response(response: RetrievalResponse, verbose: bool = False) -> None:
    print(f"\n{BAR}")
    print(f"  Query: {response.query}")
    print(f"{BAR}")

    qa = response.query_analysis
    if qa is not None:
        print(f"  Need      : {qa.information_need}   Relation: {qa.relation}")
        print(f"  Entities  : {qa.entities or '-'}   "
              f"Property terms: {qa.property_terms or '-'}")
        if qa.requested_modalities:
            print(f"  Modalities: {qa.requested_modalities}")

    if not response.results:
        print("  No results found.")
        return

    print(f"\n  Top {len(response.results)} result(s):\n")

    for i, r in enumerate(response.results, 1):
        print(f"  {i}.")
        print(f"     ID        : {r.id}")
        print(f"     Type      : {r.type}")
        print(f"     Score     : {r.score:.4f}  (ranking score; not a probability)")
        print(f"     Page      : {r.page}")
        print(f"     Document  : {r.doc_id}")
        if r.section:
            print(f"     Section   : {r.section[:70]}")
        if r.caption:
            print(f"     Caption   : {r.caption[:80]}")
        if r.artifact_path:
            print(f"     Artifact  : {r.artifact_path}")
        content_preview = r.content.replace("\n", " ")[:160]
        print(f"     Content   : {content_preview}…")
        if r.context_chunks:
            print(f"     Context   : {r.context_chunks[0][:80]}…")
        if r.assessment:
            a = r.assessment
            line = f"     Evidence  : {a['label']}"
            if a["requested_associated"]:
                line += f"  requested->{list(a['requested_associated'])}"
            if a["other_associated"]:
                line += f"  other->{list(a['other_associated'])}"
            print(line)
        d = r.score_details or {}
        if d.get("added_by"):
            print(f"     Added by  : {d['added_by']}")
        elif verbose and d:
            print(f"     Ranking   : tier={d.get('tier')} base_rank={d.get('base_rank')} "
                  f"({d.get('base_order')}) adj={d.get('adjustments')} "
                  f"dense={d.get('dense_rank')} bm25={d.get('bm25_rank')} "
                  f"focused={d.get('focused_bm25_rank')} rrf={d.get('rrf_score')} "
                  f"ce={d.get('rerank_score')}")
        print()

    v = response.validation
    if v is not None:
        print(SEP)
        print(f"  Validation status : {v.status}   (scope: {v.scope})")
        print(f"  Entity mismatch   : {v.entity_mismatch}")
        if v.requested_entities:
            print(f"  Requested         : {v.requested_entities}")
            print(f"  Supported         : {v.supported_entities or '-'}")
            if v.partially_supported_entities:
                print(f"  Caption-only      : {v.partially_supported_entities}")
            print(f"  Mentioned only    : {v.mentioned_only_entities or '-'}")
            print(f"  Not found         : {v.missing_entities or '-'}")
            if v.mismatched_entities:
                print(f"  Other methods     : {list(v.mismatched_entities)}")
            if v.established_comparisons:
                comps = [" vs ".join(c) for c in v.established_comparisons]
                print(f"  Comparisons found : {comps}")
        print(f"  Explanation       : {v.explanation}")
        for c in v.caveats:
            print(f"  Caveat            : {c}")

    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose retrieval quality for a query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query string. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Restrict retrieval to a single document ID.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="Filter by evidence type(s): text, table, figure.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate pool size for dense+BM25 retrieval (default: 20).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip CrossEncoder reranking (faster but lower quality).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--diversify",
        action="store_true",
        help="Apply modality-aware diversification.",
    )
    parser.add_argument(
        "--no-entity-aware",
        action="store_true",
        help="Ablation: disable entity-aware re-ranking (previous ordering).",
    )
    parser.add_argument(
        "--document-check",
        action="store_true",
        help="Scan all indexed evidence of the document(s) before reporting absence.",
    )
    parser.add_argument(
        "--legacy-index",
        action="store_true",
        help="Use the pre-uid v3 collection and BM25 file (before/after runs).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-stage ranking details.",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_DIR,
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"ChromaDB collection (default: {COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--bm25-path",
        default=None,
        help=f"BM25 index file (default: {BM25_PATH}).",
    )
    args = parser.parse_args()

    collection = args.collection or (
        LEGACY_COLLECTION if args.legacy_index else COLLECTION_NAME
    )
    bm25_path = args.bm25_path or (
        LEGACY_BM25_PATH if args.legacy_index else BM25_PATH
    )

    try:
        retriever = AcademicRetriever(
            persist_directory=args.chroma_dir,
            collection_name=collection,
            bm25_path=bm25_path,
            reranker_model=(
                None if args.no_rerank
                else "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
        )
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("Run 'python retrieval/indexer.py --reset' first to build the index.")
        sys.exit(1)

    def run_query(q: str) -> None:
        response = retriever.search(
            query=q,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            doc_id=args.doc_id,
            types=args.types,
            use_reranker=not args.no_rerank,
            diversify=args.diversify,
            entity_aware=not args.no_entity_aware,
            document_check=args.document_check,
        )
        if args.as_json:
            print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
        else:
            print_response(response, verbose=args.verbose)

    if args.query:
        run_query(args.query)
    else:
        # Interactive mode
        print("\nHybrid Retrieval Diagnostics (type 'exit' to quit)\n")
        while True:
            try:
                q = input("Query: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if q.lower() in ("exit", "quit", "q"):
                break
            if not q:
                continue
            run_query(q)


if __name__ == "__main__":
    main()
