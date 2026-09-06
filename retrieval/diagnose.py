"""
retrieval/diagnose.py

Diagnostic CLI for evaluating retrieval quality before connecting an LLM.

Prints a human-readable breakdown of each result:
  - Evidence ID, type, score
  - Page, document, section
  - Caption (for tables/figures)
  - Artifact path (for visual evidence)
  - Content snippet

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
"""

import argparse
import json
import sys
import os

# Allow running from the project root: python retrieval/diagnose.py
sys.path.insert(0, os.path.dirname(__file__))

from retriever import AcademicRetriever
from schema import RetrievalResponse

CHROMA_DIR = "data/chroma_db"
BM25_PATH  = "data/bm25_index.json"

BAR = "=" * 72
SEP = "-" * 72


def print_response(response: RetrievalResponse) -> None:
    print(f"\n{BAR}")
    print(f"  Query: {response.query}")
    print(f"{BAR}")

    if not response.results:
        print("  No results found.")
        return

    print(f"\n  Top {len(response.results)} result(s):\n")

    for i, r in enumerate(response.results, 1):
        print(f"  {i}.")
        print(f"     ID        : {r.id}")
        print(f"     Type      : {r.type}")
        print(f"     Score     : {r.score:.4f}")
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
        print()

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
        "--chroma-dir",
        default=CHROMA_DIR,
    )
    parser.add_argument(
        "--bm25-path",
        default=BM25_PATH,
    )
    args = parser.parse_args()

    try:
        retriever = AcademicRetriever(
            persist_directory=args.chroma_dir,
            bm25_path=args.bm25_path,
            reranker_model=(
                None if args.no_rerank
                else "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
        )
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("Run 'python retrieval/indexer.py' first to build the index.")
        sys.exit(1)

    def run_query(q: str) -> None:
        response = retriever.search(
            query=q,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            doc_id=args.doc_id,
            types=args.types,
            use_reranker=not args.no_rerank,
        )
        if args.as_json:
            print(json.dumps(response.to_dict(), indent=2))
        else:
            print_response(response)

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
