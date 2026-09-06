import os
import sys

# Compute project root so we can reliably find the data and parser folders
# no matter where this script is executed from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Ensure we can import from retrieval module
sys.path.insert(0, SCRIPT_DIR)

from indexer import DocumentIndexer
from retriever import AcademicRetriever

def main():
    print("==================================================")
    print("  1. INDEXING THE DOCUMENTS")
    print("==================================================")
    
    # Absolute paths pointing to the project root
    chroma_dir = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    bm25_path = os.path.join(PROJECT_ROOT, "data", "bm25_index.json")
    
    indexer = DocumentIndexer(
        persist_directory=chroma_dir,
        bm25_path=bm25_path
    )
    
    processed_dir = os.path.join(PROJECT_ROOT, "parser", "data", "processed")
    if not os.path.exists(processed_dir):
        print(f"Error: Could not find '{processed_dir}'.")
        print("Please make sure you have run the parser first to generate the parsed JSONs.")
        sys.exit(1)
        
    print("Resetting old index (if any) and re-indexing all parsed documents...")
    indexer.reset()
    indexer.index_directory(processed_dir)
    print("\nIndexing complete!\n")

    print("==================================================")
    print("  2. TESTING THE RETRIEVAL PIPELINE")
    print("==================================================")
    
    print("Loading the retrieval models (this might take a few seconds)...")
    retriever = AcademicRetriever(
        persist_directory=chroma_dir,
        bm25_path=bm25_path
    )
    
    test_queries = [
        "What hyperparameters were used for the DQN?",
        "Compare travel times between PI-ALINEA and DQN"
    ]
    
    for query in test_queries:
        print(f"\n[QUERY]: {query}")
        print("-" * 50)
        
        # We enable diversification so text doesn't drown out tables and figures
        response = retriever.search(query=query, top_k=3, diversify=True)
        
        for i, result in enumerate(response.results, 1):
            print(f"#{i} [Score: {result.score:.4f}] - Type: {result.type.upper()}")
            print(f"   Doc: {result.doc_id} (Page: {result.page})")
            if result.artifact_path:
                print(f"   Image/Text Path: {result.artifact_path}")
            
            # Print a short snippet of the content
            snippet = result.content[:150].replace('\n', ' ')
            print(f"   Snippet: {snippet}...\n")


if __name__ == "__main__":
    # Force UTF-8 encoding for Windows terminals so arrows/emoji don't crash
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        
    main()
