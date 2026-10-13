"""
retrieval/indexer.py

Indexes all evidence produced by the parser into:
  1. ChromaDB (dense vector store, for semantic retrieval)
  2. BM25Index (lexical index, for keyword retrieval)

Usage
-----
    python retrieval/indexer.py                             # index parser/data/processed
    python retrieval/indexer.py --processed-dir data/processed

Key changes from the previous version
--------------------------------------
- Consumes parser JSON via normalizer.py rather than expecting 'text_blocks'
- Indexes tables and figures (not just text)
- Uses Evidence / to_chroma_metadata() for ChromaDB metadata
- Builds and persists the BM25 index alongside ChromaDB
- Collection upgraded to 'academic_documents_v3' (new schema is incompatible)
- v4: vector-store / BM25 keys are doc-qualified uids ('<doc_id>::<id>').
  Parser table/figure ids ('table_p8_1') repeat across papers; in v3 later
  papers silently overwrote earlier ones (148 items -> 137 stored).
  v4 also stores derived metadata (entities, evidence_role, table headers…).
  The v3 collection and data/bm25_index.json are left untouched.
"""

import argparse
import json
import os
from typing import List

import chromadb

from bm25_index import BM25Index
from normalizer import normalize_document
from schema import Evidence


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "academic_documents_v4"
BM25_PATH = "data/bm25_index_v4.json"
CHUNK_SIZE = 450
OVERLAP = 100


# ---------------------------------------------------------------------------
# DocumentIndexer
# ---------------------------------------------------------------------------

class DocumentIndexer:
    """
    Indexes parsed academic documents into ChromaDB and the BM25 index.

    Each call to index_document:
      1. Loads the *_parsed.json
      2. Normalises to Evidence via normalizer.normalize_document()
      3. Embeds evidence content with sentence-transformers
      4. Upserts to ChromaDB with metadata
      5. Upserts to BM25Index and saves
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        persist_directory: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        bm25_path: str = BM25_PATH,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = OVERLAP,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap = overlap

        from sentence_transformers import SentenceTransformer

        print(f"[Indexer] Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Multimodal academic document retrieval"},
        )

        self.bm25 = BM25Index(index_path=bm25_path)
        self._bm25_path = bm25_path

    @classmethod
    def from_components(
        cls,
        embedding_model,
        collection,
        bm25: BM25Index,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = OVERLAP,
    ) -> "DocumentIndexer":
        """Build an indexer from existing components (no model download)."""
        obj = object.__new__(cls)
        obj.persist_directory = None
        obj.collection_name = getattr(collection, "name", "")
        obj.chunk_size = chunk_size
        obj.overlap = overlap
        obj.embedding_model = embedding_model
        obj.client = None
        obj.collection = collection
        obj.bm25 = bm25
        obj._bm25_path = bm25.index_path
        return obj

    def reset(self) -> None:
        """Delete and recreate the ChromaDB collection and clear the BM25 index."""
        print(f"[Indexer] Resetting collection '{self.collection_name}'…")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Multimodal academic document retrieval"},
        )
        # Overwrite BM25 with an empty index
        import json as _json
        import os as _os
        if _os.path.exists(self._bm25_path):
            _os.remove(self._bm25_path)
        from bm25_index import BM25Index as _BM25
        self.bm25 = _BM25(index_path=self._bm25_path)
        print("[Indexer] Reset complete.")

    # -------------------------------------------------------------------
    # Core indexing
    # -------------------------------------------------------------------

    def load_parsed_document(self, json_path: str) -> dict:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Parsed document not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def index_document(self, json_path: str) -> None:
        """
        Index one parsed document JSON file into ChromaDB and BM25.
        """
        doc = self.load_parsed_document(json_path)
        doc.setdefault("doc_id", os.path.basename(json_path))
        self.index_parsed(doc)

    def index_parsed(self, doc: dict, show_progress: bool = True) -> int:
        """
        Index one already-loaded parsed document.  Returns the item count.

        Replace semantics: any evidence previously indexed for this doc_id is
        removed first, so re-chunking never leaves stale chunks behind.
        """
        doc_id = doc.get("doc_id", "unknown")
        print(f"\n[Indexer] Indexing: {doc_id}")

        evidence_list: List[Evidence] = normalize_document(
            doc,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        if not evidence_list:
            print(f"[Indexer] No evidence produced for {doc_id}. Skipping.")
            return 0

        # Guard: duplicate uids inside one document would overwrite silently.
        uids = [ev.uid for ev in evidence_list]
        if len(uids) != len(set(uids)):
            dups = sorted({u for u in uids if uids.count(u) > 1})
            raise ValueError(f"Duplicate evidence uids in {doc_id}: {dups}")

        # Remove this document's previous entries (replace semantics).
        try:
            self.collection.delete(where={"doc_id": {"$eq": doc_id}})
        except Exception as exc:  # empty collection on some Chroma versions
            print(f"[Indexer] (no previous entries removed for {doc_id}: {exc})")
        self.bm25.remove_document(doc_id)

        # --- ChromaDB ---
        contents = [ev.content for ev in evidence_list]
        metadatas = [ev.to_chroma_metadata() for ev in evidence_list]

        print(f"[Indexer] Generating embeddings for {len(evidence_list)} items…")
        embeddings = self.embedding_model.encode(
            contents, show_progress_bar=show_progress, normalize_embeddings=True
        )

        self.collection.upsert(
            ids=uids,
            documents=contents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        print(f"[Indexer] ChromaDB upserted {len(evidence_list)} items.")

        # --- BM25 ---
        self.bm25.add_evidence(evidence_list)
        self.bm25.save()

        print(f"[Indexer] [OK] {doc_id} indexed successfully.")
        return len(evidence_list)

    def index_directory(
        self,
        processed_directory: str = "data/processed",
    ) -> None:
        """
        Walk processed_directory and index every *_parsed.json found.
        """
        if not os.path.exists(processed_directory):
            raise FileNotFoundError(
                f"Processed directory not found: {processed_directory}"
            )

        json_files = []
        for root, _, files in os.walk(processed_directory):
            for fname in files:
                if fname.endswith("_parsed.json"):
                    json_files.append(os.path.join(root, fname))

        print(f"[Indexer] Found {len(json_files)} parsed document(s).")

        for json_path in json_files:
            try:
                self.index_document(json_path)
            except Exception as exc:
                print(f"[Indexer] [WARN] Failed to index {json_path}: {exc}")

        print("\n[Indexer] Indexing complete.")
        self._print_stats()

    # -------------------------------------------------------------------
    # Inspection
    # -------------------------------------------------------------------

    def get_collection_stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "document_count": self.collection.count(),
            "bm25_entries": len(self.bm25),
            "bm25_by_type": self.bm25.stats().get("by_type", {}),
        }

    def _print_stats(self) -> None:
        stats = self.get_collection_stats()
        print("\n[Indexer] Vector store statistics:")
        print(f"  Collection : {stats['collection']}")
        print(f"  Total items: {stats['document_count']}")
        print(f"  BM25 total : {stats['bm25_entries']}")
        print(f"  BM25 types : {stats['bm25_by_type']}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index parsed documents.")
    parser.add_argument(
        "--processed-dir",
        default="parser/data/processed",
        help="Root directory containing *_parsed.json files.",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_DIR,
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--bm25-path",
        default=BM25_PATH,
        help="Path for the BM25 index JSON file.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the ChromaDB collection (and BM25 file) "
             "before indexing. Only the configured collection is touched.",
    )
    args = parser.parse_args()

    indexer = DocumentIndexer(
        persist_directory=args.chroma_dir,
        bm25_path=args.bm25_path,
    )
    if args.reset:
        indexer.reset()
    indexer.index_directory(args.processed_dir)