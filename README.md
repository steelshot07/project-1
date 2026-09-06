# Project 1: Academic Document Parser and Retriever

This project is an end-to-end pipeline that takes academic PDFs, deeply understands their visual and textual layout (using vision models and PyMuPDF), and indexes them into a hybrid vector + lexical search engine.

The system is composed of two main stages:
1. **The Parser**: Extracts text chunks, detects tables and figures with bounding boxes, groups them by section and page, and extracts crops/texts into a normalized JSON format.
2. **The Retrieval Pipeline**: Ingests the parsed output into a dual ChromaDB (Dense) and BM25 (Lexical) index, and supports semantic querying with Reciprocal Rank Fusion and CrossEncoder re-ranking. 

## Requirements

You must have the dependencies installed. It is highly recommended to use a virtual environment (`.venv`).

```powershell
pip install -r requirements.txt
```

> [!WARNING]
> HuggingFace may display a warning about unauthenticated requests to the HF Hub (which limits download speeds). You can safely ignore this, or set the `HF_TOKEN` environment variable to silence it.

---

## 1. Running the Parser

The parser takes raw academic PDFs and extracts their text, figures, and tables. 

Place your raw PDFs in `parser/data/raw_pdfs`. Then run:

```powershell
python parser/main.py
```

**What it does:**
- Analyzes each PDF page using `PyMuPDF`.
- Uses a vision model to detect tables and figures.
- Extracts visual crops (saved in `parser/data/processed/<doc_id>/figures`).
- Extracts table contents into text (saved in `parser/data/processed/<doc_id>/tables`).
- Outputs a normalized `_parsed.json` file for each document containing bounding boxes, captions, and text chunks.

---

## 2. Running the Retrieval Pipeline

Once the documents have been parsed, you can index them and query them.

### Automated End-to-End Test

To quickly index all parsed documents and run a few test queries, run the end-to-end script from the root of the project:

```powershell
python retrieval/run_retrieval.py
```

This will automatically:
1. Load your `parser/data/processed` JSONs.
2. Build/Reset the `ChromaDB` (Vector) and `rank-bm25` (Lexical) indexes in the `data/` folder.
3. Load the retrieval and reranker models.
4. Run sample queries showing that text, tables, and figures can all be retrieved based on semantic similarity.

### Interactive Query Diagnostics

If you want to test your own queries against the papers and see exactly what evidence is retrieved, use the interactive `diagnose.py` tool:

```powershell
python retrieval/diagnose.py
```
*(Tip: In Windows, if you experience Unicode errors with emojis/arrows, run it with `$env:PYTHONUTF8=1; python retrieval/diagnose.py`)*

### Finding Similar Papers (Corpus Similarity)

The project includes an architecturally separate module for computing document-level similarity (useful for finding related papers or clustering). 

```powershell
# Find papers similar to '2012.12104v1'
python retrieval/corpus_similarity.py --source 2012.12104v1 --top-k 3

# Compare two specific papers
python retrieval/corpus_similarity.py --source 2012.12104v1 --compare 2012.13026v1
```

---

## 3. Running the Unit Tests

The retrieval module is fully covered by 65 deterministic unit tests spanning schema serialization, BM25 indexing, hybrid fusion math, and provenance preservation.

```powershell
python -m pytest retrieval/tests/test_retrieval.py -v
```
