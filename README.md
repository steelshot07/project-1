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

Useful flags:

```powershell
# Per-stage ranking details (dense/BM25 ranks, RRF, CrossEncoder logit, tier, adjustments)
python retrieval/diagnose.py "What hyperparameters were used for the DQN?" -v

# Scan the whole paper before reporting a method as absent
python retrieval/diagnose.py "Compare travel times between PI-ALINEA and DQN" --doc-id 2012.12104v1 --document-check

# Ablation: previous ordering without entity-aware re-ranking
python retrieval/diagnose.py "What hyperparameters were used for the DQN?" --no-entity-aware

# Query the legacy v3 index (kept for before/after comparison)
python retrieval/diagnose.py "What hyperparameters were used for the DQN?" --legacy-index --no-entity-aware
```

### Entity-aware retrieval and evidence validation

Semantic similarity does not guarantee that evidence concerns the method a question names. A "Hyperparameters used for training SAC agents" table is topically close to "What hyperparameters were used for the DQN?", but it does not answer it. The retrieval layer therefore adds three deterministic steps (no LLM calls):

| Module | Role |
|---|---|
| `retrieval/entities.py` | Detects method names. It uses a small, extensible registry of explicit acronym/expansion pairs (DQN = "deep Q-network") plus a shape-based fallback for unregistered names. Different methods are **never** declared synonyms. "DRL is a broader category that includes DQN" is recorded only to explain mismatches. |
| `retrieval/query_understanding.py` | Extracts the information need, method names, relation (`property_of` / `comparison` / `mention`), property terms and requested modality. The original query is still used for dense and BM25 retrieval. |
| `retrieval/evidence_validator.py` | Labels each evidence item (`SUPPORTS`, `PARTIAL`, `MENTIONS_ONLY`, `OTHER_METHOD`, `RELATED_CONTEXT`, `REFERENCE_LIST`, `UNRELATED`) and returns a structured `ValidationResult` with a status of `SUPPORTED`, `PARTIALLY_SUPPORTED`, `ENTITY_MISMATCH`, `INSUFFICIENT_EVIDENCE` or `NOT_FOUND_IN_DOCUMENT`. |
| `retrieval/ranking.py` | Applies entity-aware re-ranking on top of dense + BM25 → RRF → CrossEncoder (see below). |

`AcademicRetriever.search()` returns a `RetrievalResponse` with `results`, `query_analysis` and `validation`. Each result carries `uid`, `score_details`, `assessment` and structured `metadata`. The old `query` / `results` interface is unchanged.

**Absence is never inferred from top-k.** Without `document_check=True`, a missing method is reported as "not found in the retrieved evidence". `NOT_FOUND_IN_DOCUMENT` is only set after a scan of all indexed evidence of the paper(s).

**Ranking.** RRF scores and CrossEncoder logits are on incomparable scales and neither is a probability, so they are never added together. The relevance pipeline contributes only its *order*, converted to `base = 1 - (rank-1)/(N-1)`. Explicit evidence signals adjust that base in the same units, and three hard tiers apply:

- tier 0: evidence that ties a requested method to the requested property
- tier 1: everything else
- tier 2: bibliography chunks, unless the query asks about references

A wrong-method table can therefore never outrank method-specific evidence, but it stays visible as related context. The resulting `score` is an ordinal ranking score, not a probability or answer confidence. The weights are in `ranking.WEIGHTS`.

**Figures** are retrieved through their caption and the body text that cites them ("Fig. 8 presents…"). Image pixels are not embedded. The CrossEncoder is a text reranker.

### Index version (re-index required)

The index is now `academic_documents_v4` + `data/bm25_index_v4.json`. Parser table/figure ids such as `table_p8_1` repeat across papers. In v3 they silently overwrote each other (148 evidence items → 137 stored), which dropped, among others, the PI-ALINEA mean-travel-time table. v4 keys every item by `uid = "<doc_id>::<id>"` and stores derived metadata. The v3 collection and `data/bm25_index.json` are left untouched. Build v4 with:

```powershell
python retrieval/indexer.py --reset
```

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

The retrieval module is covered by 133 deterministic unit tests. They span schema serialization, BM25 indexing, hybrid fusion math, provenance preservation, entity detection, query understanding, evidence validation, entity-aware ranking, corpus isolation and the two observed failure cases (DQN hyperparameters; PI-ALINEA vs DQN). They need no model downloads: the end-to-end tests use a hashing embedder and in-memory ChromaDB.

```powershell
python -m pytest retrieval/tests -v
```

The live smoke test (requires the v4 index) checks retrieval plus the validation verdicts for the two failure cases:

```powershell
python retrieval/smoke_test.py -v
```
