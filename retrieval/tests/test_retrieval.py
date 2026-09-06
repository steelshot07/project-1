"""
retrieval/tests/test_retrieval.py

Retrieval pipeline test suite.

Test categories:
1. Normalizer: parser JSON → Evidence list
2. BM25Index: indexing and lexical search
3. Schema: Evidence / RetrievalResult / RetrievalResponse serialisation
4. Hybrid fusion: RRF correctness
5. Metadata & provenance preservation
6. Artifact path preservation

Tests that touch ChromaDB or the embedding model are skipped if the
environment is not available (CI-safe).

Run with:
    cd retrieval
    python -m pytest tests/test_retrieval.py -v
"""

import json
import os
import sys
import tempfile
import unittest

# Add the retrieval directory to path so imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from normalizer import normalize_document
from bm25_index import BM25Index, _tokenize
from schema import Evidence, RetrievalResult, RetrievalResponse
from retriever import reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "test_corpus.json"
)


def load_fixture() -> dict:
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Normalizer tests
# ---------------------------------------------------------------------------

class TestNormalizer(unittest.TestCase):

    def setUp(self):
        self.doc = load_fixture()
        self.evidence_list = normalize_document(self.doc)

    def test_returns_evidence_list(self):
        self.assertIsInstance(self.evidence_list, list)
        self.assertGreater(len(self.evidence_list), 0)

    def test_all_types_present(self):
        types = {ev.type for ev in self.evidence_list}
        self.assertIn("text", types, "Text evidence should be produced")
        self.assertIn("table", types, "Table evidence should be produced")
        self.assertIn("figure", types, "Figure evidence should be produced")

    def test_tables_have_correct_ids(self):
        table_ids = {ev.id for ev in self.evidence_list if ev.type == "table"}
        self.assertIn("table_p9_1", table_ids)
        self.assertIn("table_p10_1", table_ids)

    def test_figures_have_correct_ids(self):
        fig_ids = {ev.id for ev in self.evidence_list if ev.type == "figure"}
        self.assertIn("fig_p3_0", fig_ids)

    def test_table_content_includes_caption(self):
        table = next(ev for ev in self.evidence_list if ev.id == "table_p9_1")
        self.assertIn("Key parameters", table.content)

    def test_table_content_includes_cells(self):
        table = next(ev for ev in self.evidence_list if ev.id == "table_p9_1")
        # "Freeze interval F" is a cell value
        self.assertIn("Freeze interval F", table.content)

    def test_figure_content_is_caption(self):
        fig = next(ev for ev in self.evidence_list if ev.id == "fig_p3_0")
        self.assertIn("deep neural network", fig.content)

    def test_doc_id_preserved(self):
        for ev in self.evidence_list:
            self.assertEqual(ev.doc_id, "test_paper_1")

    def test_page_preserved_for_tables(self):
        table = next(ev for ev in self.evidence_list if ev.id == "table_p9_1")
        self.assertEqual(table.page, 9)

    def test_page_preserved_for_figures(self):
        fig = next(ev for ev in self.evidence_list if ev.id == "fig_p3_0")
        self.assertEqual(fig.page, 3)

    def test_artifact_path_for_table(self):
        table = next(ev for ev in self.evidence_list if ev.id == "table_p10_1")
        self.assertIsNotNone(table.artifact_path)
        self.assertIn("table_p10_1.txt", table.artifact_path)
        # Should use forward slashes
        self.assertNotIn("\\", table.artifact_path)

    def test_artifact_path_for_figure(self):
        fig = next(ev for ev in self.evidence_list if ev.id == "fig_p3_0")
        self.assertIsNotNone(fig.artifact_path)
        self.assertIn("fig_p3_0.png", fig.artifact_path)

    def test_artifact_path_none_for_text(self):
        text_ev = [ev for ev in self.evidence_list if ev.type == "text"]
        for ev in text_ev:
            self.assertIsNone(ev.artifact_path)

    def test_bbox_preserved_for_tables(self):
        table = next(ev for ev in self.evidence_list if ev.id == "table_p9_1")
        self.assertIsNotNone(table.bbox)
        self.assertEqual(len(table.bbox), 4)

    def test_caption_preserved_for_tables(self):
        table = next(ev for ev in self.evidence_list if ev.id == "table_p9_1")
        self.assertIsNotNone(table.caption)
        self.assertIn("Key parameters", table.caption)

    def test_section_assigned_to_text_chunks(self):
        text_chunks = [ev for ev in self.evidence_list if ev.type == "text"]
        sections = {ev.section for ev in text_chunks if ev.section}
        self.assertTrue(len(sections) > 0, "Text chunks should have section context")

    def test_text_page_derived_from_block_id(self):
        # With a small fixture all blocks may merge into one chunk
        # whose primary page = min(pages) = 9. Check metadata too.
        text_chunks = [ev for ev in self.evidence_list if ev.type == "text"]
        all_pages = set()
        for ev in text_chunks:
            if ev.page is not None:
                all_pages.add(ev.page)
            if ev.metadata.get("start_page") is not None:
                all_pages.add(ev.metadata["start_page"])
            if ev.metadata.get("end_page") is not None:
                all_pages.add(ev.metadata["end_page"])
        self.assertTrue(len(all_pages) > 0, "Text chunks should carry page info")
        self.assertTrue(
            9 in all_pages or 10 in all_pages,
            f"Expected page 9 or 10, got: {all_pages}",
        )


# ---------------------------------------------------------------------------
# 2. BM25Index tests
# ---------------------------------------------------------------------------

class TestBM25Index(unittest.TestCase):

    def setUp(self):
        doc = load_fixture()
        self.evidence_list = normalize_document(doc)
        # Use a temp file for persistence
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.index = BM25Index(index_path=self.tmp.name)
        self.index.add_evidence(self.evidence_list)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_index_length(self):
        self.assertEqual(len(self.index), len(self.evidence_list))

    def test_tokenizer(self):
        tokens = _tokenize("Freeze interval F = 10^4")
        self.assertIn("freeze", tokens)
        self.assertIn("interval", tokens)

    def test_freezing_interval_query(self):
        """'replay buffer' is unique to table_p9_1 in the fixture."""
        results = self.index.search("replay buffer size", top_k=5)
        ids = [eid for eid, _ in results]
        self.assertIn(
            "table_p9_1", ids,
            f"table_p9_1 not in BM25 top-5 for 'replay buffer size'. Got: {ids}"
        )

    def test_mean_travel_times_query(self):
        """'PI-ALINEA no controls' are unique to table_p10_1 in the fixture."""
        results = self.index.search("PI-ALINEA no controls", top_k=5)
        ids = [eid for eid, _ in results]
        self.assertIn(
            "table_p10_1", ids,
            f"table_p10_1 not in BM25 top-5 for 'PI-ALINEA no controls'. Got: {ids}"
        )

    def test_doc_id_filter(self):
        results = self.index.search("travel times", doc_id="nonexistent_doc")
        self.assertEqual(results, [])

    def test_type_filter_tables_only(self):
        results = self.index.search("parameters", types=["table"])
        for eid, _ in results:
            ev = next(e for e in self.evidence_list if e.id == eid)
            self.assertEqual(ev.type, "table")

    def test_type_filter_figures_only(self):
        results = self.index.search("architecture neural network", types=["figure"])
        for eid, _ in results:
            ev = next(e for e in self.evidence_list if e.id == eid)
            self.assertEqual(ev.type, "figure")

    def test_persistence_round_trip(self):
        self.index.save()
        reloaded = BM25Index(index_path=self.tmp.name)
        self.assertEqual(len(reloaded), len(self.index))
        results = reloaded.search("replay buffer size", top_k=5)
        ids = [eid for eid, _ in results]
        self.assertIn("table_p9_1", ids)

    def test_upsert_semantics(self):
        """Adding the same evidence again should not grow the index."""
        initial_len = len(self.index)
        self.index.add_evidence(self.evidence_list)
        self.assertEqual(len(self.index), initial_len)

    def test_scores_positive(self):
        results = self.index.search("Q-learning", top_k=5)
        for _, score in results:
            self.assertGreater(score, 0)

    def test_results_sorted_descending(self):
        results = self.index.search("deep reinforcement learning", top_k=10)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ---------------------------------------------------------------------------
# 3. Schema tests
# ---------------------------------------------------------------------------

class TestSchema(unittest.TestCase):

    def _make_evidence(self, **kwargs) -> Evidence:
        defaults = dict(
            id="table_p9_1",
            doc_id="test_paper_1",
            type="table",
            content="Key parameters...",
            page=9,
            section=None,
            bbox=[84.0, 279.0, 378.0, 409.0],
            caption="Table 2: Key parameters",
            artifact_path="data/processed/test_paper_1/tables/table_p9_1.txt",
        )
        defaults.update(kwargs)
        return Evidence(**defaults)

    def test_chroma_metadata_types(self):
        ev = self._make_evidence()
        meta = ev.to_chroma_metadata()
        for v in meta.values():
            self.assertIsInstance(v, (str, int, float, bool))

    def test_chroma_metadata_bbox_serialised(self):
        ev = self._make_evidence()
        meta = ev.to_chroma_metadata()
        self.assertIsInstance(meta["bbox"], str)
        parsed = json.loads(meta["bbox"])
        self.assertEqual(parsed, [84.0, 279.0, 378.0, 409.0])

    def test_from_chroma_result_round_trip(self):
        ev = self._make_evidence()
        meta = ev.to_chroma_metadata()
        restored = Evidence.from_chroma_result(ev.id, ev.content, meta)
        self.assertEqual(restored.id, ev.id)
        self.assertEqual(restored.doc_id, ev.doc_id)
        self.assertEqual(restored.type, ev.type)
        self.assertEqual(restored.page, ev.page)
        self.assertEqual(restored.bbox, ev.bbox)
        self.assertEqual(restored.artifact_path, ev.artifact_path)

    def test_none_page_serialised_as_minus_one(self):
        ev = self._make_evidence(page=None)
        meta = ev.to_chroma_metadata()
        self.assertEqual(meta["page"], -1)

    def test_from_chroma_result_minus_one_page_to_none(self):
        ev = self._make_evidence(page=None)
        meta = ev.to_chroma_metadata()
        restored = Evidence.from_chroma_result(ev.id, ev.content, meta)
        self.assertIsNone(restored.page)

    def test_retrieval_result_to_dict(self):
        r = RetrievalResult(
            id="table_p10_1",
            doc_id="test_paper_1",
            type="table",
            score=0.91,
            content="...",
            page=10,
            artifact_path="data/processed/test_paper_1/tables/table_p10_1.txt",
        )
        d = r.to_dict()
        self.assertEqual(d["id"], "table_p10_1")
        self.assertEqual(d["type"], "table")
        self.assertAlmostEqual(d["score"], 0.91)
        self.assertEqual(d["artifact_path"],
                         "data/processed/test_paper_1/tables/table_p10_1.txt")

    def test_retrieval_response_to_dict(self):
        r = RetrievalResult(
            id="fig_p3_0", doc_id="test_paper_1", type="figure",
            score=0.85, content="caption text", page=3,
            artifact_path="data/processed/test_paper_1/figures/fig_p3_0.png",
        )
        resp = RetrievalResponse(query="architecture", results=[r])
        d = resp.to_dict()
        self.assertEqual(d["query"], "architecture")
        self.assertEqual(len(d["results"]), 1)
        self.assertEqual(d["results"][0]["artifact_path"],
                         "data/processed/test_paper_1/figures/fig_p3_0.png")


# ---------------------------------------------------------------------------
# 4. RRF fusion tests
# ---------------------------------------------------------------------------

class TestRRFFusion(unittest.TestCase):

    def test_rrf_merges_lists(self):
        dense = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
        lexical = [("B", 5.0), ("C", 4.0), ("D", 3.0)]
        fused = reciprocal_rank_fusion([dense, lexical])
        ids = [eid for eid, _ in fused]
        # B and C appear in both lists — should rank highly
        self.assertIn("B", ids[:2])

    def test_rrf_scores_descending(self):
        dense = [("A", 0.9), ("B", 0.8)]
        lexical = [("B", 5.0), ("A", 4.0)]
        fused = reciprocal_rank_fusion([dense, lexical])
        scores = [s for _, s in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rrf_handles_empty_list(self):
        dense = [("A", 0.9), ("B", 0.8)]
        fused = reciprocal_rank_fusion([dense, []])
        self.assertEqual(len(fused), 2)

    def test_rrf_unique_candidates(self):
        dense = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
        lexical = [("C", 5.0), ("A", 4.0), ("D", 3.0)]
        fused = reciprocal_rank_fusion([dense, lexical])
        ids = [eid for eid, _ in fused]
        self.assertEqual(len(ids), len(set(ids)), "No duplicates in fused results")


# ---------------------------------------------------------------------------
# 5. Artifact path preservation (end-to-end without a real vector DB)
# ---------------------------------------------------------------------------

class TestArtifactPathPreservation(unittest.TestCase):

    def test_table_artifact_path_survives_normalizer(self):
        doc = load_fixture()
        evidence = normalize_document(doc)
        table = next(ev for ev in evidence if ev.id == "table_p10_1")
        self.assertEqual(
            table.artifact_path,
            "data/processed/test_paper_1/tables/table_p10_1.txt",
        )

    def test_figure_artifact_path_survives_normalizer(self):
        doc = load_fixture()
        evidence = normalize_document(doc)
        fig = next(ev for ev in evidence if ev.id == "fig_p3_0")
        self.assertEqual(
            fig.artifact_path,
            "data/processed/test_paper_1/figures/fig_p3_0.png",
        )

    def test_artifact_path_survives_chroma_round_trip(self):
        doc = load_fixture()
        evidence = normalize_document(doc)
        table = next(ev for ev in evidence if ev.id == "table_p10_1")

        meta = table.to_chroma_metadata()
        restored = Evidence.from_chroma_result(table.id, table.content, meta)
        self.assertEqual(restored.artifact_path, table.artifact_path)

    def test_retrieval_result_exposes_artifact_path(self):
        r = RetrievalResult(
            id="table_p10_1",
            doc_id="test_paper_1",
            type="table",
            score=0.91,
            content="...",
            page=10,
            artifact_path="data/processed/test_paper_1/tables/table_p10_1.txt",
        )
        d = r.to_dict()
        self.assertIn("artifact_path", d)
        self.assertIsNotNone(d["artifact_path"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
