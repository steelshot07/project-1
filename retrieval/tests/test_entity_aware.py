"""
retrieval/tests/test_entity_aware.py

Deterministic tests for entity-aware retrieval and method-aware evidence
validation.  No model downloads, no network, no external APIs:

- embeddings come from a hashing bag-of-words encoder (HashingEmbedder),
- ChromaDB runs in memory,
- the REAL normaliser, indexer, BM25 index, retriever, ranking and
  validator code paths are exercised.

Fixture: tests/fixtures/method_corpus.json — two synthetic papers that
reproduce the two observed failures:

  synth_grid : mentions DQN only as previous work; reports SAC and
               imitation-learning hyperparameter tables.
  synth_ramp : compares PI-ALINEA with the proposed DRL method on travel
               times; mentions deep Q-learning, never DQN.

Both papers deliberately contain a table with the parser id 'table_p8_1'
(regression test for the cross-document id collision).

Run with:
    cd retrieval
    python -m pytest tests/ -v
"""

import json
import os
import re
import sys
import tempfile
import unittest
import uuid
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb

from bm25_index import BM25Index
from entities import entity_names, find_entities, related, same_entity
from evidence_signals import derive_metadata, is_reference_list
from evidence_validator import (
    ENTITY_MISMATCH, INSUFFICIENT_EVIDENCE, NOT_FOUND_IN_DOCUMENT,
    OTHER_METHOD, PARTIALLY_SUPPORTED, REFERENCE_LIST, SUPPORTED, SUPPORTS,
    EvidenceAssessment, assess_evidence, validate,
)
from indexer import DocumentIndexer
from normalizer import normalize_document
from query_understanding import analyze_query
from ranking import (
    TIER_DEFAULT, TIER_METHOD_SUPPORT, TIER_REFERENCE, base_score,
    rank_candidates,
)
from retriever import AcademicRetriever
from schema import Evidence, RetrievalResponse, RetrievalResult, make_uid


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "method_corpus.json")

DQN_HP = "What hyperparameters were used for the DQN?"
PI_DQN = "Compare travel times between PI-ALINEA and DQN"
SAC_HP = "What are the hyperparameters used by SAC?"
MEAN_TT = "What were the mean travel times in the experiments?"
FIG_TT = "Show the figure comparing vehicle travel times."
UNRELATED = "What is the boiling point of liquid nitrogen at sea level?"


def load_docs():
    with open(FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


class HashingEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode()."""

    DIM = 512
    _TOK = re.compile(r"[a-z0-9]+")

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._TOK.findall(t.lower()):
                tok = tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok
                out[i, zlib.crc32(tok.encode()) % self.DIM] += 1.0
            n = np.linalg.norm(out[i])
            if normalize_embeddings and n > 0:
                out[i] /= n
        return out


def build_index():
    """Index the fixture with the real DocumentIndexer into memory."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(f"test_{uuid.uuid4().hex[:8]}")
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    os.unlink(tmp.name)
    bm25 = BM25Index(index_path=tmp.name)
    embedder = HashingEmbedder()
    indexer = DocumentIndexer.from_components(embedder, collection, bm25)
    for doc in load_docs():
        indexer.index_parsed(doc, show_progress=False)
    retriever = AcademicRetriever.from_components(embedder, collection, bm25)
    return retriever, collection, bm25, tmp.name


class _RetrieverCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever, cls.collection, cls.bm25, cls.bm25_path = build_index()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.bm25_path):
            os.unlink(cls.bm25_path)

    def search(self, q, **kw):
        kw.setdefault("top_k", 5)
        kw.setdefault("use_reranker", False)
        return self.retriever.search(q, **kw)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

class TestEntities(unittest.TestCase):

    def test_alinea_not_matched_inside_pi_alinea(self):
        self.assertEqual(entity_names("PI-ALINEA was tuned"), ["PI-ALINEA"])
        self.assertEqual(entity_names("PI-ALINEA extends ALINEA"), ["PI-ALINEA", "ALINEA"])

    def test_dqn_and_drl_are_not_the_same_entity(self):
        self.assertFalse(same_entity("DQN", "DRL"))
        self.assertFalse(same_entity("DQN", "Deep Q-learning"))
        self.assertFalse(same_entity("SAC", "Imitation learning"))

    def test_explicit_expansions_are_the_same_entity(self):
        self.assertTrue(same_entity("DQN", "deep Q-network"))
        self.assertTrue(same_entity("SAC", "soft actor-critic"))
        self.assertTrue(same_entity("DRL", "deep reinforcement learning"))

    def test_relatedness_is_informational_only(self):
        self.assertIsNotNone(related("DQN", "DRL"))
        self.assertFalse(same_entity("DQN", "DRL"))

    def test_longest_match_wins(self):
        self.assertEqual(entity_names("deep Q-learning agent"), ["Deep Q-learning"])
        self.assertEqual(entity_names("deep reinforcement learning (DRL)"), ["DRL"])

    def test_capitalised_words_are_not_entities(self):
        self.assertEqual(entity_names("What Were The Main Results Of This Paper?"), [])

    def test_ordinary_acronyms_are_not_entities(self):
        self.assertEqual(entity_names("Published by IEEE, see the PDF; RMSE in MW"), [])

    def test_unregistered_method_shapes_detected(self):
        names = entity_names("FooNet outperforms ResNet-50 and XYZ-NET")
        for n in ("FooNet", "ResNet-50", "XYZ-NET"):
            self.assertIn(n, names)
        sources = {m.canonical: m.source for m in find_entities("FooNet beats DQN")}
        self.assertEqual(sources["FooNet"], "pattern")
        self.assertEqual(sources["DQN"], "registry")


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------

class TestQueryUnderstanding(unittest.TestCase):

    def test_dqn_hyperparameters(self):
        a = analyze_query(DQN_HP)
        self.assertEqual(a.entities, ["DQN"])
        self.assertEqual(a.information_need, "hyperparameters")
        self.assertEqual(a.relation, "property_of")
        self.assertEqual(a.original_query, DQN_HP)

    def test_comparison_between_methods(self):
        a = analyze_query(PI_DQN)
        self.assertEqual(a.entities, ["PI-ALINEA", "DQN"])
        self.assertEqual(a.relation, "comparison")
        self.assertIn("travel", a.property_terms)
        self.assertIn("time", a.property_terms)

    def test_figure_modality_detected(self):
        a = analyze_query(FIG_TT)
        self.assertEqual(a.requested_modalities, ["figure"])
        self.assertEqual(a.entities, [])

    def test_table_modality_detected(self):
        self.assertEqual(analyze_query("Which table lists the learning rate?").requested_modalities,
                         ["table"])

    def test_unknown_query_falls_back(self):
        a = analyze_query(UNRELATED)
        self.assertEqual(a.entities, [])
        self.assertEqual(a.relation, "general")
        self.assertEqual(a.information_need, "general")

    def test_doc_filter_recorded(self):
        self.assertEqual(analyze_query(DQN_HP, doc_id="synth_grid").doc_id, "synth_grid")

    def test_focused_query_keeps_entity_and_property(self):
        fq = analyze_query(DQN_HP).focused_query()
        self.assertIn("DQN", fq)
        self.assertIn("hyperparameters", fq)


# ---------------------------------------------------------------------------
# Evidence signals / metadata
# ---------------------------------------------------------------------------

class TestEvidenceSignals(unittest.TestCase):

    def test_bibliography_detected(self):
        refs = ("[1] V. Mnih, K. Kavukcuoglu et al., Human-level control, Nature, vol. 518, "
                "pp. 529-533, 2015. [2] T. Haarnoja, A. Zhou, and S. Levine. Soft actor-critic. "
                "In Proc. ICML, 2018. [3] J. Duan et al., IEEE Trans. Power Systems, 2020.")
        self.assertTrue(is_reference_list(refs))

    def test_citation_heavy_table_is_not_bibliography(self):
        table = ("Feature extraction | Protected templates | Feature vectors\n"
                 "FaceNet [30] | 96.3 % | 86.0 %\nDlib [50] | 75.0 % | 87.3 %\n"
                 "LightCNN [31] | 96.0 % | 85.3 %")
        self.assertFalse(is_reference_list(table))

    def test_prose_with_inline_citations_is_not_bibliography(self):
        prose = ("Ramp metering reduces the total time spent in the traffic system [1, 2]. "
                 "Several field tests have demonstrated its benefits [3], and DRL methods "
                 "were recently proposed for signal control [4] with promising results.")
        self.assertFalse(is_reference_list(prose))

    def test_metadata_method_only_when_unambiguous(self):
        m = derive_metadata("table", "x", caption="Table 5: Hyperparameters used for training SAC agents")
        self.assertEqual(m["method"], "SAC")
        self.assertEqual(m["evidence_role"], "hyperparameters")
        m2 = derive_metadata("figure", "x", caption="Figure 11: Queue length for a) PI-ALINEA and b) DRL.")
        self.assertNotIn("method", m2)  # two entities -> uncertain -> omitted
        m3 = derive_metadata("figure", "x", caption="Figure 3: State representation.")
        self.assertEqual(m3["evidence_role"], "unknown")  # nothing fabricated

    def test_metadata_is_chroma_safe(self):
        for ev in normalize_document(load_docs()[1]):
            for k, v in ev.to_chroma_metadata().items():
                self.assertIsInstance(v, (str, int, float, bool), k)


# ---------------------------------------------------------------------------
# Test 8: normalisation regression
# ---------------------------------------------------------------------------

class TestNormalizationRegression(unittest.TestCase):

    def setUp(self):
        self.docs = load_docs()

    def test_page_and_section_text_not_duplicated(self):
        """Every block text appears in exactly one chunk (overlap disabled)."""
        doc = self.docs[0]
        chunks = [e for e in normalize_document(doc, chunk_size=1000, overlap=0)
                  if e.type == "text"]
        joined = "\n".join(c.content for c in chunks)
        for page in doc["pages"]:
            for block in page["blocks"]:
                if len(block["text"]) >= 5:
                    self.assertEqual(joined.count(block["text"]), 1, block["block_id"])

    def test_duplicate_block_id_in_pages_indexed_once(self):
        chunks = [e for e in normalize_document(self.docs[0], chunk_size=1000, overlap=0)
                  if e.type == "text"]
        block_ids = ",".join(c.metadata["block_ids"] for c in chunks).split(",")
        self.assertEqual(block_ids.count("p1_b1"), 1)

    def test_references_form_their_own_chunk(self):
        chunks = [e for e in normalize_document(self.docs[0], chunk_size=1000, overlap=0)
                  if e.type == "text"]
        ref = [c for c in chunks if "Haarnoja" in c.content]
        self.assertEqual(len(ref), 1)
        self.assertNotIn("Deep reinforcement learning (DRL) has been applied", ref[0].content)
        self.assertTrue(ref[0].metadata["is_reference_list"])
        appendix = [c for c in chunks if "Table 5 and Table 6" in c.content]
        self.assertFalse(appendix[0].metadata["is_reference_list"])

    def test_table_structure_preserved(self):
        ev = next(e for e in normalize_document(self.docs[0]) if e.id == "table_p8_1")
        self.assertEqual(json.loads(ev.metadata["table_headers"]),
                         ["Hyperparameter", "Description", "Value"])
        self.assertIn("lr", json.loads(ev.metadata["row_labels"]))
        rows = json.loads(ev.metadata["table_data"])
        self.assertIn(["lr", "learning rate", "0.0003"], rows)
        # value stays on the same searchable line as its parameter name
        self.assertIn("lr | learning rate | 0.0003", ev.content)

    def test_figure_context_from_citing_text(self):
        ev = next(e for e in normalize_document(self.docs[1]) if e.id == "fig_p10_0")
        self.assertEqual(ev.metadata["label"], "Figure 8")
        self.assertEqual(ev.metadata["referenced_by"], "p9_b1")
        self.assertIn("Referenced in text:", ev.content)
        self.assertEqual(ev.caption, "Figure 8: Vehicle travel times in the mainline.")
        self.assertEqual(ev.artifact_path, "data/processed/synth_ramp/figures/fig_p10_0.png")


# ---------------------------------------------------------------------------
# Index: uid keys, collisions, schema compatibility
# ---------------------------------------------------------------------------

class TestUidAndCollisions(_RetrieverCase):

    def test_no_items_lost_to_id_collisions(self):
        total = sum(len(normalize_document(d)) for d in load_docs())
        self.assertEqual(self.collection.count(), total)
        self.assertEqual(len(self.bm25), total)

    def test_same_parser_id_in_two_documents_both_retrievable(self):
        got = self.collection.get(
            ids=[make_uid("synth_grid", "table_p8_1"), make_uid("synth_ramp", "table_p8_1")],
            include=["metadatas"],
        )
        self.assertEqual(sorted(m["doc_id"] for m in got["metadatas"]),
                         ["synth_grid", "synth_ramp"])
        self.assertTrue(all(m["id"] == "table_p8_1" for m in got["metadatas"]))

    def test_reindex_replaces_document(self):
        before = self.collection.count()
        indexer = DocumentIndexer.from_components(HashingEmbedder(), self.collection, self.bm25)
        indexer.index_parsed(load_docs()[0], show_progress=False)
        self.assertEqual(self.collection.count(), before)
        self.assertEqual(len(self.bm25), before)

    def test_from_chroma_result_restores_parser_id(self):
        ev = normalize_document(load_docs()[0])[0]
        restored = Evidence.from_chroma_result(ev.uid, ev.content, ev.to_chroma_metadata())
        self.assertEqual(restored.id, ev.id)
        self.assertEqual(restored.uid, ev.uid)

    def test_bm25_default_returns_parser_ids_for_backward_compat(self):
        ids = [e for e, _ in self.bm25.search("replay buffer", top_k=5)]
        self.assertIn("table_p7_1", ids)
        uids = [e for e, _ in self.bm25.search("replay buffer", top_k=5, return_uids=True)]
        self.assertIn("synth_ramp::table_p7_1", uids)

    def test_legacy_response_constructor_still_works(self):
        r = RetrievalResult(id="x", doc_id="d", type="text", score=0.1, content="c")
        d = RetrievalResponse(query="q", results=[r]).to_dict()
        self.assertEqual(set(d), {"query", "results"})
        self.assertEqual(d["results"][0]["uid"], "d::x")


# ---------------------------------------------------------------------------
# Test 1: exact method hyperparameters (DQN — not reported)
# ---------------------------------------------------------------------------

class TestDQNHyperparameters(_RetrieverCase):

    def test_corpus_wide(self):
        resp = self.search(DQN_HP)
        v = resp.validation
        self.assertEqual(v.status, ENTITY_MISMATCH)
        self.assertTrue(v.entity_mismatch)
        self.assertEqual(v.requested_entities, ["DQN"])
        self.assertNotIn("DQN", v.supported_entities)
        self.assertEqual(v.supporting_evidence_ids, [])
        # Mentioned (as previous work) is distinguished from established.
        self.assertIn("DQN", v.mentioned_only_entities)
        self.assertIn("SAC", v.mismatched_entities)
        self.assertIn("must not be attributed to DQN", v.explanation)

    def test_sac_and_il_tables_are_context_not_dqn_evidence(self):
        resp = self.search(DQN_HP, doc_id="synth_grid")
        by_uid = {r.uid: r for r in resp.results}
        sac = by_uid.get("synth_grid::table_p8_1")
        il = by_uid.get("synth_grid::table_p8_2")
        self.assertIsNotNone(sac, "SAC table should still be shown as related context")
        self.assertIsNotNone(il, "IL table should still be shown as related context")
        for r in (sac, il):
            self.assertEqual(r.assessment["label"], OTHER_METHOD)
            self.assertNotIn("DQN", r.assessment["requested_associated"])
        self.assertIn("SAC", sac.assessment["other_associated"])
        self.assertIn("Imitation learning", il.assessment["other_associated"])

    def test_mention_of_dqn_is_retrieved_with_provenance(self):
        resp = self.search(DQN_HP, doc_id="synth_grid")
        mention = [r for r in resp.results if "DQN" in r.assessment["requested_mentioned"]]
        self.assertTrue(mention)
        self.assertEqual(mention[0].assessment["label"], "MENTIONS_ONLY")
        self.assertEqual(mention[0].page, 1)
        self.assertEqual(mention[0].doc_id, "synth_grid")

    def test_document_check_reports_not_found_in_document(self):
        resp = self.search(DQN_HP, doc_id="synth_grid", document_check=True)
        v = resp.validation
        self.assertEqual(v.status, NOT_FOUND_IN_DOCUMENT)
        self.assertEqual(v.scope, "document")
        self.assertEqual(v.document_findings["synth_grid"]["DQN"]["status"], "mentioned_only")
        self.assertTrue(v.entity_mismatch, "mismatch must remain visible")

    def test_without_document_check_absence_is_not_claimed(self):
        v = self.search(DQN_HP, doc_id="synth_grid").validation
        self.assertNotEqual(v.status, NOT_FOUND_IN_DOCUMENT)
        self.assertTrue(any("document-level check" in c for c in v.caveats))

    def test_reference_list_ranked_below_content(self):
        resp = self.search(DQN_HP, doc_id="synth_grid", top_k=10)
        tiers = [r.score_details["tier"] for r in resp.results]
        refs = [i for i, r in enumerate(resp.results) if r.assessment["label"] == REFERENCE_LIST]
        for i in refs:
            self.assertEqual(tiers[i], TIER_REFERENCE)
            self.assertTrue(all(t <= TIER_REFERENCE for t in tiers[:i]))
            self.assertTrue(all(resp.results[j].assessment["label"] == REFERENCE_LIST
                                for j in range(i, len(resp.results))))


# ---------------------------------------------------------------------------
# Test 2: method-specific comparison (PI-ALINEA vs DQN)
# ---------------------------------------------------------------------------

class TestPIAlineaVersusDQN(_RetrieverCase):

    def test_travel_time_evidence_retained(self):
        resp = self.search(PI_DQN)
        uids = [r.uid for r in resp.results]
        self.assertIn("synth_ramp::table_p8_1", uids)
        self.assertIn("synth_ramp::fig_p10_0", uids)

    def test_comparison_not_established_and_drl_not_substituted(self):
        v = self.search(PI_DQN).validation
        self.assertEqual(v.status, ENTITY_MISMATCH)
        self.assertTrue(v.entity_mismatch)
        # Corpus-wide, the other paper may MENTION DQN; it is never established.
        self.assertIn("DQN", v.missing_entities + v.mentioned_only_entities)
        self.assertNotIn("DQN", v.supported_entities)
        self.assertIn("DQN", self.search(PI_DQN, doc_id="synth_ramp").validation.missing_entities)
        self.assertIn("PI-ALINEA", v.supported_entities)
        self.assertIn(["DRL", "PI-ALINEA"], v.established_comparisons)
        self.assertIn("DRL", v.mismatched_entities)
        self.assertIn("NOT established", v.explanation)
        self.assertTrue(any("DRL is not treated as DQN" in n for n in v.related_entity_notes))

    def test_document_check_keeps_figure_and_flags_absence(self):
        resp = self.search(PI_DQN, doc_id="synth_ramp", document_check=True)
        v = resp.validation
        self.assertEqual(v.status, NOT_FOUND_IN_DOCUMENT)
        self.assertEqual(v.document_findings["synth_ramp"]["DQN"]["status"], "not_found")
        self.assertEqual(v.document_findings["synth_ramp"]["PI-ALINEA"]["status"], "associated")
        self.assertIn("synth_ramp::fig_p10_0", [r.uid for r in resp.results])
        self.assertIn(["DRL", "PI-ALINEA"], v.established_comparisons)

    def test_valid_comparison_is_supported(self):
        v = self.search("Compare travel times between PI-ALINEA and DRL").validation
        self.assertEqual(v.status, SUPPORTED)
        self.assertFalse(v.entity_mismatch)


# ---------------------------------------------------------------------------
# Test 3: valid method-specific query
# ---------------------------------------------------------------------------

class TestSACHyperparameters(_RetrieverCase):

    def test_sac_table_ranks_first_and_is_supported(self):
        resp = self.search(SAC_HP)
        self.assertEqual(resp.results[0].uid, "synth_grid::table_p8_1")
        self.assertEqual(resp.results[0].assessment["label"], SUPPORTS)
        v = resp.validation
        self.assertEqual(v.status, SUPPORTED)
        self.assertIn("synth_grid::table_p8_1", v.supporting_evidence_ids)
        self.assertFalse(v.entity_mismatch)

    def test_wrong_method_table_never_outranks_supporting_evidence(self):
        resp = self.search(SAC_HP, top_k=10)
        labels = [r.assessment["label"] for r in resp.results]
        first_other = labels.index(OTHER_METHOD) if OTHER_METHOD in labels else len(labels)
        last_support = max(i for i, l in enumerate(labels) if l in (SUPPORTS, "PARTIAL"))
        self.assertLess(last_support, first_other)


# ---------------------------------------------------------------------------
# Test 4: valid table query
# ---------------------------------------------------------------------------

class TestMeanTravelTimes(_RetrieverCase):

    def test_table_ranks_highly_with_values_and_provenance(self):
        resp = self.search(MEAN_TT)
        top2 = [r.uid for r in resp.results[:2]]
        self.assertIn("synth_ramp::table_p8_1", top2)
        t = next(r for r in resp.results if r.uid == "synth_ramp::table_p8_1")
        self.assertIn("179 [-4%]", t.content)
        self.assertIn("PI-ALINEA | DRL", t.content)
        self.assertEqual(t.page, 10)
        self.assertEqual(t.bbox, [72.0, 400.0, 520.0, 560.0])
        self.assertEqual(t.artifact_path, "data/processed/synth_ramp/tables/table_p8_1.txt")
        self.assertIn(["Mean", "201", "176", "172"], json.loads(t.metadata["table_data"]))
        self.assertEqual(resp.validation.status, SUPPORTED)


# ---------------------------------------------------------------------------
# Test 5: figure query
# ---------------------------------------------------------------------------

class TestFigureQuery(_RetrieverCase):

    def test_travel_time_figure_returned_with_artifact(self):
        resp = self.search(FIG_TT, top_k=3)
        figs = [r for r in resp.results if r.type == "figure"]
        self.assertTrue(figs)
        self.assertEqual(figs[0].uid, "synth_ramp::fig_p10_0")
        self.assertEqual(figs[0].artifact_path,
                         "data/processed/synth_ramp/figures/fig_p10_0.png")
        self.assertEqual(figs[0].page, 10)
        self.assertIn("Vehicle travel times", figs[0].caption)
        self.assertEqual(figs[0].score_details["adjustments"].get("requested_modality"), 0.2)
        self.assertTrue(any("pixels" in n for n in figs[0].assessment["notes"]))

    def test_no_forced_figure_for_text_query(self):
        resp = self.search(SAC_HP, top_k=3)
        self.assertFalse(any(r.score_details["adjustments"].get("requested_modality")
                             for r in resp.results))


# ---------------------------------------------------------------------------
# Test 6: unrelated query
# ---------------------------------------------------------------------------

class TestUnrelatedQuery(_RetrieverCase):

    def test_no_false_support(self):
        resp = self.search(UNRELATED)
        self.assertTrue(resp.results, "retrieval still returns nearest evidence")
        self.assertEqual(resp.validation.status, INSUFFICIENT_EVIDENCE)
        self.assertEqual(resp.validation.supporting_evidence_ids, [])

    def test_unknown_method_not_supported(self):
        v = self.search("What hyperparameters were used for FooNet?").validation
        self.assertNotEqual(v.status, SUPPORTED)
        self.assertIn("FooNet", v.missing_entities)


# ---------------------------------------------------------------------------
# Test 7: provenance regression
# ---------------------------------------------------------------------------

class TestProvenance(_RetrieverCase):

    def test_every_result_keeps_provenance(self):
        for q in (DQN_HP, PI_DQN, SAC_HP, MEAN_TT, FIG_TT, UNRELATED):
            for r in self.search(q, top_k=8).results:
                d = r.to_dict()
                self.assertTrue(d["id"])
                self.assertEqual(d["uid"], make_uid(d["doc_id"], d["id"]))
                self.assertIn(d["doc_id"], ("synth_grid", "synth_ramp"))
                self.assertIn(d["type"], ("text", "table", "figure"))
                self.assertIsNotNone(d["page"])
                if d["type"] in ("table", "figure"):
                    self.assertTrue(d["caption"])
                    self.assertEqual(len(d["bbox"]), 4)
                    self.assertTrue(d["artifact_path"])
                    self.assertNotIn("\\", d["artifact_path"])
                else:
                    self.assertIsNone(d["artifact_path"])
                self.assertIn("score_details", d)
                self.assertIn("assessment", d)

    def test_response_serialises_analysis_and_validation(self):
        d = self.search(DQN_HP).to_dict()
        json.dumps(d)  # must be JSON-serialisable for downstream consumers
        self.assertEqual(d["query_analysis"]["entities"], ["DQN"])
        self.assertEqual(d["validation"]["status"], ENTITY_MISMATCH)

    def test_scores_are_not_probabilities(self):
        for r in self.search(DQN_HP).results:
            self.assertIn("not a probability", r.score_details["score_semantics"])


# ---------------------------------------------------------------------------
# Test 9: corpus isolation
# ---------------------------------------------------------------------------

class TestCorpusIsolation(_RetrieverCase):

    def test_dense_respects_doc_filter(self):
        for q in (PI_DQN, DQN_HP):
            for uid, _ in self.retriever._dense_retrieve(q, 50, "synth_grid", None):
                self.assertTrue(uid.startswith("synth_grid::"), uid)

    def test_bm25_respects_doc_filter(self):
        for q in (PI_DQN, DQN_HP, "travel times PI-ALINEA"):
            for uid, _ in self.retriever._lexical_retrieve(q, 50, "synth_grid", None):
                self.assertTrue(uid.startswith("synth_grid::"), uid)

    def test_search_respects_doc_filter_including_document_check(self):
        resp = self.search(PI_DQN, doc_id="synth_grid", top_k=10, document_check=True)
        self.assertTrue(all(r.doc_id == "synth_grid" for r in resp.results))
        self.assertEqual(resp.validation.documents_checked, ["synth_grid"])

    def test_type_filter(self):
        resp = self.search(MEAN_TT, types=["figure"], top_k=10)
        self.assertTrue(resp.results)
        self.assertTrue(all(r.type == "figure" for r in resp.results))


# ---------------------------------------------------------------------------
# Diversification still works on the new pipeline
# ---------------------------------------------------------------------------

class TestDiversificationIntegration(_RetrieverCase):

    def test_caps_respected(self):
        resp = self.search(PI_DQN, top_k=4, diversify=True,
                           max_per_type={"text": 1, "table": 1, "figure": 2})
        types = [r.type for r in resp.results]
        self.assertLessEqual(types.count("text"), 1)
        self.assertLessEqual(types.count("table"), 1)
        self.assertIn("figure", types)

    def test_entity_aware_can_be_disabled(self):
        resp = self.search(DQN_HP, entity_aware=False)
        self.assertTrue(resp.results)
        self.assertIn("legacy", resp.results[0].score_details["score_semantics"])
        self.assertIsNotNone(resp.validation)  # validation is independent of ranking


# ---------------------------------------------------------------------------
# Ranking unit tests (pure)
# ---------------------------------------------------------------------------

def _assessment(uid, label, other=None):
    return EvidenceAssessment(uid=uid, evidence_id=uid, doc_id="d", type="table",
                              page=1, label=label, other_associated=other or {})


class TestRanking(unittest.TestCase):

    def test_base_score_is_rank_based(self):
        self.assertEqual(base_score(1, 5), 1.0)
        self.assertEqual(base_score(5, 5), 0.0)
        self.assertEqual(base_score(1, 1), 1.0)

    def test_wrong_method_rank1_cannot_outrank_support_rank20(self):
        a = analyze_query(DQN_HP)
        ordered = [("wrong", "table")] + [(f"x{i}", "text") for i in range(18)] + [("right", "table")]
        assess = {"wrong": _assessment("wrong", OTHER_METHOD, {"SAC": "caption"}),
                  "right": _assessment("right", SUPPORTS)}
        ranked = rank_candidates(ordered, a, assess)
        self.assertEqual(ranked[0].uid, "right")
        self.assertEqual(ranked[0].tier, TIER_METHOD_SUPPORT)
        self.assertIn("wrong", [c.uid for c in ranked])  # demoted, not hidden

    def test_related_family_penalised_less_than_unrelated_method(self):
        a = analyze_query(DQN_HP)
        ordered = [("sac", "table"), ("dql", "table")]
        assess = {"sac": _assessment("sac", OTHER_METHOD, {"SAC": "caption"}),
                  "dql": _assessment("dql", OTHER_METHOD, {"Deep Q-learning": "caption"})}
        ranked = {c.uid: c for c in rank_candidates(ordered, a, assess)}
        self.assertIn("other_method", ranked["sac"].adjustments)
        self.assertIn("related_method", ranked["dql"].adjustments)

    def test_wrong_method_property_evidence_above_no_signal_evidence(self):
        """SAC hyperparameters are still useful context for a DQN question;
        an item with neither the method nor the property is not."""
        a = analyze_query(DQN_HP)
        ordered = [(f"x{i}", "text") for i in range(20)]
        ordered[1] = ("junk", "text")
        ordered[2] = ("sac", "table")
        assess = {u: _assessment(u, "RELATED_CONTEXT") for u, _ in ordered}
        assess["junk"] = _assessment("junk", "UNRELATED")
        assess["sac"] = _assessment("sac", OTHER_METHOD, {"SAC": "caption"})
        order = [c.uid for c in rank_candidates(ordered, a, assess)]
        self.assertLess(order.index("sac"), order.index("junk"))

    def test_general_query_keeps_relevance_order(self):
        a = analyze_query("ramp metering overview")
        ordered = [(f"u{i}", "text") for i in range(5)]
        assess = {u: _assessment(u, "RELATED_CONTEXT") for u, _ in ordered}
        ranked = rank_candidates(ordered, a, assess)
        self.assertEqual([c.uid for c in ranked], [u for u, _ in ordered])
        self.assertTrue(all(c.tier == TIER_DEFAULT for c in ranked))

    def test_reference_list_tier(self):
        a = analyze_query("ramp metering overview")
        ordered = [("ref", "text"), ("body", "text")]
        assess = {"ref": _assessment("ref", REFERENCE_LIST),
                  "body": _assessment("body", "RELATED_CONTEXT")}
        self.assertEqual([c.uid for c in rank_candidates(ordered, a, assess)], ["body", "ref"])

    def test_reference_query_does_not_demote_references(self):
        a = analyze_query("Which references are cited for soft actor-critic?")
        refs = Evidence(id="r", doc_id="d", type="text", section="References",
                        content="[1] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic. "
                                "In Proc. ICML, 2018. [2] J. Duan et al., IEEE Trans. Power Systems, "
                                "vol. 35, no. 1, pp. 814-817, 2020.")
        self.assertNotEqual(assess_evidence(refs, a).label, REFERENCE_LIST)


# ---------------------------------------------------------------------------
# Validator unit tests (pure)
# ---------------------------------------------------------------------------

class TestValidatorUnits(unittest.TestCase):

    def _table(self, caption, rows, tid="t1"):
        content = "\n".join([caption] + [" | ".join(r) for r in rows])
        return Evidence(id=tid, doc_id="d", type="table", content=content, page=1, caption=caption)

    def test_caption_alone_is_not_full_support(self):
        """A mis-attached caption: body has no hyperparameter vocabulary."""
        ev = self._table("Table 5: Hyperparameters used for training SAC agents",
                         [["Name", "Description", "Unit"], ["V m", "voltage magnitude", "p.u"],
                          ["n s", "state dimension", "1442"]])
        a = assess_evidence(ev, analyze_query(SAC_HP))
        self.assertTrue(a.caption_only)
        self.assertEqual(a.label, "PARTIAL")
        v = validate(analyze_query(SAC_HP), [ev])
        self.assertEqual(v.status, PARTIALLY_SUPPORTED)

    def test_row_level_association_in_property_table(self):
        ev = self._table("Table 3: Policy performance details",
                         [["Policy", "Unsolvable cases"], ["random policy", "1.3%"],
                          ["SAC agent", "1.6%"], ["imitation learning agent", "2.1%"]])
        a = assess_evidence(ev, analyze_query("What was the performance of SAC?"))
        self.assertIn("SAC", a.requested_associated)

    def test_mention_query(self):
        ev = Evidence(id="c", doc_id="d", type="text", page=1,
                      content="We use soft actor-critic (SAC) as our algorithm.")
        v = validate(analyze_query("What is SAC?"), [ev])
        self.assertEqual(v.status, PARTIALLY_SUPPORTED)

    def test_empty_evidence(self):
        v = validate(analyze_query(DQN_HP), [])
        self.assertEqual(v.status, INSUFFICIENT_EVIDENCE)
        self.assertEqual(v.missing_entities, ["DQN"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
