import os
import sys
import shutil
import tempfile
import unittest
import json
import io
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import logging

# ===== Logging setup =====
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===== Add project root =====
project_root = Path(__file__).resolve().parent.parent.parent  # URAG_V2
sys.path.insert(0, str(project_root))

# ===== Import FAQ module =====
from src.llm.URag_F import FAQ

# Aliases cho dễ dùng
Provenance = FAQ.Provenance
FAQItem = FAQ.FAQItem
FAQIndex = FAQ.FAQIndex
LLM = FAQ.LLM
main = FAQ.main
build_enriched_faq = FAQ.build_enriched_faq
Tier1FAQAgent = FAQ.Tier1FAQAgent
read_jsonl = FAQ.read_jsonl
write_jsonl = FAQ.write_jsonl
simple_chunks = FAQ.simple_chunks


class TestFAQAdvanced(unittest.TestCase):
    def setUp(self):
        logger.info("=== Setting up test environment ===")
        self.tmpdir = tempfile.mkdtemp()
        logger.info(f"Temporary directory created: {self.tmpdir}")

        self.sample_prov = Provenance(source_type="seed_faq", source_id="test_id")
        self.sample_item = FAQItem(
            id="test1",
            question="Thủ đô của Việt Nam là gì?",
            answer="Hà Nội",
            canonical_id="c1",
            provenance=self.sample_prov
        )
        self.vn_item = FAQItem(
            id="test2",
            question="Ai là chủ tịch đầu tiên của Việt Nam?",
            answer="Hồ Chí Minh",
            canonical_id="c2",
            provenance=Provenance(source_type="seed_faq", source_id="test2")
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        logger.info(f"Temporary directory cleaned: {self.tmpdir}")
        logger.info("=== Test environment torn down ===")

    # ===== Model tests =====
    def test_provenance_model(self):
        prov = Provenance(source_type="doc_qa", doc_path="/path/to/doc", source_id="doc1")
        self.assertEqual(prov.source_type, "doc_qa")
        self.assertEqual(prov.source_id, "doc1")
        created = datetime.fromisoformat(prov.created_at)
        self.assertTrue((datetime.now(timezone.utc) - created).total_seconds() < 10)
        prov_dict = prov.model_dump()
        logger.info(f"Provenance dict: {prov_dict}")
        self.assertIn("created_at", prov_dict)

    def test_faqitem_model(self):
        item = self.vn_item
        self.assertEqual(item.answer, "Hồ Chí Minh")
        item_dict = item.model_dump()
        self.assertEqual(item_dict["provenance"]["source_type"], "seed_faq")

        with self.assertRaises(ValueError):
            FAQItem(id="1", question="", answer="A", canonical_id="c", provenance=self.sample_prov)
        with self.assertRaises(ValueError):
            FAQItem(id="1", question="Q", answer="", canonical_id="c", provenance=self.sample_prov)

    # ===== Utils =====
    def test_jsonl_utils(self):
        path = os.path.join(self.tmpdir, "test.jsonl")
        rows = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        write_jsonl(path, rows)
        loaded = read_jsonl(path)
        self.assertEqual(loaded, rows)

    def test_simple_chunks(self):
        text = "Câu một. Câu hai. Câu ba."
        chunks = simple_chunks(text, max_chars=10)
        self.assertEqual(chunks, ["Câu một.", "Câu hai.", "Câu ba."])

    # ===== FAQ Index =====
    @patch('sentence_transformers.SentenceTransformer')
    def test_faq_index(self, mock_st):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = [
            np.random.rand(2, 384).astype(np.float32),  # build
            np.random.rand(1, 384).astype(np.float32)   # search
        ]
        mock_st.return_value = mock_model

        idx = FAQIndex()
        idx.build([self.sample_item, self.vn_item])
        results = idx.search("Thủ đô VN?", top_k=2)
        self.assertEqual(len(results), 2)

    # ===== LLM Helpers =====
    @patch('src.llm.URag_F.FAQ.pipeline')
    def test_llm_helpers(self, mock_pipeline):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "PromptGenerated text"}]
        mock_pipeline.return_value = mock_pipe

        llm = LLM(model="gpt2")
        result = llm.generate("Prompt")
        self.assertEqual(result, "Generated text")

    # ===== Enriched FAQ =====
    @patch('src.llm.URag_F.FAQ.LLM')
    @patch('src.llm.URag_F.FAQ.parse_jsonl_block')
    @patch('src.llm.URag_F.FAQ.parse_bullets')
    @patch('sentence_transformers.SentenceTransformer')
    def test_build_enriched_faq(self, mock_st, mock_parse_bullets, mock_parse_jsonl, mock_llm):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_st.return_value = mock_model

        mock_llm.return_value.generate.side_effect = [
            '{"question": "Thủ đô VN?", "answer": "Hà Nội"}',
            "- Thủ đô của Việt Nam là gì?\n- VN capital?"
        ]
        mock_parse_jsonl.return_value = [{"question": "Thủ đô VN?", "answer": "Hà Nội"}]
        mock_parse_bullets.return_value = ["Thủ đô của Việt Nam là gì?", "VN capital?"]

        seed_path = os.path.join(self.tmpdir, "seed.jsonl")
        write_jsonl(seed_path, [{"question": "Seed Q", "answer": "Seed A"}])
        doc_path = Path(self.tmpdir) / "doc.txt"
        doc_path.write_text("Việt Nam có thủ đô là Hà Nội.", encoding="utf-8")
        out_dir = os.path.join(self.tmpdir, "out")

        stats = build_enriched_faq(doc_dir=self.tmpdir,
                                   initial_faq_path=seed_path,
                                   out_dir=out_dir,
                                   paraphrase_n=2)
        self.assertGreaterEqual(stats["count_total_enriched"], 3)

    # ===== Tier1 Agent =====
    @patch('sentence_transformers.SentenceTransformer')
    def test_tier1_faq_agent(self, mock_st):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_st.return_value = mock_model

        idx = FAQIndex()
        idx.build([self.vn_item])
        idx.save(self.tmpdir)

        agent = Tier1FAQAgent(self.tmpdir, threshold_faq=0.5, top_k=1)
        ans = agent.answer("Ai là chủ tịch đầu tiên VN?")
        self.assertFalse(ans["escalated"])

        # fake empty index
        empty_dir = os.path.join(self.tmpdir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        with open(os.path.join(empty_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"embed_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       "dimension": 384}, f)
        write_jsonl(os.path.join(empty_dir, "faq_items.jsonl"), [])
        np.save(os.path.join(empty_dir, "faq_matrix.npy"), np.zeros((0, 384), dtype=np.float32))

        agent_empty = Tier1FAQAgent(empty_dir, top_k=1)
        ans_empty = agent_empty.answer("Query")
        self.assertTrue(ans_empty["escalated"])

    # ===== CLI Tests =====
    @patch('sys.argv', ['script.py', 'build', '--docs', 'docs', '--out', 'out'])
    @patch('src.llm.URag_F.FAQ.build_enriched_faq')
    @patch('json.dumps')
    def test_cli_build(self, mock_dumps, mock_build):
        mock_build.return_value = {"stats": "mock"}
        main()
        mock_build.assert_called_once()
        mock_dumps.assert_called_once()

    @patch('sys.argv', ['script.py', 'chat', '--index', 'index'])
    @patch('builtins.input', side_effect=['Thủ đô VN?', 'exit'])
    @patch('src.llm.URag_F.FAQ.Tier1FAQAgent.answer')
    def test_cli_chat(self, mock_answer, mock_input):
        # fake index dir
        os.makedirs("index", exist_ok=True)
        with open("index/config.json", "w", encoding="utf-8") as f:
            json.dump({"embed_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       "dimension": 384}, f)
        write_jsonl("index/faq_items.jsonl", [])
        np.save("index/faq_matrix.npy", np.zeros((0, 384), dtype=np.float32))

        mock_answer.return_value = {"answer": "Hà Nội", "escalated": False,
                                   "score": None, "matched_question": None,
                                   "canonical_id": None, "provenance": {}}
        with patch('builtins.print') as mock_print:
            main()
            mock_answer.assert_called_once_with("Thủ đô VN?")
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            self.assertIn("FAQ agent ready.", "".join(print_calls))

    @patch('sys.argv', ['script.py', 'invalid_cmd'])
    def test_cli_invalid_command(self):
        fake_stderr = io.StringIO()
        with patch('sys.stderr', fake_stderr):
            with self.assertRaises(SystemExit):
                main()
        output = fake_stderr.getvalue().lower()
        self.assertIn("usage", output)


if __name__ == "__main__":
    unittest.main()
