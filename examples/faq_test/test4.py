#!/usr/bin/env python3
"""Comprehensive Test Suite for FAQ Agent
Tests all functionality, edge cases, performance, and potential bugs
Compatible with the provided project structure"""

import os
import sys
import json
import uuid
import shutil
import tempfile
import threading
import time
import random
import string
import unittest
import io
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import logging

# ===== Logging setup =====
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===== Add project root =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

# Import the FAQ module
try:
    from src.llm.URag_F.FAQ import (
        Provenance, FAQItem, FAQIndex, LLM, Tier1FAQAgent,
        build_enriched_faq, read_jsonl, write_jsonl, simple_chunks,
        parse_jsonl_block, parse_bullets, QA_EXTRACT_PROMPT, PARAPHRASE_PROMPT,
        main
    )
except ImportError as e:
    logger.error("Could not import FAQ module. Make sure FAQ.py is in the correct directory.")
    sys.exit(1)

# ===== MOCKING LIBRARIES =====
def mock_sentence_transformer(model_name):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_model.encode.return_value = np.zeros((1, 384)).astype(np.float32)
    return mock_model

def mock_llm_pipeline(task, model):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": ""}]
    return mock_pipe

patcher_st = patch('sentence_transformers.SentenceTransformer', side_effect=mock_sentence_transformer)
patcher_pipe = patch('transformers.pipeline', side_effect=mock_llm_pipeline)
patcher_st.start()
patcher_pipe.start()

class TestDataGenerator:
    """Generates test data for various scenarios"""
    @staticmethod
    def generate_random_string(length: int = 100) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))

    @staticmethod
    def generate_multilingual_questions() -> List[Dict[str, str]]:
        return [
            {"question": "What is machine learning?", "answer": "ML is a subset of AI"},
            {"question": "¿Qué es el aprendizaje automático?", "answer": "El ML es un subconjunto de IA"},
            {"question": "Máy học là gì?", "answer": "Máy học là một phần của trí tuệ nhân tạo"},
            {"question": "機械学習とは何ですか？", "answer": "機械学習はAIの一部です"},
            {"question": "Что такое машинное обучение?", "answer": "МО является частью ИИ"},
            {"question": "ما هو التعلم الآلي؟", "answer": "التعلم الآلي جزء من الذكاء الاصطناعي"},
        ]

class TestProvenance(unittest.TestCase):
    """Test Provenance model validation"""
    def test_valid_provenance_creation(self):
        prov = Provenance(source_type="seed_faq", source_id="test123")
        self.assertEqual(prov.source_type, "seed_faq")
        self.assertEqual(prov.source_id, "test123")
        self.assertIsNotNone(prov.created_at)

    def test_empty_source_type_validation(self):
        with self.assertRaises(ValueError):
            Provenance(source_type="", source_id="test")

class TestFAQItem(unittest.TestCase):
    """Test FAQItem model validation"""
    def setUp(self):
        self.sample_prov = Provenance(source_type="test")

    def test_valid_faq_item_creation(self):
        item = FAQItem(
            id="test-id",
            question="Thủ đô của Việt Nam là gì?",
            answer="Hà Nội",
            canonical_id="canon-id",
            provenance=self.sample_prov
        )
        self.assertEqual(item.question, "Thủ đô của Việt Nam là gì?")
        self.assertEqual(item.answer, "Hà Nội")

    def test_empty_question_validation(self):
        with self.assertRaises(ValueError):
            FAQItem(id="1", question="", answer="Answer", canonical_id="1", provenance=self.sample_prov)

    def test_empty_answer_validation(self):
        with self.assertRaises(ValueError):
            FAQItem(id="1", question="Question", answer="", canonical_id="1", provenance=self.sample_prov)

class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_write_jsonl(self):
        test_data = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        test_path = os.path.join(self.temp_dir, "test.jsonl")
        write_jsonl(test_path, test_data)
        read_data = read_jsonl(test_path)
        self.assertEqual(read_data, test_data)

    def test_read_jsonl_malformed(self):
        test_path = os.path.join(self.temp_dir, "malformed.jsonl")
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        data = read_jsonl(test_path)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], {"valid": "json"})
        self.assertEqual(data[1], {"another": "valid"})

    def test_simple_chunks_normal(self):
        text = "Câu một. Câu hai. Câu ba."
        chunks = simple_chunks(text, max_chars=10)
        self.assertEqual(chunks, ["Câu một.", "Câu hai.", "Câu ba."])

    def test_simple_chunks_edge_cases(self):
        self.assertEqual(simple_chunks(""), [])
        with self.assertRaises(ValueError):
            simple_chunks("text", max_chars=0)

    def test_parse_jsonl_block(self):
        jsonl_text = '''{"question": "Q1", "answer": "A1"}
        {"question": "Q2", "answer": "A2"}
        invalid json
        {"incomplete": "missing answer"}
        {"question": "Q3", "answer": "A3"}'''
        result = parse_jsonl_block(jsonl_text)
        self.assertEqual(len(result), 3)

    def test_parse_bullets(self):
        bullet_text = '''- First item
        • Second item
        * Third item
        Regular line without bullet'''
        result = parse_bullets(bullet_text)
        expected = ["First item", "Second item", "Third item", "Regular line without bullet"]
        self.assertEqual(result, expected)

class TestFAQIndex(unittest.TestCase):
    """Test FAQ indexing and search functionality"""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_items = [
            FAQItem(id=str(uuid.uuid4()), question="Máy học là gì?", answer="Máy học là một phần của trí tuệ nhân tạo.", canonical_id="ml-1", provenance=Provenance(source_type="test")),
            FAQItem(id=str(uuid.uuid4()), question="Deep learning hoạt động như thế nào?", answer="Deep learning sử dụng mạng neural với nhiều lớp.", canonical_id="dl-1", provenance=Provenance(source_type="test")),
        ]
        self.mock_nn_fit = patch('sklearn.neighbors.NearestNeighbors.fit').start()
        self.mock_nn_kneighbors = patch('sklearn.neighbors.NearestNeighbors.kneighbors').start()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.mock_nn_fit.stop()
        self.mock_nn_kneighbors.stop()

    def test_index_build_and_search(self):
        self.mock_nn_kneighbors.return_value = (np.array([[0.05, 0.1]]), np.array([[0, 1]]))
        
        index = FAQIndex()
        index.build(self.test_items)
        self.assertEqual(len(index.items), 2)
        
        results = index.search("máy học", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0][0], FAQItem)

    def test_index_empty_items(self):
        index = FAQIndex()
        with self.assertRaises(ValueError):
            index.build([])

    def test_search_without_build(self):
        index = FAQIndex()
        with self.assertRaises(ValueError):
            index.search("test query")

    def test_index_save_load(self):
        index = FAQIndex()
        index.build(self.test_items)
        index.save(self.temp_dir)
        
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "faq_matrix.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "faq_items.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))

        loaded_index = FAQIndex.load(self.temp_dir)
        self.assertEqual(len(loaded_index.items), len(self.test_items))

class TestLLM(unittest.TestCase):
    """Test LLM class and its interactions"""
    @patch('src.llm.URag_F.FAQ.LLM.generate')
    def test_llm_generate(self, mock_generate):
        mock_generate.return_value = 'Mocked LLM response'
        llm = LLM()
        response = llm.generate("Test prompt")
        self.assertEqual(response, 'Mocked LLM response')
        mock_generate.assert_called_once_with("Test prompt")

class TestTier1FAQAgent(unittest.TestCase):
    """Test Tier1FAQAgent functionality"""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index_dir = os.path.join(self.temp_dir, "index")
        os.makedirs(self.index_dir)
        
        self.test_items = [
            FAQItem(id="1", question="Làm thế nào để đặt lại mật khẩu?", answer="Vào trang đăng nhập, nhấp 'Quên mật khẩu'.", canonical_id="pwd-reset", provenance=Provenance(source_type="test")),
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.llm.URag_F.FAQ.FAQIndex')
    def test_answer_direct_match(self, mock_index_class):
        mock_index = MagicMock()
        mock_index.search.return_value = [(self.test_items[0], 0.98)]
        mock_index_class.load.return_value = mock_index

        agent = Tier1FAQAgent(self.index_dir, threshold_faq=0.9)
        result = agent.answer("Làm thế nào để đặt lại mật khẩu?")
        
        self.assertFalse(result["escalated"])
        self.assertIn("mật khẩu", result["answer"])
        self.assertEqual(result["canonical_id"], "pwd-reset")
        self.assertAlmostEqual(result["score"], 0.98)

    @patch('src.llm.URag_F.FAQ.FAQIndex')
    def test_answer_escalation(self, mock_index_class):
        mock_index = MagicMock()
        mock_index.search.return_value = [(self.test_items[0], 0.6)]
        mock_index_class.load.return_value = mock_index

        agent = Tier1FAQAgent(self.index_dir, threshold_faq=0.8)
        result = agent.answer("unrelated query")
        
        self.assertTrue(result["escalated"])
        answer = result.get("answer")
        self.assertIsNotNone(answer)
        if answer:  # Ensure answer is not None before checking
            self.assertIn("Xin lỗi", answer)

    @patch('src.llm.URag_F.FAQ.FAQIndex')
    def test_no_results_escalation(self, mock_index_class):
        mock_index = MagicMock()
        mock_index.search.return_value = []
        mock_index_class.load.return_value = mock_index

        agent = Tier1FAQAgent(self.index_dir, threshold_faq=0.8)
        result = agent.answer("unrelated query")
        self.assertTrue(result["escalated"])

class TestBuildEnrichedFAQ(unittest.TestCase):
    """Test build_enriched_faq function"""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        self.out_dir = os.path.join(self.temp_dir, "index")
        self.seed_path = os.path.join(self.temp_dir, "seed.jsonl")
        os.makedirs(self.docs_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.llm.URag_F.FAQ.LLM')
    @patch('src.llm.URag_F.FAQ.FAQIndex')
    def test_build_pipeline(self, mock_index_class, mock_llm_class):
        with open(os.path.join(self.docs_dir, "doc1.txt"), "w", encoding="utf-8") as f:
            f.write("Câu một. Câu hai.")
        
        seed_data = [{"question": "Câu một là gì?", "answer": "Câu một."}]
        write_jsonl(self.seed_path, seed_data)

        mock_llm = Mock()
        mock_llm.generate.side_effect = ['{"question": "Câu hai là gì?", "answer": "Câu hai."}', "- Một cách hỏi khác"]
        mock_llm_class.return_value = mock_llm

        mock_index = MagicMock()
        mock_index_class.return_value = mock_index
        mock_index.save.side_effect = lambda path: Path(path).joinpath("faq_items.jsonl").touch()

        stats = build_enriched_faq(
            doc_dir=self.docs_dir,
            initial_faq_path=self.seed_path,
            out_dir=self.out_dir,
            paraphrase_n=1
        )

        self.assertEqual(stats["count_seed"], 1)
        self.assertEqual(stats["unique_canonicals"], 2)
        self.assertIn("index_dir", stats)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "faq_items.jsonl")))
        
class TestIntegrationAndEndToEnd(unittest.TestCase):
    """Integration and end-to-end tests"""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.llm.URag_F.FAQ.LLM')
    @patch('src.llm.URag_F.FAQ.FAQIndex')
    def test_full_pipeline_integration(self, mock_index_class, mock_llm_class):
        docs_dir = os.path.join(self.temp_dir, "docs")
        out_dir = os.path.join(self.temp_dir, "index")
        seed_path = os.path.join(self.temp_dir, "seed.jsonl")
        os.makedirs(docs_dir)

        with open(os.path.join(docs_dir, "guide.txt"), "w", encoding="utf-8") as f:
            f.write("Python là ngôn ngữ lập trình. Nó được sử dụng cho phát triển web và khoa học dữ liệu.")
        
        seed_data = [{"question": "Python là gì?", "answer": "Python là ngôn ngữ lập trình."}]
        write_jsonl(seed_path, seed_data)

        mock_llm = Mock()
        mock_llm.generate.side_effect = [
            '{"question": "Python được dùng làm gì?", "answer": "Dùng cho phát triển web"}',
            "- Python lập trình là gì?\n- Python hoạt động như thế nào?",
        ]
        mock_llm_class.return_value = mock_llm

        mock_index_instance = MagicMock()
        mock_index_instance.search.return_value = [
            (FAQItem(id="1", question="Python là gì?", answer="Python là ngôn ngữ lập trình.", canonical_id="1", provenance=Provenance(source_type="test")), 0.99)
        ]
        mock_index_class.load.return_value = mock_index_instance
        mock_index_class.return_value = mock_index_instance

        stats = build_enriched_faq(
            doc_dir=docs_dir,
            initial_faq_path=seed_path,
            out_dir=out_dir,
            paraphrase_n=1
        )
        self.assertGreater(stats["count_seed"], 0)
        self.assertTrue(os.path.exists(out_dir))

        agent = Tier1FAQAgent(out_dir, threshold_faq=0.7)
        result = agent.answer("Nói cho tôi về Python")
        self.assertIn("answer", result)
        self.assertFalse(result["escalated"])

class TestCLI(unittest.TestCase):
    """Test CLI functionality"""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.index_dir = os.path.join(self.temp_dir, "index")
        os.makedirs(self.index_dir, exist_ok=True)
        
        with open(os.path.join(self.index_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"embed_model_name": "mock", "dimension": 384}, f)
        write_jsonl(os.path.join(self.index_dir, "faq_items.jsonl"), [])
        np.save(os.path.join(self.index_dir, "faq_matrix.npy"), np.zeros((0, 384), dtype=np.float32))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('sys.argv', ['script.py', 'build', '--docs', 'docs', '--out', 'out'])
    @patch('src.llm.URag_F.FAQ.build_enriched_faq')
    @patch('json.dumps', return_value='{}')
    def test_cli_build(self, mock_dumps, mock_build):
        with patch('builtins.print'):
            main()
            mock_build.assert_called_once()
    
    @patch('sys.argv', ['script.py', 'chat', '--index'])
    @patch('builtins.input', side_effect=['Python là gì?', 'exit'])
    @patch('src.llm.URag_F.FAQ.Tier1FAQAgent')
    def test_cli_chat(self, mock_agent_class, mock_input):
        sys.argv.append(self.index_dir)
        mock_agent = MagicMock()
        mock_agent.answer.return_value = {"answer": "Python là ngôn ngữ lập trình", "escalated": False}
        mock_agent_class.return_value = mock_agent

        with patch('builtins.print') as mock_print:
            main()
            mock_agent.answer.assert_called_once_with("Python là gì?")
            print_calls = " ".join([str(call) for call in mock_print.call_args_list])
            self.assertIn("Python là ngôn ngữ lập trình", print_calls)

if __name__ == "__main__":
    try:
        suite = unittest.TestLoader().loadTestsFromNames([
            '__main__.TestProvenance',
            '__main__.TestFAQItem',
            '__main__.TestUtilities',
            '__main__.TestFAQIndex',
            '__main__.TestLLM',
            '__main__.TestTier1FAQAgent',
            '__main__.TestBuildEnrichedFAQ',
            '__main__.TestIntegrationAndEndToEnd',
            '__main__.TestCLI',
        ])
        
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
        
    finally:
        patcher_st.stop()
        patcher_pipe.stop()