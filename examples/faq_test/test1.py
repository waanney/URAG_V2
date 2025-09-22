import os
import shutil
import tempfile
import unittest
from pathlib import Path
import sys
import importlib.util
from unittest.mock import patch, MagicMock

# Add the project root and src directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import using the correct module path
try:
    # Try importing directly from the module
    from llm.URag_F.FAQ import (
        Provenance, FAQItem, read_jsonl, write_jsonl, simple_chunks,
        FAQIndex, Tier1FAQAgent, build_enriched_faq
    )
except ImportError as e:
    print(f"Import error: {e}")
    # If that fails, try a more direct approach with proper error handling
    faq_path = project_root / "src" / "llm" / "URag_F" / "FAQ.py"
    
    if not faq_path.exists():
        raise ImportError(f"FAQ.py not found at {faq_path}")
    
    # Load the module directly from file
    spec = importlib.util.spec_from_file_location("FAQ", faq_path)
    if spec is None:
        raise ImportError(f"Could not create spec for {faq_path}")
    
    FAQ = importlib.util.module_from_spec(spec)
    
    # Execute the module
    try:
        spec.loader.exec_module(FAQ)  # type: ignore
    except Exception as e:
        raise ImportError(f"Failed to execute module: {e}")
    
    # Manually import the required classes/functions
    try:
        Provenance = FAQ.Provenance
        FAQItem = FAQ.FAQItem
        read_jsonl = FAQ.read_jsonl
        write_jsonl = FAQ.write_jsonl
        simple_chunks = FAQ.simple_chunks
        FAQIndex = FAQ.FAQIndex
        Tier1FAQAgent = FAQ.Tier1FAQAgent
        build_enriched_faq = FAQ.build_enriched_faq
    except AttributeError as e:
        raise ImportError(f"Missing expected attribute in FAQ module: {e}")

class TestFAQ(unittest.TestCase):
    def setUp(self):
        # Tạo thư mục tạm để test
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_jsonl_utils(self):
        path = os.path.join(self.tmpdir, "data.jsonl")
        rows = [{"question": "Q1", "answer": "A1"}]
        write_jsonl(path, rows)
        loaded = read_jsonl(path)
        self.assertEqual(rows, loaded)

    def test_simple_chunks(self):
        text = "Câu 1. Câu 2. Câu 3."
        chunks = simple_chunks(text, max_chars=10)
        self.assertTrue(all(isinstance(c, str) for c in chunks))
        self.assertGreaterEqual(len(chunks), 2)

    def test_faq_index_build_search_save_load(self):
        # Tạo FAQItem giả
        prov = Provenance(source_type="seed_faq")
        item = FAQItem(
            id="1", question="Thủ đô Việt Nam là gì?",
            answer="Hà Nội", canonical_id="c1", provenance=prov
        )
        idx = FAQIndex()
        idx.build([item])
        
        # Use top_k=1 since we only have one item
        results = idx.search("Thủ đô của VN?", top_k=1)
        self.assertEqual(results[0][0].answer, "Hà Nội")

        # Test save/load
        idx.save(self.tmpdir)
        idx2 = FAQIndex.load(self.tmpdir)
        results2 = idx2.search("VN thủ đô?", top_k=1)
        self.assertEqual(results2[0][0].answer, "Hà Nội")

    @patch('llm.URag_F.FAQ.LLM.generate')
    @patch('llm.URag_F.FAQ.parse_jsonl_block')
    @patch('llm.URag_F.FAQ.parse_bullets')
    def test_build_enriched_faq(self, mock_parse_bullets, mock_parse_jsonl_block, mock_generate):
        # Mock the LLM responses
        mock_generate.side_effect = [
            '{"question": "Thủ đô Việt Nam là gì?", "answer": "Hà Nội"}',
            "- Thủ đô của Việt Nam\n- Việt Nam thủ đô ở đâu?"
        ]
        mock_parse_jsonl_block.return_value = [{"question": "Thủ đô Việt Nam là gì?", "answer": "Hà Nội"}]
        mock_parse_bullets.return_value = ["Thủ đô của Việt Nam", "Việt Nam thủ đô ở đâu?"]
        
        # Tạo document giả
        doc_path = Path(self.tmpdir) / "doc.txt"
        doc_path.write_text("Việt Nam có thủ đô là Hà Nội.", encoding="utf-8")
        out_dir = Path(self.tmpdir) / "out"
        stats = build_enriched_faq(
            doc_dir=self.tmpdir,
            initial_faq_path=None,
            out_dir=str(out_dir),
            paraphrase_n=1,
            llm_model="sshleifer/tiny-gpt2"  # model nhỏ để test
        )
        self.assertIn("index_dir", stats)
        self.assertTrue((out_dir / "faq_matrix.npy").exists())

    def test_tier1_agent(self):
        # Index nhỏ
        prov = Provenance(source_type="seed_faq")
        item = FAQItem(
            id="1", question="Ai là chủ tịch Hồ Chí Minh?",
            answer="Lãnh tụ VN", canonical_id="c1", provenance=prov
        )
        idx = FAQIndex()
        idx.build([item])
        idx.save(self.tmpdir)

        # Agent with top_k=1 since we only have one item
        agent = Tier1FAQAgent(index_dir=self.tmpdir, threshold_faq=0.1, top_k=1)
        ans = agent.answer("Chủ tịch HCM là ai?")
        self.assertIn("Lãnh tụ VN", ans["answer"])

        # Với câu lạ => escalated
        agent2 = Tier1FAQAgent(index_dir=self.tmpdir, threshold_faq=0.99, top_k=1)
        ans2 = agent2.answer("Hỏi lạ hoắc")
        self.assertTrue(ans2["escalated"])

if __name__ == "__main__":
    unittest.main()