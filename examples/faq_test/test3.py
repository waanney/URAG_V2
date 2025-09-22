import os
import sys
import shutil
import tempfile
import unittest
import builtins

# thêm URAG_V2 vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

from src.llm.URag_F import FAQ

Provenance = FAQ.Provenance
FAQItem = FAQ.FAQItem
FAQIndex = FAQ.FAQIndex
Tier1FAQAgent = FAQ.Tier1FAQAgent
main = FAQ.main
read_jsonl = FAQ.read_jsonl
write_jsonl = FAQ.write_jsonl
simple_chunks = FAQ.simple_chunks


class TestFAQStrict(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.prov = Provenance(source_type="seed_faq", source_id="strict")
        self.item1 = FAQItem(id="1", question="Việt Nam có bao nhiêu tỉnh?", answer="63", canonical_id="c1", provenance=self.prov)
        self.item2 = FAQItem(id="2", question="Thủ đô của Việt Nam là gì?", answer="Hà Nội", canonical_id="c2", provenance=self.prov)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ===== Validation =====
    def test_invalid_provenance(self):
        with self.assertRaises(ValueError):
            Provenance(source_type="", source_id="bad")

    def test_invalid_faqitem(self):
        with self.assertRaises(ValueError):
            FAQItem(id="x", question="", answer="A", canonical_id="c", provenance=self.prov)

    # ===== Index =====
    def test_index_save_load_consistency(self):
        idx = FAQIndex()
        idx.build([self.item1, self.item2])
        save_path = os.path.join(self.tmpdir, "faq_index")
        idx.save(save_path)

        loaded = FAQIndex.load(save_path)
        res1 = idx.search("Thủ đô của VN?", top_k=1)
        res2 = loaded.search("Thủ đô của VN?", top_k=1)
        self.assertEqual(res1[0][0], res2[0][0])
        self.assertAlmostEqual(float(res1[0][1]), float(res2[0][1]), places=5)

    def test_index_empty_build_raises(self):
        idx = FAQIndex()
        with self.assertRaises(ValueError):
            idx.build([])

    # ===== Agent =====
    def test_agent_threshold_escalation_with_low_similarity(self):
        idx = FAQIndex()
        idx.build([self.item1])
        idx.save(self.tmpdir)

        agent = Tier1FAQAgent(self.tmpdir, threshold_faq=0.95)
        ans = agent.answer("Câu hỏi hoàn toàn khác")  # similarity thấp
        self.assertTrue(ans["escalated"])

    def test_agent_with_multiple_items(self):
        items = [FAQItem(id=str(i), question=f"Hỏi {i}", answer=f"Đáp {i}", canonical_id=f"c{i}", provenance=self.prov)
                 for i in range(20)]
        idx = FAQIndex()
        idx.build(items)
        idx.save(self.tmpdir)

        agent = Tier1FAQAgent(self.tmpdir, threshold_faq=0.2, top_k=5)
        ans = agent.answer("Hỏi 10")
        self.assertFalse(ans["escalated"])
        self.assertEqual(ans["answer"], "Đáp 10")

    # ===== Utils =====
    def test_jsonl_roundtrip(self):
        path = os.path.join(self.tmpdir, "data.jsonl")
        rows = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(3)]
        write_jsonl(path, rows)
        loaded = read_jsonl(path)
        self.assertEqual(rows, loaded)

    def test_simple_chunks_limit(self):
        text = " ".join([f"Câu {i}." for i in range(50)])
        chunks = simple_chunks(text, max_chars=30)
        self.assertTrue(all(len(c) <= 30 for c in chunks))
        self.assertGreater(len(chunks), 5)

    # ===== CLI =====
    def test_cli_chat_with_real_index(self):
        idx = FAQIndex()
        idx.build([self.item1, self.item2])
        idx.save(os.path.join(self.tmpdir, "index"))

        sys.argv = ["script.py", "chat", "--index", os.path.join(self.tmpdir, "index")]
        inputs = ["Thủ đô VN?", "exit"]
        outputs = []

        def fake_input(prompt=""):
            return inputs.pop(0)

        def fake_print(*args, **kwargs):
            outputs.append(" ".join(str(a) for a in args))

        real_input, real_print = builtins.input, builtins.print
        builtins.input, builtins.print = fake_input, fake_print

        try:
            main()
        finally:
            builtins.input, builtins.print = real_input, real_print

        out_str = " ".join(outputs).lower()
        self.assertIn("faq agent ready", out_str)


if __name__ == "__main__":
    unittest.main()
