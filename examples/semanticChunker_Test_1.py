import unittest
import os
import json

# Import semanticChunker from src
from src.llm.URag_D import semanticChunker as sc


class TestSemanticChunker(unittest.TestCase):

    def setUp(self):
        # Input file (relative path)
        self.test_file = os.path.join(
            "examples", "semanticChunker_Test_input", "semanticChunker_Test_Input_1.txt"
        )
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Test input file not found: {self.test_file}")

    # Uncomment this if you want to auto-clean JSON files after tests
    # def tearDown(self):
    #     for fname in ["semantic_chunks.json", "chunk_outputs.json"]:
    #         if os.path.exists(fname):
    #             os.remove(fname)

    def test_semantic_chunk(self):
        """Should split text into semantic chunks by paragraph."""
        text = "Một đoạn.\n\nHai đoạn."
        chunks = sc.semantic_chunk(text)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "Một đoạn.")

    def test_process_chunk_real(self):
        """Should call Gemini and return a list of quick descriptions."""
        chunk = sc.Chunk(text="Đây là đoạn thử nghiệm để sinh mô tả nhanh.")
        result = sc.process_chunk(chunk)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(all(isinstance(r, str) for r in result))
        self.assertTrue(len(result) > 0)

    def test_semantic_chunker_pipeline_real(self):
        """Should run the pipeline on sample text and return real outputs."""
        text = "Đây là một đoạn văn.\n\nĐây là đoạn văn khác."
        result = sc.semantic_chunker_pipeline(text)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(all(isinstance(r, str) for r in result))
        self.assertEqual(len(result), 2)  # one per paragraph

    def test_read_txt_file(self):
        """Should read file content as string."""
        content = sc.read_txt_file(self.test_file)
        self.assertTrue(len(content) > 0)
        self.assertIsInstance(content, str)

    def test_save_outputs_to_json(self):
        """Should save outputs to JSON and read them back."""
        outputs = ["Mô tả A", "Mô tả B"]
        sc.save_outputs_to_json(outputs, "chunk_outputs.json")
        with open("chunk_outputs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, outputs)

    def test_semantic_chunker_main_real1(self):
        """Should run the full pipeline and save semantic_chunks.json."""
        sc.semantic_chunker(self.test_file)

        self.assertTrue(os.path.exists("semantic_chunks.json"))

        with open("semantic_chunks.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertTrue(isinstance(data, list))
        self.assertTrue(all(isinstance(r, str) for r in data))

        # Check number of summaries equals number of paragraphs in input file
        input_text = sc.read_txt_file(self.test_file)
        expected_chunk_count = len(sc.semantic_chunk(input_text))
        self.assertEqual(len(data), expected_chunk_count)

    def test_semantic_chunker_main_real_input2(self):
        """Run the full pipeline on input_2.txt."""
        test_file = os.path.join("examples", "semanticChunker_Test_input", "semanticChunker_Test_Input_2.txt")
        sc.semantic_chunker(test_file)

        with open("semantic_chunks.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        input_text = sc.read_txt_file(test_file)
        expected_chunk_count = len(sc.semantic_chunk(input_text))
        self.assertEqual(len(data), expected_chunk_count)

# def test_semantic_chunker_main_real_input3(self):
#     """Run the full pipeline on input_3.txt."""
#     test_file = os.path.join("examples", "semanticChunker_Test_input", "semanticChunker_Test_Input_3.txt")
#     sc.semantic_chunker(test_file)

#     with open("semantic_chunks.json", "r", encoding="utf-8") as f:
#         data = json.load(f)

#     input_text = sc.read_txt_file(test_file)
#     expected_chunk_count = len(sc.semantic_chunk(input_text))
#     self.assertEqual(len(data), expected_chunk_count)

if __name__ == "__main__":
    unittest.main()
