import os
import shutil
import json
import sys
import textwrap

# --- NEW: Import dotenv to load environment variables from a .env file ---
from dotenv import load_dotenv

# --- Import the functions and classes from your main script ---
# This assumes 'faq_generator_gemini.py' is in the same directory.
from src.llm.URag_F.FAQ import build_enriched_faq

# ======================= Test Configuration =======================
TEST_DATA_DIR = "temp_faq_test_data"
DOCS_DIR = os.path.join(TEST_DATA_DIR, "input_docs")
SEED_FILE = os.path.join(TEST_DATA_DIR, "seed_faqs.jsonl")
OUTPUT_FILE = os.path.join(TEST_DATA_DIR, "output_faqs.jsonl")

# ======================= Helper Functions =======================

def setup_test_environment():
    """Creates a temporary, isolated environment for the test."""
    print("--- 1. Setting up test environment ---")
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)

    os.makedirs(DOCS_DIR, exist_ok=True)

    doc_content = textwrap.dedent("""
        The Kuiper Belt is a circumstellar disc in the outer Solar System, extending from the orbit of Neptune at 30 astronomical units (AU) to approximately 50 AU from the Sun.
        It is similar to the asteroid belt, but is far larger—20 times as wide and 20 to 200 times as massive.
        Like the asteroid belt, it consists mainly of small bodies or remnants from when the Solar System formed.
    """)
    with open(os.path.join(DOCS_DIR, "space_doc.txt"), "w", encoding="utf-8") as f:
        f.write(doc_content)
        
    with open(SEED_FILE, "w", encoding="utf-8") as f:
        faq1 = {"id": "seed_001", "question": "What is the warranty policy?", "answer": "All products come with a one-year limited warranty."}
        f.write(json.dumps(faq1) + "\n")

    print(f"✅ Created temporary data in: {TEST_DATA_DIR}")

def cleanup_test_environment():
    """Removes the temporary test environment."""
    print("\n--- 4. Cleaning up test environment ---")
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
        print(f"🗑️ Removed temporary directory: {TEST_DATA_DIR}")

# ======================= Main Test Logic =======================

def main():
    """Main function to run the end-to-end data generation test."""
    print("=============================================")
    print(">>> RUNNING END-TO-END FAQ GENERATOR TEST <<<")
    print("=============================================")

    # --- NEW: Load environment variables from .env file ---
    load_dotenv()

    if "GOOGLE_API_KEY" not in os.environ or not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ ERROR: GOOGLE_API_KEY not found.")
        print("Please create a .env file in the root directory with the line: GOOGLE_API_KEY='your_api_key_here'")
        return

    try:
        # --- Setup ---
        setup_test_environment()
        
        # --- Execute the Build Pipeline ---
        print("\n--- 2. Running the `build_enriched_faq` pipeline ---")
        stats = build_enriched_faq(
            doc_dir=DOCS_DIR,
            initial_faq_path=SEED_FILE,
            output_file=OUTPUT_FILE,
            paraphrase_n=2 # Keep low for faster testing
        )
        print("\n--- Build Pipeline Finished ---")
        print("Build stats:", json.dumps(stats, indent=2))

        # --- Verification Step 1: Check build statistics ---
        print("\n--- 3. Verifying build results ---")
        assert stats["count_seed"] == 1, "Should have loaded 1 seed FAQ."
        assert stats["count_from_docs"] > 0, "Should have generated at least one FAQ from the document."
        assert stats["count_total_enriched"] > stats["count_seed"], "Total enriched count should be higher than seed count."
        assert stats["unique_canonicals"] > 0, "Should have at least one canonical entry."
        print("✅ Build statistics are valid.")

        # --- Verification Step 2: Check for output files ---
        # The main output file is the most important one
        assert os.path.exists(OUTPUT_FILE), f"Main output file is missing: {OUTPUT_FILE}"
        
        # Check that the canonical reference file was also created
        canonical_file = os.path.join(os.path.dirname(OUTPUT_FILE), "output_faqs_canonical.jsonl")
        assert os.path.exists(canonical_file), f"Canonical reference file is missing: {canonical_file}"
        print(f"✅ All expected output files were created.")
        
        # Optional: Check if the output file is not empty and contains valid JSONL
        with open(OUTPUT_FILE, "r") as f:
            lines = f.readlines()
            assert len(lines) > 0, "Output file should not be empty."
            json.loads(lines[0]) # Test if the first line is valid JSON
        print("✅ Output file content is valid.")

        print("\n🎉🎉🎉 END-TO-END TEST SUCCEEDED! 🎉🎉🎉")

    except Exception as e:
        print(f"\n❌❌❌ TEST FAILED: {e} ❌❌❌")
        import traceback
        traceback.print_exc()
        
    finally:
        # --- Cleanup ---
        cleanup_test_environment()

# ======================= Script Entry Point =======================

if __name__ == "__main__":
    main()
