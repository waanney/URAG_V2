# -*- coding: utf-8 -*-
"""
MetaManager: The Central Orchestrator for RAG Ingestion
=========================================================

This manager coordinates the document and FAQ pipelines by directing
traffic to the appropriate sub-managers (DManager and FManager).

Workflow:
1.  If the input is a directory of **documents**:
    - The `URagDManager` is invoked.
    - DManager loads, chunks, and creates augmented text chunks using an LLM.
    - These augmented chunks are embedded and indexed into the 'doc' database.
    - The augmented chunks are then passed to the `FManager`.
    - FManager uses the chunks to generate, enrich, embed, and index new FAQs
      into the 'faq' database.

2.  If the input is a file of **FAQs**:
    - The `FManager` is invoked directly.
    - FManager loads the root FAQs, enriches them (paraphrasing), embeds them,
      and indexes them into the 'faq' database.
"""
from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Import Sub-Managers and their Configurations ---
from src.managers.urag_d_manager import URagDManager, DManagerConfig
from src.managers.urag_f_manager import FManager, FManagerConfig, IFaqGenerator, IEmbedder
from src.llm.URag_D.document_loader import DocLoaderLC

# --- Import Core Components ---
from src.embedding.embedding_agent import EmbedderAgent, EmbConfig
from src.indexing.indexing_agent import Metric

# --- Import Helpers from existing files for the Adapter ---
# These components are used to create a compatible FAQ generator
from src.llm.URag_F.FAQ import LLM, QA_EXTRACT_PROMPT, PARAPHRASE_PROMPT, parse_jsonl_block, parse_bullets
import uuid

# ======================= Adapter for FAQ Generation =======================

class FAQGeneratorAdapter(IFaqGenerator):
    """
    An adapter that uses the logic from `FAQ.py` to satisfy the
    `IFaqGenerator` protocol required by `FManager`.
    """
    def __init__(self, llm_model: str = "gpt2"):
        # Initialize the LLM helper from FAQ.py
        self.llm = LLM(model=llm_model)
        print(f"[FAQGeneratorAdapter] Initialized with LLM model: {llm_model}")

    def generate_roots(self, augmented_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates root FAQs from text chunks by extracting question-answer pairs.
        """
        print(f"[FAQGeneratorAdapter] Generating root FAQs from {len(augmented_chunks)} chunks...")
        root_faqs: List[Dict[str, Any]] = []
        for chunk in augmented_chunks:
            text = chunk.get("text", "")
            if not text:
                continue
            
            # Use the prompt and logic from FAQ.py
            prompt = QA_EXTRACT_PROMPT.format(chunk=text)
            generated_text = self.llm.generate(prompt)
            qas = parse_jsonl_block(generated_text)

            for qa in qas:
                cid = str(uuid.uuid4())
                root_faqs.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "canonical_id": cid,
                    "source": chunk.get("metadata", {}).get("source", "doc_qa_generated"),
                    "metadata": {"doc_id": chunk.get("doc_id"), "chunk_id": chunk.get("chunk_id")}
                })
        print(f"[FAQGeneratorAdapter] Generated {len(root_faqs)} root FAQs.")
        return root_faqs

    def enrich_from_roots(self, roots: List[Dict[str, Any]], paraphrase_n: int = 5) -> List[Dict[str, Any]]:
        """
        Enriches root FAQs by generating paraphrased questions.
        The returned list includes both the original roots and the new paraphrases.
        """
        print(f"[FAQGeneratorAdapter] Enriching {len(roots)} root FAQs with {paraphrase_n} paraphrases each...")
        enriched_faqs: List[Dict[str, Any]] = list(roots) # Start with the originals

        for root in roots:
            question = root.get("question")
            if not question:
                continue
            
            # Use the paraphrasing prompt and logic from FAQ.py
            prompt = PARAPHRASE_PROMPT.format(q=question, n=paraphrase_n)
            generated_text = self.llm.generate(prompt, temperature=0.9)
            variants = parse_bullets(generated_text)

            for vq in variants[:paraphrase_n]:
                if not vq or len(vq) < 3:
                    continue
                
                enriched_faqs.append({
                    "question": vq,
                    "answer": root["answer"],
                    "canonical_id": root.get("canonical_id"),
                    "is_paraphrase": True,
                    "source": "paraphrase_generated",
                    "metadata": root.get("metadata", {})
                })
        print(f"[FAQGeneratorAdapter] Total enriched FAQs (roots + paraphrases): {len(enriched_faqs)}.")
        return enriched_faqs

# ======================= Meta Manager Configuration =======================

@dataclass
class MetaManagerConfig:
    """Consolidated configuration for the entire RAG ingestion pipeline."""
    # Shared settings
    collection_base: str = "ura_rag_meta"
    metric_type: Metric = "COSINE"
    language: str = "default"  # "vi" or "default"
    
    # FAQ Generation settings
    faq_gen_llm_model: str = "gpt2"
    paraphrase_n: int = 5
    
    # Child manager configurations
    d_manager_config: DManagerConfig = field(default_factory=DManagerConfig)
    f_manager_config: FManagerConfig = field(default_factory=FManagerConfig)

# ======================= Meta Manager =======================

class MetaManager:
    """
    The main orchestrator that manages the DManager and FManager to process
    documents and FAQs according to the specified workflow.
    """
    def __init__(self, cfg: MetaManagerConfig):
        self.cfg = cfg
        print("[MetaManager] Initializing...")

        # 1. Instantiate the Embedder Agent (to be used by FManager)
        # Note: DManager currently creates its own embedder internally.
        # This could be refactored for a single shared instance.
        emb_cfg = EmbConfig(
            language=("vi" if self.cfg.language.lower() == "vi" else "default"),
            metric=self.cfg.metric_type,
        )
        self.embedder: IEmbedder = EmbedderAgent(emb_cfg)
        print(f"[MetaManager] EmbedderAgent initialized for language: {self.cfg.language}")

        # 2. Instantiate the FAQ Generator Adapter
        self.faq_generator: IFaqGenerator = FAQGeneratorAdapter(llm_model=cfg.faq_gen_llm_model)

        # 3. Configure and instantiate child managers
        self.cfg.d_manager_config.collection_base = self.cfg.collection_base
        self.cfg.d_manager_config.metric_type = self.cfg.metric_type
        self.cfg.d_manager_config.language = self.cfg.language
        self.d_manager = URagDManager(self.cfg.d_manager_config)
        print("[MetaManager] URagDManager initialized.")

        self.f_manager = FManager(self.faq_generator, self.embedder, self.cfg.f_manager_config)
        print("[MetaManager] FManager initialized.")
        
        # Helper for loading FAQ files
        self.doc_loader = DocLoaderLC()

    def run(self, input_path: str, input_type: str) -> Dict[str, Any]:
        """
        Executes the main ingestion pipeline.

        Args:
            input_path (str): Path to the input data (directory for docs, file for FAQs).
            input_type (str): Type of input, either "docs" or "faqs".

        Returns:
            A dictionary containing the results of the operations.
        """
        start_time = time.time()
        print(f"\n[MetaManager] Starting run for input_type='{input_type}' at path='{input_path}'")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_type == "docs":
            # --- Full Document-to-FAQ Pipeline ---
            # 1. Process documents with DManager
            print("[MetaManager] --- Step 1: Processing documents with DManager ---")
            d_manager_result = self.d_manager.run(docs_root_dir=input_path, return_augmented=True)
            augmented_chunks = d_manager_result.get("augmented_chunks", [])
            
            if not augmented_chunks:
                print("[MetaManager] DManager produced no augmented chunks. Halting pipeline.")
                return {"d_manager_summary": d_manager_result.get("summary"), "f_manager_summary": "skipped"}

            # 2. Pass augmented chunks to FManager
            print("\n[MetaManager] --- Step 2: Generating FAQs from documents with FManager ---")
            f_manager_result = self.f_manager.run_from_augmented(
                augmented=augmented_chunks,
                collection_base=self.cfg.collection_base,
                paraphrase_n=self.cfg.paraphrase_n,
                metric=self.cfg.metric_type,
            )
            
            total_time = time.time() - start_time
            return {
                "status": "success",
                "input_type": "docs",
                "total_duration_sec": round(total_time, 2),
                "d_manager_summary": d_manager_result.get("summary"),
                "f_manager_summary": f_manager_result.get("summary"),
                "indexing_responses": {
                    "docs": d_manager_result.get("indexing_status"),
                    "faqs": f_manager_result.get("resp", {}).get("status")
                }
            }

        elif input_type == "faqs":
            # --- FAQ-Only Pipeline ---
            print("[MetaManager] --- Step 1: Loading root FAQs ---")
            root_faqs = self.doc_loader.load_faq_json(input_path)
            
            if not root_faqs:
                print("[MetaManager] No valid root FAQs found in file. Halting pipeline.")
                return {"status": "error", "message": "No FAQs loaded from file."}
                
            print(f"[MetaManager] Loaded {len(root_faqs)} root FAQs.")
            
            print("\n[MetaManager] --- Step 2: Enriching and indexing FAQs with FManager ---")
            f_manager_result = self.f_manager.run_from_roots(
                roots=root_faqs,
                collection_base=self.cfg.collection_base,
                paraphrase_n=self.cfg.paraphrase_n,
                metric=self.cfg.metric_type,
            )

            total_time = time.time() - start_time
            return {
                "status": "success",
                "input_type": "faqs",
                "total_duration_sec": round(total_time, 2),
                "f_manager_summary": f_manager_result.get("summary"),
                "indexing_responses": {
                    "faqs": f_manager_result.get("resp", {}).get("status")
                }
            }
        else:
            raise ValueError(f"Invalid input_type: '{input_type}'. Must be 'docs' or 'faqs'.")

# ======================= Example Usage =======================

if __name__ == '__main__':
    # --- Create dummy data for demonstration ---
    print("--- Setting up dummy data for demonstration ---")
    
    # 1. Dummy documents
    os.makedirs("demo_docs", exist_ok=True)
    with open("demo_docs/solar_system.txt", "w", encoding="utf-8") as f:
        f.write("The solar system consists of the Sun and the celestial bodies that orbit it. This includes planets, moons, and asteroids. Scientists study these objects to understand cosmic formation.")
    with open("demo_docs/oceans.txt", "w", encoding="utf-8") as f:
        f.write("Earth's oceans are vast bodies of saltwater. They regulate the planet's climate and support a diverse range of marine life. The deepest point is the Mariana Trench.")

    # 2. Dummy FAQ file (JSONL format)
    os.makedirs("demo_faqs", exist_ok=True)
    with open("demo_faqs/tech_faqs.jsonl", "w", encoding="utf-8") as f:
        faq1 = {"question": "What is a CPU?", "answer": "The Central Processing Unit (CPU) is the primary component of a computer that executes instructions."}
        faq2 = {"question": "What is RAM?", "answer": "Random Access Memory (RAM) is a form of computer memory that can be read and changed in any order, typically used to store working data."}
        f.write(json.dumps(faq1) + "\n")
        f.write(json.dumps(faq2) + "\n")
    print("Dummy data created in 'demo_docs/' and 'demo_faqs/' directories.\n")

    # --- Initialize and Run the MetaManager ---
    
    # Configure the manager
    # NOTE: Using 'gpt2' for the LLM part as it's small and accessible,
    # but a better model like 'distilgpt2' or a fine-tuned one is recommended for real tasks.
    config = MetaManagerConfig(
        collection_base="rag_demo_collection",
        faq_gen_llm_model="gpt2", # A small, locally runnable model for demo purposes
        paraphrase_n=2 # Reduce for faster demo
    )

    # Create the manager instance
    manager = MetaManager(config)

    # --- Run Pipeline 1: Documents ---
    try:
        print("=============================================")
        print(">>> EXECUTING DOCUMENT-TO-FAQ PIPELINE <<<")
        print("=============================================")
        doc_result = manager.run(input_path="demo_docs", input_type="docs")
        print("\n--- DOCUMENT PIPELINE FINAL RESULT ---")
        print(json.dumps(doc_result, indent=2))
        print("--- END OF DOCUMENT PIPELINE ---")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during the document pipeline: {e}")
        import traceback
        traceback.print_exc()


    # --- Run Pipeline 2: FAQs ---
    try:
        print("\n\n=============================================")
        print(">>> EXECUTING FAQ-ONLY PIPELINE <<<")
        print("=============================================")
        faq_result = manager.run(input_path="demo_faqs/tech_faqs.jsonl", input_type="faqs")
        print("\n--- FAQ PIPELINE FINAL RESULT ---")
        print(json.dumps(faq_result, indent=2))
        print("--- END OF FAQ PIPELINE ---")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during the FAQ pipeline: {e}")
        import traceback
        traceback.print_exc()