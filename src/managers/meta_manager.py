# -*- coding: utf-8 -*-
"""
meta_manager.py — Central Orchestrator for RAG Ingestion

Workflow:
1) input_type='docs'
   - URagDManager: load -> chunk -> LLM augment -> embed -> index (DOC)
   - FManager: augmented_chunks -> generate roots (FAQAgent) -> enrich (paraphrase Q only) -> embed -> index (FAQ)

2) input_type='faqs'
   - load FAQs (.json / .jsonl) -> enrich (FAQAgent) -> embed -> index (FAQ)
"""

from __future__ import annotations
import os, json, time, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------- Sub-managers ----------
from src.managers.urag_d_manager import URagDManager, DManagerConfig
from src.managers.urag_f_manager import (
    FManager, FManagerConfig, IFaqGenerator, IEmbedder
)

# ---------- Core components ----------
from src.embedding.embedding_agent import EmbedderAgent, EmbConfig
from src.indexing.indexing_agent import Metric

# ---------- Load FAQs from file ----------
from src.llm.URag_D.document_loader import DocLoaderLC

# ---------- FAQ (LLM) ----------
from src.llm.URag_F.FAQ import FAQAgent, FAQPair


# ======================= FAQ Generator Adapter =======================

class FAQGeneratorAdapter(IFaqGenerator):
    """
    Adapter để FManager dùng được logic từ FAQAgent (enrich chỉ đổi question, answer giữ nguyên).
    """
    def __init__(
        self,
        api_key_env: str = "GEMINI_API_KEY",
        model_name: str = "gemini-1.5-flash",
        min_pairs: int = 4,
        enrich_pairs_per_seed: int = 4,
    ):
        api_key = os.getenv(api_key_env) or os.getenv("GOOGLE_API_KEY") or ""
        if not api_key:
            raise RuntimeError(
                "FAQGeneratorAdapter: thiếu GEMINI_API_KEY/GOOGLE_API_KEY trong môi trường."
            )
        self._agent = FAQAgent(
            api_key=api_key,
            model_name=model_name,
            min_pairs=min_pairs,
            enrich_pairs_per_seed=enrich_pairs_per_seed,
        )

    # ---- IFaqGenerator API ----

    def generate_roots(self, augmented_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Duyệt từng augmented chunk -> gọi FAQAgent.generate(text) -> gom root FAQs.
        Trả về list dicts:
          {question, answer, canonical_id, source, metadata}
        """
        roots: List[Dict[str, Any]] = []
        for ch in augmented_chunks:
            # chọn text tốt nhất để extract Q/A (ưu tiên augmented)
            text = ch.get("transformed") or ch.get("text") or ch.get("original") or ""
            if not isinstance(text, str) or not text.strip():
                continue

            pairs = self._agent.generate(text)  # List[FAQPair]
            src = (ch.get("metadata") or {}).get("source", "doc_qa_generated")
            meta = {
                "doc_id": ch.get("doc_id"),
                "chunk_id": ch.get("chunk_id"),
                **(ch.get("metadata") or {}),
            }

            for p in pairs:
                if not p.question or not p.answer:
                    continue
                roots.append({
                    "question": p.question.strip(),
                    "answer": p.answer.strip(),
                    "canonical_id": str(uuid.uuid4()),
                    "source": src,
                    "metadata": meta,
                })
        return roots

    def enrich_from_roots(self, roots: List[Dict[str, Any]], paraphrase_n: int = 5) -> List[Dict[str, Any]]:
        """
        Paraphrase câu hỏi (answer giữ nguyên). Trả về cả roots + paraphrases.
        """
        if not roots:
            return []

        # tạo danh sách FAQPair từ roots
        seed_pairs: List[FAQPair] = []
        for r in roots:
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                seed_pairs.append(FAQPair(question=q, answer=a))

        # nếu muốn override số biến thể/seed theo tham số paraphrase_n:
        if paraphrase_n and paraphrase_n != self._agent.enrich_pairs_per_seed:
            self._agent.enrich_pairs_per_seed = paraphrase_n

        new_pairs = self._agent.enrich(seed_pairs)  # chỉ câu hỏi mới, answer = seed answer
        enriched: List[Dict[str, Any]] = list(roots)  # cộng dồn

        # map lại thành dicts đúng schema FManager
        id_by_answer: Dict[str, str] = {}
        meta_by_answer: Dict[str, Dict[str, Any]] = {}
        for r in roots:
            a = (r.get("answer") or "").strip()
            if a and a not in id_by_answer:
                id_by_answer[a] = r.get("canonical_id") or str(uuid.uuid4())
                meta_by_answer[a] = r.get("metadata") or {}

        for p in new_pairs:
            a = p.answer.strip()
            enriched.append({
                "question": p.question.strip(),
                "answer": a,  # giữ nguyên
                "canonical_id": id_by_answer.get(a, str(uuid.uuid4())),
                "is_paraphrase": True,
                "source": "paraphrase_generated",
                "metadata": meta_by_answer.get(a, {}),
            })

        return enriched


# ======================= Meta Manager Config =======================

@dataclass
class MetaManagerConfig:
    # Shared
    collection_base: str = "ura_rag_meta"
    metric_type: Metric = "COSINE"
    language: str = "default"  # "vi" | "default"

    # FAQ gen
    faq_model_name: str = "gemini-1.5-flash"
    faq_min_pairs: int = 4
    faq_paraphrase_n: int = 5
    faq_api_key_env: str = "GEMINI_API_KEY"

    # children configs
    d_manager_config: DManagerConfig = field(default_factory=DManagerConfig)
    f_manager_config: FManagerConfig = field(default_factory=FManagerConfig)


# ======================= Meta Manager =======================

class MetaManager:
    """
    Orchestrator điều phối DManager & FManager theo input_type: 'docs' | 'faqs'
    """
    def __init__(self, cfg: MetaManagerConfig):
        self.cfg = cfg
        print("[MetaManager] Init...")

        # Embedder cho FManager (DManager tự dùng embedder riêng nội bộ)
        emb_cfg = EmbConfig(
            language=("vi" if self.cfg.language.lower() == "vi" else "default"),
            metric=self.cfg.metric_type,
        )
        self.embedder: IEmbedder = EmbedderAgent(emb_cfg)
        print(f"[MetaManager] Embedder ready (language={self.cfg.language})")

        # FAQ generator adapter (dùng FAQAgent)
        self.faq_generator: IFaqGenerator = FAQGeneratorAdapter(
            api_key_env=self.cfg.faq_api_key_env,
            model_name=self.cfg.faq_model_name,
            min_pairs=self.cfg.faq_min_pairs,
            enrich_pairs_per_seed=self.cfg.faq_paraphrase_n,
        )

        # ---------- Child managers ----------
        # Khớp với URagDManager mới: dùng 'milvus_collection_base', 'metric', 'lang'
        self.cfg.d_manager_config.milvus_collection_base = self.cfg.collection_base
        self.cfg.d_manager_config.metric = self.cfg.metric_type
        self.cfg.d_manager_config.lang = ("vi" if self.cfg.language.lower() == "vi" else "default")
        self.d_manager = URagDManager(self.cfg.d_manager_config)  # uses run_pipeline(), get_augmented_chunks()

        self.f_manager = FManager(self.faq_generator, self.embedder, self.cfg.f_manager_config)

        # Loader: đọc JSON/JSONL FAQs khi input_type='faqs'
        self.doc_loader = DocLoaderLC()

    # ------------- public API -------------

    def run(self, input_path: str, input_type: str) -> Dict[str, Any]:
        """
        input_type: 'docs' | 'faqs'
        - 'docs': chạy full document->FAQ pipeline
        - 'faqs': enrich & index trực tiếp từ file FAQ (.json / .jsonl)
        """
        t0 = time.time()
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_type == "docs":
            # Step 1: Documents -> chunks -> LLM augment -> embed/index (DOC)
            print("[MetaManager] step-1: DManager.run_pipeline(...)")
            d_resp = self.d_manager.run_pipeline(
                root_dir=input_path,
                collection_base=self.cfg.collection_base
            )  # returns indexing response for DOCs  :contentReference[oaicite:2]{index=2}

            augmented = self.d_manager.get_augmented_chunks()  # List[AugmentedItem] (TypedDict)

            if not augmented:
                return {
                    "status": "ok",
                    "input_type": "docs",
                    "total_sec": round(time.time() - t0, 2),
                    "d_manager_summary": d_resp,
                    "f_manager_summary": "skipped_no_augmented"
                }

            # 🔧 Chuyển TypedDict -> Dict[str, Any] để Pylance im lặng
            augmented_dicts: List[Dict[str, Any]] = [dict(x) for x in augmented]

            # Step 2: Augmented -> FManager
            f_res = self.f_manager.run_from_augmented(
                augmented=augmented_dicts,
                collection_base=self.cfg.collection_base,
                paraphrase_n=self.cfg.faq_paraphrase_n,
                metric=self.cfg.metric_type,
            )

            return {
                "status": "success",
                "input_type": "docs",
                "total_sec": round(time.time() - t0, 2),
                "d_manager_summary": d_resp,
                "f_manager_summary": f_res.get("summary"),
                "indexing_responses": {
                    "docs": d_resp,
                    "faqs": f_res.get("resp", {}),
                }
            }

        elif input_type == "faqs":
            # Step 1: load FAQ file
            print("[MetaManager] load root FAQs file")
            roots = self.doc_loader.load_faq_json(input_path)
            if not roots:
                return {"status": "error", "message": "No FAQs loaded from file."}

            # Step 2: enrich + embed/index
            print("[MetaManager] FManager.run_from_roots(...)")
            f_res = self.f_manager.run_from_roots(
                roots=roots,
                collection_base=self.cfg.collection_base,
                paraphrase_n=self.cfg.faq_paraphrase_n,
                metric=self.cfg.metric_type,
            )  # :contentReference[oaicite:5]{index=5}
            return {
                "status": "success",
                "input_type": "faqs",
                "total_sec": round(time.time() - t0, 2),
                "f_manager_summary": f_res.get("summary"),
                "indexing_responses": {"faqs": f_res.get("resp", {})}
            }

        else:
            raise ValueError("input_type must be one of: 'docs', 'faqs'")


# ======================= Quick demo =======================
if __name__ == "__main__":
    # tạo thử data đơn giản
    os.makedirs("demo_docs", exist_ok=True)
    with open("demo_docs/a.txt","w",encoding="utf-8") as f:
        f.write("Điểm sàn, mốc thời gian đăng ký xét tuyển, và chính sách học bổng nếu có sẽ được công bố chính thức.")

    os.makedirs("demo_faqs", exist_ok=True)
    with open("demo_faqs/root.jsonl","w",encoding="utf-8") as f:
        f.write(json.dumps({"question":"Điểm sàn tuyển sinh là bao nhiêu?","answer":"Nhà trường công bố theo từng năm."}, ensure_ascii=False) + "\n")

    cfg = MetaManagerConfig(
        collection_base="rag_meta_demo",
        faq_model_name="gemini-1.5-flash",
        faq_min_pairs=3,
        faq_paraphrase_n=2,
        language="vi",
    )
    mm = MetaManager(cfg)

    print("\n>>> DOCS PIPELINE")
    print(json.dumps(mm.run("demo_docs","docs"), ensure_ascii=False, indent=2))

    print("\n>>> FAQS PIPELINE")
    print(json.dumps(mm.run("demo_faqs/root.jsonl","faqs"), ensure_ascii=False, indent=2))
