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


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _sec(ms: float) -> str:
    return f"{ms:.2f}s"

class _StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = time.time()
        print(f"[{_ts()}] ▶ START {self.name}")
    def end(self, extra: str = ""):
        dt = time.time() - self.t0
        print(f"[{_ts()}] ◀ END   {self.name} in {_sec(dt)} {extra}".rstrip())


# ======================= FAQ Generator Adapter =======================

class FAQGeneratorAdapter(IFaqGenerator):
    """
    Adapter để FManager dùng được logic từ FAQAgent (enrich chỉ đổi question, answer giữ nguyên).
    Dùng KERNEL (trong FAQAgent) để lấy LLM; không cấu hình key/model tại đây.
    """
    def __init__(
        self,
        min_pairs: int = 2,
        enrich_pairs_per_seed: int = 2,
    ):
        # Không truyền api_key/model_name -> FAQAgent sẽ gọi KERNEL.get_active_model()
        self._agent = FAQAgent(
            api_key=None,
            model_name=None,
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
            headline = (ch.get("metadata") or {}).get("headline", "")
            content = ch.get("original") or ch.get("text") or ch.get("transformed") or ""  
            text = f"[Tiêu đề tóm tắt: {headline}]\n\n{content}" if headline else content
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
        if paraphrase_n and paraphrase_n != self._agent.enriched_pairs_per_seed:
            self._agent.enriched_pairs_per_seed = paraphrase_n

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

    # FAQ gen (model/key do KERNEL quyết định; giữ các tham số nội bộ cho FAQAgent)
    faq_min_pairs: int = 2
    faq_paraphrase_n: int = 2

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
            model_name=self.cfg.d_manager_config.emb_model_name,         
            vi_model_name=self.cfg.d_manager_config.emb_vi_model_name,
            language="default",
            metric=self.cfg.metric_type,
        )
        self.embedder: IEmbedder = EmbedderAgent(emb_cfg)
        print(f"[MetaManager] Embedder ready (language={self.cfg.language})")

        # FAQ generator adapter (dùng FAQAgent qua KERNEL)
        self.faq_generator: IFaqGenerator = FAQGeneratorAdapter(
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
            )  # returns indexing response for DOCs

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
            print(f"[MetaManager] augmented_chunks = {len(augmented)} for base={self.cfg.collection_base}")  # DEBUG
            # Step 2: Augmented -> FManager
            print(f"[MetaManager] call FManager.run_from_augmented(base={self.cfg.collection_base})")  # DEBUG
            f_res = self.f_manager.run_from_augmented(
                augmented=augmented_dicts,
                collection_base=self.cfg.collection_base,
                paraphrase_n=self.cfg.faq_paraphrase_n,
                metric=self.cfg.metric_type,
            )
            print(f"[MetaManager] FManager summary = {f_res.get('summary')}, resp keys = {list((f_res.get('resp') or {}).keys())}")  # DEBUG
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
            )
            return {
                "status": "success",
                "input_type": "faqs",
                "total_sec": round(time.time() - t0, 2),
                "f_manager_summary": f_res.get("summary"),
                "indexing_responses": {"faqs": f_res.get("resp", {})}
            }

        else:
            raise ValueError("input_type must be one of: 'docs', 'faqs'")

    # --- API 1: chạy từ tài liệu đã chuẩn hoá ---
    def run_from_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        documents: List[{'doc_id': str, 'text': str, 'metadata': dict}]
        Điều phối y hệt 'docs' flow nhưng bỏ bước đọc thư mục.
        - Nếu URagDManager có run_pipeline_from_records(...) -> dùng trực tiếp.
        - Ngược lại: fallback ghi tạm .txt rồi gọi run(..., 'docs').
        """
        t0 = time.time()
        # Ưu tiên API trực tiếp nếu d_manager có
        if hasattr(self.d_manager, "run_pipeline_from_records"):
            d_resp = self.d_manager.run_pipeline_from_records(
                documents=documents,
                collection_base=self.cfg.collection_base,
            )
            augmented = self.d_manager.get_augmented_chunks() or []
        else:
            # Fallback: ghi ra thư mục tạm -> dùng run(..., 'docs')
            import tempfile, shutil
            from pathlib import Path
            tmpdir = Path(tempfile.mkdtemp(prefix="mm_docs_"))
            try:
                for d in documents:
                    did = d.get("doc_id") or str(uuid.uuid4())
                    with open(tmpdir / f"{did}.txt", "w", encoding="utf-8") as f:
                        f.write(d.get("text") or "")
                d_resp = self.d_manager.run_pipeline(
                    root_dir=str(tmpdir),
                    collection_base=self.cfg.collection_base
                )
                augmented = self.d_manager.get_augmented_chunks() or []
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        if not augmented:
            return {
                "status": "success",
                "input_type": "docs(records)",
                "total_sec": round(time.time() - t0, 2),
                "d_manager_summary": d_resp,
                "f_manager_summary": "skipped_no_augmented",
            }

        augmented_dicts: List[Dict[str, Any]] = [dict(x) for x in augmented]
        f_res = self.f_manager.run_from_augmented(
            augmented=augmented_dicts,
            collection_base=self.cfg.collection_base,
            paraphrase_n=self.cfg.faq_paraphrase_n,
            metric=self.cfg.metric_type,
        )
        return {
            "status": "success",
            "input_type": "docs(records)",
            "total_sec": round(time.time() - t0, 2),
            "d_manager_summary": d_resp,
            "f_manager_summary": f_res.get("summary"),
            "indexing_responses": {"docs": d_resp, "faqs": f_res.get("resp", {})},
        }

    # --- API 2: chạy từ FAQ roots đã chuẩn hoá ---
    def run_from_faqs(self, roots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        roots: List[{'question': str, 'answer': str, ...}]
        Enrich (paraphrase Q, giữ A) -> embed & index FAQ.
        """
        t0 = time.time()
        if not roots:
            return {"status": "error", "message": "Empty roots."}

        f_res = self.f_manager.run_from_roots(
            roots=roots,
            collection_base=self.cfg.collection_base,
            paraphrase_n=self.cfg.faq_paraphrase_n,
            metric=self.cfg.metric_type,
        )
        return {
            "status": "success",
            "input_type": "faqs(roots)",
            "total_sec": round(time.time() - t0, 2),
            "f_manager_summary": f_res.get("summary"),
            "indexing_responses": {"faqs": f_res.get("resp", {})},
        }

    # --- API 3: tiện ích CSV cột `context` ---
    def run_from_csv_contexts(
        self,
        csv_path: str,
        context_col: str = "context",
        id_col: Optional[str] = None,
        min_len: int = 5,
        default_source: str = "csv_src",
    ) -> Dict[str, Any]:
        """
        Đọc 1 CSV, lấy cột `context` xem như tài liệu -> chạy full docs pipeline.
        - Yêu cầu DocLoaderLC có load_csv_contexts() và to_augmented_inputs().
        - Bảo toàn 'doc_id' theo từng dòng (ưu tiên metadata.doc_id).
        """

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        # 1) Đọc CSV thành Document-like
        docs_lc = self.doc_loader.load_csv_contexts(
            csv_path=csv_path,
            context_col=context_col,
            id_col=id_col,
            source_col=None,
            min_len=min_len,
        )
        # 2) Chuẩn hoá -> [{doc_id,text,metadata}]
        augmented = self.doc_loader.to_augmented_inputs(docs_lc, default_source=default_source)
        for x in augmented:
            x["doc_id"] = x.get("metadata", {}).get("doc_id", x["doc_id"])

        total = len(augmented)
        for i, x in enumerate(augmented, 1):
            did = x.get("doc_id")
            print(f"[MetaManager] Processing {i}/{total} doc_id={did}")

        # 3) Chạy như records
        return self.run_from_documents(augmented)