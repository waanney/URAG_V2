# examples/run_csv_context_into_manager.py
# -*- coding: utf-8 -*-
"""
Run thẳng: nạp cột 'context' từ 1 CSV -> đẩy vào MetaManager (index + FAQ pipeline).
- Không nhận tham số CLI. Chỉnh các hằng số CONFIG phía dưới rồi chạy file này.
"""

from __future__ import annotations
import os, json
from typing import List, Dict, Any

# ======================== CONFIG ========================
CSV_PATH       = "datasets/train_multihop.csv"  # <-- ĐƯỜNG DẪN CSV CỦA BẠN
CONTEXT_COL    = "context"               # tên cột mang nội dung
ID_COL         = "id"                    # cột id (optional, để None nếu không có)
MIN_LEN        = 5                       # bỏ qua context quá ngắn
COLLECTION     = "viquad_demo_1"      # tên base của collection Milvus
LANGUAGE       = "default"               # "vi" | "default"
DEFAULT_SOURCE = "csv_src"

# (tuỳ chọn) đặt GEMINI_API_KEY/GOOGLE_API_KEY để FAQ enrichment hoạt động tốt
# os.environ["GEMINI_API_KEY"] = "sk-..."

# ===================== BOOT LLM KERNEL =====================
from src.llm.llm_kernel import KERNEL, GoogleConfig, OllamaConfig

def ensure_kernel_active():
    cfg = KERNEL.load_active_config()
    if cfg is not None:
        print(f"[KERNEL] Loaded active config: provider={cfg.provider}, model={cfg.model}")
        return
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        KERNEL.set_active_config(GoogleConfig(model=model_name, api_key=api_key))
        print(f"[KERNEL] Active=Google (model={model_name})")
        return
    if os.getenv("USE_OLLAMA", "0") == "1":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        KERNEL.set_active_config(OllamaConfig(model=model_name, base_url=base_url))
        print(f"[KERNEL] Active=Ollama (model={model_name}, url={base_url})")
        return
    print("[WARN] No active LLM config found. FAQ enrichment may be skipped.")

# ===================== MANAGER & LOADER ====================
from src.managers.meta_manager import MetaManager, MetaManagerConfig
from src.llm.URag_D.doc_loader_agent import DocumentLoaderAgent

def main():
    ensure_kernel_active()

    # Khởi tạo MetaManager
    mm = MetaManager(MetaManagerConfig(
        collection_base=COLLECTION,
        language=LANGUAGE,
    ))

    # 1) Nếu MetaManager đã có tiện ích run_from_csv_contexts => dùng luôn
    if hasattr(mm, "run_from_csv_contexts"):
        print("[MetaManager] Using run_from_csv_contexts(...)")
        res = mm.run_from_csv_contexts(
            csv_path=CSV_PATH,
            context_col=CONTEXT_COL,
            id_col=ID_COL,
            min_len=MIN_LEN,
            default_source=DEFAULT_SOURCE
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # 2) Fallback: dùng DocumentLoaderAgent hút context -> run_from_documents
    print("[MetaManager] Fallback: agent.load_context_csv_records -> run_from_documents")
    agent = DocumentLoaderAgent()
    records = agent.load_context_csv_records(
        csv_path=CSV_PATH,
        context_col=CONTEXT_COL,
        id_col=ID_COL,
        min_len=MIN_LEN,
        default_source=DEFAULT_SOURCE
    )
    docs: List[Dict[str, Any]] = [r.model_dump() for r in records]

    if hasattr(mm, "run_from_documents"):
        res = mm.run_from_documents(docs)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # 3) Fallback cuối (chỉ khi bạn chưa thêm run_from_documents): ghi tạm .txt rồi chạy "docs"
    import tempfile, shutil
    from pathlib import Path
    tmpdir = Path(tempfile.mkdtemp(prefix="csv_ctx_docs_"))
    try:
        for d in docs:
            with open(tmpdir / f"{d['doc_id']}.txt", "w", encoding="utf-8") as f:
                f.write(d["text"])
        res = mm.run(str(tmpdir), "docs")
        print(json.dumps(res, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
