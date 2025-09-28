# -*- coding: utf-8 -*-
"""
Nạp toàn bộ tài liệu trong thư mục ROOT_DIR -> MetaManager (semantic chunk + index + FAQ pipeline nếu có).
"""

from __future__ import annotations
import os, json
from typing import List, Dict, Any

# ====== CONFIG ======
ROOT_DIR       = "datasets/docs"   # <--- thư mục chứa .pdf, .docx, .txt, .md, .csv
COLLECTION     = "viquad_final_0"  # tên base collection Milvus
LANGUAGE       = "default"         # "vi" | "default"
DEFAULT_SOURCE = "doc_src"         # nếu file thiếu metadata.source
LIMIT_DOCS     = None              # đặt số để test nhanh (vd 50), hoặc None

# (tuỳ chọn) FAQ JSON/JSONL bổ sung cùng lúc
USE_FAQ        = False
FAQ_PATH       = "datasets/faq.jsonl"

# ====== BOOT KERNEL (chỉ cần nếu bạn có bước FAQ enrichment dùng LLM) ======
from src.llm.llm_kernel import KERNEL, GoogleConfig, OllamaConfig, OpenAIConfig

def ensure_kernel_active():
    loaded = KERNEL.load_active_config()
    if loaded is not None:
        print(f"[KERNEL] Loaded active config: provider={loaded.provider}, model={loaded.model}")
        return
    if os.getenv("OPENAI_API_KEY"):
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        KERNEL.set_active_config(OpenAIConfig(model=model_name, api_key=api_key))
        print(f"[KERNEL] Active=OpenAI (model={model_name})")
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
    print("[WARN] No active LLM config found. FAQ enrichment (nếu có) có thể bị bỏ qua.")

# ====== MANAGER & LOADER ======
from src.managers.meta_manager import MetaManager, MetaManagerConfig
from src.llm.URag_D.doc_loader_agent import DocumentLoaderAgent, DLRequest, DLConfig

def main():
    ensure_kernel_active()

    # 1) Khởi tạo MetaManager
    mm = MetaManager(MetaManagerConfig(
        collection_base=COLLECTION,
        language=LANGUAGE,
    ))

    # 2) Đọc tài liệu (và optional: FAQ) bằng DocumentLoaderAgent
    loader = DocumentLoaderAgent(
        DLConfig(
            pdf_glob="**/*.pdf",
            docx_glob="**/*.docx",
            txt_glob="**/*.txt",
            md_glob="**/*.md",
            csv_glob="**/*.csv",      # CSV sẽ được coi như văn bản nếu không dùng CSVLoader
            use_unstructured=False,   # bật True + other_glob nếu bạn có Unstructured
            autodetect_encoding=True,
        )
    )

    mode = "both" if USE_FAQ else "normal"
    req = DLRequest(
        mode=mode,
        root_dir=ROOT_DIR,
        faq_path=FAQ_PATH if USE_FAQ else None,
        default_source=DEFAULT_SOURCE,
        limit_docs=LIMIT_DOCS,
    )
    resp = loader.run(req)
    docs = [d.model_dump() for d in resp.documents]   # -> [{doc_id, text, metadata}, ...]

    # 3) Đẩy vào MetaManager
    if hasattr(mm, "run_from_documents"):
        out = mm.run_from_documents(docs)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        # Fallback: nếu MetaManager cũ chưa có run_from_documents
        import tempfile, shutil
        from pathlib import Path
        tmpdir = Path(tempfile.mkdtemp(prefix="docs_load_"))
        try:
            for i, d in enumerate(docs, 1):
                with open(tmpdir / f"{d['doc_id'] or ('doc_'+str(i))}.txt", "w", encoding="utf-8") as f:
                    f.write(d["text"])
            out = mm.run(str(tmpdir), "docs")
            print(json.dumps(out, ensure_ascii=False, indent=2))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
