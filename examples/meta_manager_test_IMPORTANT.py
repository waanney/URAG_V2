# -*- coding: utf-8 -*-
"""
Smoke test cho MetaManager (pipeline ingest & index)
- Tạo demo docs/faqs
- Đảm bảo KERNEL có active_config (ưu tiên Gemini nếu có GEMINI_API_KEY)
- Chạy cả 2 pipeline: 'docs' và 'faqs', in summary ra màn hình

Chạy:
    # PowerShell:
    #   $env:GEMINI_API_KEY="sk-..."   # khuyến nghị dùng Gemini để khỏi cài openai/ollama
    # Bash:
    #   export GEMINI_API_KEY=sk-...
    uv run python -m examples.meta_manager_smoketest
"""
from __future__ import annotations
import os
import json
from typing import Optional

from dotenv import load_dotenv

# Kernel (để model/provider do UI hoặc ENV quyết định)
from src.llm.llm_kernel import KERNEL, GoogleConfig, OllamaConfig

# MetaManager (đã kernel-hoá)
from src.managers.meta_manager import MetaManager, MetaManagerConfig


def ensure_kernel_active():
    """
    - Nếu UI trước đó đã lưu .llm_config.json -> load lại
    - Nếu chưa có:
        + Nếu có GEMINI_API_KEY -> ép Google (Gemini)
        + elif USE_OLLAMA=1 -> dùng Ollama local
        + else -> báo lỗi hướng dẫn set ENV
    """
    cfg = KERNEL.load_active_config()
    if cfg is not None: 
        print(f"[KERNEL] Loaded active config: provider={cfg.provider}, model={cfg.model}")
        return

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        # ép dùng Gemini với model mặc định (hoặc override qua GOOGLE_MODEL)
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

    raise SystemExit(
        "No active LLM config found.\n"
        "- Set GEMINI_API_KEY to use Gemini (recommended), e.g. export GEMINI_API_KEY=sk-...\n"
        "- Or set USE_OLLAMA=1 (and run ollama serve) to use a local model."
    )


def make_demo_data():
    os.makedirs("demo_docs", exist_ok=True)
    with open("demo_docs/a.txt", "w", encoding="utf-8") as f:
        f.write(
            "Điểm sàn, mốc thời gian đăng ký xét tuyển, và chính sách học bổng nếu có sẽ được công bố chính thức."
        )

    os.makedirs("demo_faqs", exist_ok=True)
    with open("demo_faqs/root.jsonl", "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "question": "Điểm sàn tuyển sinh là bao nhiêu?",
                    "answer": "Nhà trường công bố theo từng năm."
                },
                ensure_ascii=False
            ) + "\n"
        )


def pretty(obj, title: Optional[str] = None):
    if title:
        print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    load_dotenv()
    ensure_kernel_active()
    make_demo_data()

    # Config tối giản: model/key do KERNEL quyết định
    cfg = MetaManagerConfig(
        collection_base="rag_meta_demo",
        faq_min_pairs=3,
        faq_paraphrase_n=2,
        language="default",
    )
    mm = MetaManager(cfg)

    # Pipeline 1: từ docs
    print("\n>>> DOCS PIPELINE")
    res_docs = mm.run("demo_docs", "docs")
    pretty(res_docs, "Docs Result")

    # Pipeline 2: từ faqs (jsonl)
    print("\n>>> FAQS PIPELINE")
    res_faqs = mm.run("demo_faqs/root.jsonl", "faqs")
    pretty(res_faqs, "FAQs Result")


if __name__ == "__main__":
    main()
