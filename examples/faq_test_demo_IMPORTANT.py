# -*- coding: utf-8 -*-
"""
Smoke test cho FAQAgent (URAG-F) với KERNEL (model do UI chọn).

Cách chạy:
    # 1) (Khuyến nghị) đã có cấu hình active lưu bằng KERNEL.save_active_config()
    uv run python -m examples.faq_smoketest

    # 2) Nếu chưa có, script sẽ tự suy đoán:
    #    - Nếu USE_OLLAMA=1 => dùng Ollama (cần ollama serve & model đã pull)
    #    - Ngược lại, nếu có GEMINI_API_KEY/GOOGLE_API_KEY => dùng Gemini
"""

from __future__ import annotations
import os
from typing import List

from src.llm.URag_F.FAQ import FAQAgent, FAQPair, FAQList  # điều chỉnh import nếu path khác
from src.llm.llm_kernel import KERNEL, GoogleConfig, OllamaConfig
from dotenv import load_dotenv
load_dotenv()

def ensure_active_config():
    """
    Đảm bảo KERNEL đã có active_config:
    - Thử load từ file .llm_config.json (nếu trước đó UI đã lưu).
    - Nếu không có, dùng heuristic ENV để set nhanh:
        USE_OLLAMA=1  -> Ollama
        GEMINI_API_KEY -> Google Gemini
    """
    KERNEL.set_active_config(GoogleConfig(model="gemini-2.0-flash"))
    print("[KERNEL] Active=Google (Gemini, forced in script)")
    return


def pretty_print(title: str, faqs: List[FAQPair] | FAQList) -> None:
    print(f"\n=== {title} ===")
    items: List[FAQPair]
    if isinstance(faqs, FAQList):
        items = faqs.faqs
    else:
        items = faqs
    for i, it in enumerate(items, 1):
        print(f"{i}. Q: {it.question}\n   A: {it.answer}")


def main():
    ensure_active_config()

    # Bạn có thể thay document mẫu này bằng nội dung thực.
    sample_doc = """
    Quy chế tuyển sinh quy định thời gian đăng ký xét tuyển, ngưỡng đảm bảo chất lượng đầu vào (điểm sàn),
    và các tiêu chí xét học bổng cho sinh viên xuất sắc nếu có. Mọi thông tin chi tiết công bố trong thông báo của trường.
    """

    # model_name=None -> dùng model từ active_config (UI chọn).
    agent = FAQAgent(
        api_key=None,
        model_name="gemini-2.0-flash",
        min_pairs=4,
        enrich_pairs_per_seed=4,
    )

    # 1) Generate
    base = agent.generate(sample_doc)
    pretty_print("Base FAQs", base)

    # 2) Enrich (paraphrase câu hỏi, giữ nguyên answer)
    more = agent.enrich(base)
    pretty_print("Enriched (paraphrased questions, SAME answers)", more)

    # 3) Pipeline kết hợp
    merged = agent.generate_and_enrich(sample_doc, do_enrich=True)
    pretty_print("Merged FAQs", merged)


if __name__ == "__main__":
    main()
