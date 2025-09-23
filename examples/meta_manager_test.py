# -*- coding: utf-8 -*-
"""
examples/meta_manager_test.py — Smoke test cho MetaManager
- Tạo dữ liệu demo (docs & faqs)
- Chạy pipeline 'docs' (Doc → Augmented → FAQ → Index)
- Chạy pipeline 'faqs' (FAQ roots → Enrich → Index)
- Không có assert; chỉ in kết quả để quan sát.

Chạy:
  uv run examples/meta_manager_test.py
Yêu cầu:
  - Đã tạo file src/managers/meta_manager.py theo bản mới.
  - Có GEMINI_API_KEY (hoặc GOOGLE_API_KEY) trong môi trường.
"""

from __future__ import annotations
import os
import json
import time
from typing import Any, Dict

# Import MetaManager
from src.managers.meta_manager import MetaManager, MetaManagerConfig  # chỉnh path nếu bạn đặt khác

def make_demo_inputs(base_dir: str) -> Dict[str, str]:
    """
    Tạo dữ liệu demo:
      - <base_dir>/demo_docs/*.txt
      - <base_dir>/demo_faqs/root.jsonl
    Trả về dict các đường dẫn.
    """
    docs_dir = os.path.join(base_dir, "demo_docs")
    faqs_dir = os.path.join(base_dir, "demo_faqs")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(faqs_dir, exist_ok=True)

    # demo docs
    with open(os.path.join(docs_dir, "a.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Quy chế tuyển sinh năm nay có nêu mốc thời gian đăng ký và ngưỡng đảm bảo chất lượng đầu vào (điểm sàn). "
            "Sinh viên xuất sắc có thể được xét học bổng theo quy định."
        )
    with open(os.path.join(docs_dir, "b.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Cơ sở vật chất gồm thư viện, phòng thí nghiệm AI và hạ tầng tính toán hiệu năng cao phục vụ nghiên cứu."
        )

    # demo FAQ roots (jsonl)
    faq_path = os.path.join(faqs_dir, "root.jsonl")
    with open(faq_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question": "Điểm sàn tuyển sinh là bao nhiêu?",
            "answer": "Điểm sàn được công bố theo từng năm trong thông báo của nhà trường."
        }, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "question": "Nhà trường có học bổng cho sinh viên xuất sắc không?",
            "answer": "Có, chương trình học bổng áp dụng theo quy định và thông báo hiện hành của trường."
        }, ensure_ascii=False) + "\n")

    return {"docs_dir": docs_dir, "faq_file": faq_path}

def main() -> None:
    t_start = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    paths = make_demo_inputs(base_dir)

    # Đặt collection_base duy nhất theo thời gian
    collection_base = f"rag_meta_test_{int(time.time())}"
    print(f"[i] collection_base = {collection_base}")

    # Cấu hình MetaManager (điều chỉnh nhẹ để chạy nhanh)
    cfg = MetaManagerConfig(
        collection_base=collection_base,
        metric_type="COSINE",
        language="vi",
        # FAQ (Gemini)
        faq_model_name="gemini-1.5-flash",
        faq_min_pairs=3,       # số cặp generate tối thiểu / chunk
        faq_paraphrase_n=2,    # số biến thể câu hỏi / seed
        faq_api_key_env="GEMINI_API_KEY",
    )

    # Khởi tạo orchestrator
    try:
        mm = MetaManager(cfg)
    except Exception as e:
        print(f"[ERR] Khởi tạo MetaManager thất bại: {e}")
        raise

    # -------- Pipeline 1: DOCS --------
    print("\n=============================================")
    print(">>> EXECUTING DOCUMENT-TO-FAQ PIPELINE <<<")
    print("=============================================")
    try:
        res_docs = mm.run(input_path=paths["docs_dir"], input_type="docs")
        print("\n--- RESULT: DOCS PIPELINE ---")
        print(json.dumps(res_docs, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[ERR] Lỗi khi chạy pipeline 'docs': {e}")
        import traceback; traceback.print_exc()

    # -------- Pipeline 2: FAQ-ONLY --------
    print("\n\n=============================================")
    print(">>> EXECUTING FAQ-ONLY PIPELINE <<<")
    print("=============================================")
    try:
        res_faqs = mm.run(input_path=paths["faq_file"], input_type="faqs")
        print("\n--- RESULT: FAQ PIPELINE ---")
        print(json.dumps(res_faqs, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[ERR] Lỗi khi chạy pipeline 'faqs': {e}")
        import traceback; traceback.print_exc()

    print(f"\n[✓] Done in {round(time.time() - t_start, 2)}s")

if __name__ == "__main__":
    main()
