# -*- coding: utf-8 -*-
"""
examples/meta_manager_test.py — Smoke test cho MetaManager (có dọn Milvus + sanity search)
- Tạo dữ liệu demo (docs & faqs)
- Chạy pipeline 'docs' (Doc → Augmented → FAQ → Index)
- Chạy pipeline 'faqs' (FAQ roots → Enrich → Index)
- In kết quả để quan sát
- Thực hiện sanity check Milvus: đếm entity, search mẫu
- Cuối cùng: drop collection Milvus <collection_base>__doc và __faq (trừ khi KEEP_MILVUS=1)
"""

from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, cast

# Metric Literal type
from src.indexing.indexing_agent import Metric

# Import MetaManager
from src.managers.meta_manager import MetaManager, MetaManagerConfig  # chỉnh path nếu bạn đặt khác

# Milvus
from pymilvus import connections, utility, Collection

# Embedder để probe search
from src.embedding.embedding_agent import EmbedderAgent, EmbConfig

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB  = os.getenv("MILVUS_DB", None)  # để None nếu dùng default
KEEP       = os.getenv("KEEP_MILVUS", "0") == "1"

def make_demo_inputs(base_dir: str) -> Dict[str, str]:
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
    with open(faq_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "question": "Nhà trường có học bổng cho sinh viên xuất sắc không?",
            "answer": "Có, chương trình học bổng áp dụng theo quy định và thông báo hiện hành của trường."
        }, ensure_ascii=False) + "\n")

    return {"docs_dir": docs_dir, "faq_file": faq_path}

def connect_milvus():
    print(f"[i] Milvus connect: {MILVUS_URI} (db={MILVUS_DB or 'default'})")
    if MILVUS_DB:
        connections.connect(uri=MILVUS_URI, alias="default", db_name=MILVUS_DB)
    else:
        connections.connect(uri=MILVUS_URI, alias="default")

def drop_collection_safe(name: str):
    try:
        if utility.has_collection(name):
            try:
                col = Collection(name)
                col.release()
            except Exception:
                pass
            utility.drop_collection(name)
            print(f"[clean] Dropped collection: {name}")
        else:
            print(f"[clean] Skip (not found): {name}")
    except Exception as e:
        print(f"[clean] Drop failed for {name}: {e}")

def cleanup_milvus(collection_base: str):
    if KEEP:
        print(f"[clean] KEEP_MILVUS=1 → giữ lại collections cho base '{collection_base}'")
        return
    try:
        connect_milvus()
    except Exception as e:
        print(f"[clean] Milvus connect failed, skip cleanup: {e}")
        return
    for suffix in ("__doc", "__faq"):
        drop_collection_safe(f"{collection_base}{suffix}")

# --------- Sanity helpers ---------

def milvus_count(name: str) -> int:
    if not utility.has_collection(name):
        return -1
    col = Collection(name)
    col.load()
    return col.num_entities

def milvus_search(col_name: str, text: str, metric: str = "COSINE", top_k: int = 5):
    try:
        # ép kiểu Metric (Literal) cho EmbedderAgent (Pylance-friendly)
        metric_literal: Metric = cast(Metric, metric)

        e = EmbedderAgent(EmbConfig(language="vi", metric=metric_literal, normalize_for_cosine=True))
        qv = e.encode([text])[0]
        col = Collection(col_name)
        col.load()
        search_params = {"metric_type": metric, "params": {"ef": 128}}
        sr: Any = col.search(
            data=[qv],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id", "question", "answer", "text", "source", "ts"],
        )
        # Chuẩn hóa kiểu trả về: một số stub cho rằng là SearchFuture
        try:
            res = sr.result()  # type: ignore[attr-defined]
        except AttributeError:
            res = sr

        print(f"\n[SEARCH] {col_name} ← '{text}'")
        for hit in res[0]:
            ent = hit.entity
            q, a, t = ent.get("question"), ent.get("answer"), ent.get("text")
            print(f"  dist={hit.distance:.4f} id={ent.get('id')} src={ent.get('source')}")
            if q or a:
                print(f"   Q: {q}\n   A: {a}")
            if t:
                snippet = t[:160] + ("..." if len(t) > 160 else "")
                print(f"   TEXT: {snippet}")
    except Exception as e:
        print(f"[WARN] search failed on {col_name}: {e}")

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
        metric_type=cast(Metric, "COSINE"),  # <-- fix: ép Literal cho Pylance
        language="vi",
        # FAQ LLM (có thể dùng Gemini hoặc backend local OpenAI-compatible nếu bạn đã patch)
        faq_model_name="gemini-1.5-flash",
        faq_min_pairs=3,
        faq_paraphrase_n=2,
        faq_api_key_env="GEMINI_API_KEY",
    )

    # Khởi tạo orchestrator
    try:
        mm = MetaManager(cfg)
    except Exception as e:
        print(f"[ERR] Khởi tạo MetaManager thất bại: {e}")
        raise

    try:
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

        # -------- Milvus sanity checks --------
        try:
            connect_milvus()
            c_doc = f"{collection_base}__doc"
            c_faq = f"{collection_base}__faq"
            n_doc = milvus_count(c_doc)
            n_faq = milvus_count(c_faq)
            print("\n[Sanity] Milvus entity counts:")
            print(f"  {c_doc}: {n_doc}")
            print(f"  {c_faq}: {n_faq}")

            # probe search nếu có dữ liệu
            if (n_faq or 0) > 0:
                milvus_search(c_faq, "Điểm sàn tuyển sinh là bao nhiêu?")
            if (n_doc or 0) > 0:
                milvus_search(c_doc, "học bổng toàn phần cho sinh viên xuất sắc", top_k=5)
        except Exception as e:
            print(f"[WARN] Milvus sanity failed: {e}")

        print(f"\n[✓] Done in {round(time.time() - t_start, 2)}s")

    finally:
        # Luôn cố gắng dọn Milvus trừ khi KEEP_MILVUS=1
        cleanup_milvus(collection_base)

if __name__ == "__main__":
    main()
