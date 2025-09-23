# -*- coding: utf-8 -*-
"""
examples/search_test.py — Index -> Search (FAQ -> DOC -> Fallback)
- Không còn assert; chỉ in ra kết quả để quan sát.
"""

from __future__ import annotations
import os, time, json
from typing import Any, Dict, List, Optional, cast

from src.embedding.embedding_agent import EmbedderAgent, EmbConfig
from src.indexing.indexing_agent import (
    IndexingAgent, AgentConfig,
    UpsertIndexReq, Item
)
from src.search.search_agent import (
    SearchAgent, SearchConfig,
    GeminiConfig,  # để cấu hình nếu dùng Gemini thật
    GeminiLLM      # chỉ để dùng cho typing cast
)

# ---------- Dummy LLM (dùng khi không có GEMINI_API_KEY) ----------
class DummyLLM:
    def generate(self, prompt: str) -> str:
        start = prompt.find("<CONTEXT>")
        end = prompt.find("</CONTEXT>") if "</CONTEXT>" in prompt else len(prompt)
        ctx = prompt[start:end].replace("<CONTEXT>", "").strip() if start != -1 else ""
        return (ctx[:600] + (" ..." if len(ctx) > 600 else "")) if ctx else "Không có thông tin."

def print_hits(title: str, hits: Optional[List[Any]]) -> None:
    print(f"\n[HITS] {title}")
    if not hits:
        print("  (empty)")
        return
    for i, h in enumerate(hits, 1):
        sim = getattr(h, "similarity", None)
        score = getattr(h, "score", None)
        src = getattr(h, "source", None)
        q = getattr(h, "question", None)
        a = getattr(h, "answer", None)
        txt = getattr(h, "text", None)
        if q or a:
            print(f"  #{i}: sim={sim:.3f} score={score:.3f} src={src} Q={q!r} A={a!r}")
        else:
            snippet = (txt[:120] + "…") if txt and len(txt) > 120 else txt
            print(f"  #{i}: sim={sim:.3f} score={score:.3f} src={src} TEXT={snippet!r}")

def main() -> None:
    base = f"ura_search_demo_{int(time.time())}"
    print(f"[i] Collection base: {base}")

    # --- Embedder ---
    e_cfg = EmbConfig(language="vi", metric="COSINE", normalize_for_cosine=True)
    embedder = EmbedderAgent(e_cfg)

    # --- Indexer ---
    i_cfg = AgentConfig(
        alias="default",
        dual_collections=True,
        normalize_l2_for_cosine=True,
    )
    indexer = IndexingAgent(i_cfg)

    # --- Dữ liệu demo ---
    faqs: List[Dict[str, Any]] = [
        {"question": "Trường có tuyển sinh hệ đại trà không?", "answer": "Có. Trường có tuyển sinh hệ đại trà theo đề án hằng năm."},
        {"question": "Điểm sàn tuyển sinh là bao nhiêu?", "answer": "Điểm sàn được công bố theo từng năm; tham khảo thông báo của nhà trường."},
    ]
    faq_embeds = embedder.embed_faqs(faqs, default_source="demo_faq")

    docs: List[Dict[str, Any]] = [
        {"text": "Hướng dẫn tuyển sinh năm nay nêu rõ các mốc thời gian đăng ký và điểm sàn dự kiến.", "source": "doc_guide_2025"},
        {"text": "Thông tin học bổng: sinh viên xuất sắc có thể nhận học bổng toàn phần theo quy định.", "source": "doc_scholarship"},
        {"text": "Cơ sở vật chất bao gồm thư viện, phòng thí nghiệm AI, và hạ tầng tính toán hiệu năng cao.", "source": "doc_facilities"},
    ]
    # Tạo union-list cho đúng annotation của embed_docs
    docs_union: List[Dict[str, Any] | str] = list(docs)
    doc_embeds = embedder.embed_docs(docs_union, default_source="demo_doc")

    dim = embedder.dim
    print(f"[i] Embedding dim = {dim}")

    # --- Chuẩn bị items để upsert ---
    items: List[Item] = []
    for k, it in enumerate(faq_embeds, start=1):
        items.append(Item(
            id=f"faq_{k}",
            type="faq",
            vector=it["vector"],
            question=it["question"],
            answer=it["answer"],
            source=it.get("source", "faq"),
            metadata=it.get("metadata", {}),
            ts=it.get("ts", 0),
        ))
    for k, it in enumerate(doc_embeds, start=1):
        items.append(Item(
            id=f"doc_{k}",
            type="doc",
            vector=it["vector"],
            text=it["text"],
            source=it.get("source", "doc"),
            metadata=it.get("metadata", {}),
            ts=it.get("ts", 0),
        ))

    up_req = UpsertIndexReq(
        op="upsert",
        collection=base,    # chỉ base; agent tự thêm __faq / __doc
        dim=dim,
        metric_type="COSINE",
        items=items,
        shards_num=2,
        description="Demo URAG dual collections",
        index_params=None,
        build_index=True
    )
    up_resp = indexer.process(up_req)
    print("[i] Upsert response:")
    print(json.dumps(up_resp, ensure_ascii=False, indent=2))

    # --- SearchAgent ---
    s_cfg = SearchConfig(
        collection_base=base,
        faq_top_k=10, doc_top_k=5,
        # Ưu tiên FAQ cao hơn (hạ tFAQ, nâng tDOC). Tuỳ bạn tinh chỉnh.
        tFAQ=0.70, tDOC=0.60,
        metric="COSINE",
        max_ctx_docs=3,
        disclaimer="Lưu ý: câu trả lời được sinh dựa trên ngữ cảnh tài liệu."
    )

    gem_api = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if gem_api:
        print("[i] GEMINI_API_KEY detected -> Tier-2 gọi Gemini.")
        searcher = SearchAgent(
            s_cfg=s_cfg,
            e_cfg=e_cfg,
            i_cfg=i_cfg,
            g_cfg=GeminiConfig(model="gemini-2.5-flash"),
            llm=None,  # SearchAgent sẽ tự khởi tạo GeminiLLM
        )
    else:
        print("[i] Không có GEMINI_API_KEY -> dùng DummyLLM (offline).")
        dummy = cast(GeminiLLM, DummyLLM())  # duck-typing .generate
        searcher = SearchAgent(
            s_cfg=s_cfg,
            e_cfg=e_cfg,
            i_cfg=i_cfg,
            g_cfg=GeminiConfig(model="gemini-2.5-flash"),
            llm=dummy,
        )

    # --- TEST 1: FAQ ---
    q1 = "Điểm sàn tuyển sinh là bao nhiêu?"
    r1 = searcher.answer(q1)
    print("\n[TEST-1] Query:", q1)
    print(json.dumps(r1.model_dump(), ensure_ascii=False, indent=2))
    print_hits("FAQ top-k", r1.trace.faq_hits)
    print_hits("DOC top-k (nếu có)", r1.trace.doc_hits)
    print(f"[TEST-1] => tier = {r1.tier}")

    # --- TEST 2: DOC (không khớp FAQ) ---
    q2 = "Học bổng toàn phần có áp dụng cho sinh viên xuất sắc không?"
    r2 = searcher.answer(q2)
    print("\n[TEST-2] Query:", q2)
    print(json.dumps(r2.model_dump(), ensure_ascii=False, indent=2))
    print_hits("FAQ top-k", r2.trace.faq_hits)
    print_hits("DOC top-k (nếu có)", r2.trace.doc_hits)
    print(f"[TEST-2] => tier = {r2.tier}")

    # --- TEST 3: NO HIT ---
    q3 = "Tàu vũ trụ của NASA có đậu trước cổng trường không?"
    r3 = searcher.answer(q3)
    print("\n[TEST-3] Query:", q3)
    print(json.dumps(r3.model_dump(), ensure_ascii=False, indent=2))
    print_hits("FAQ top-k", r3.trace.faq_hits)
    print_hits("DOC top-k (nếu có)", r3.trace.doc_hits)
    print(f"[TEST-3] => tier = {r3.tier}")

    print("\n[•] Done. Quan sát tier/hits/similarity để tinh chỉnh tFAQ/tDOC, ef, v.v.")

if __name__ == "__main__":
    main()
