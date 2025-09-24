# -*- coding: utf-8 -*-
"""
Quick & Simple Runner for FManager (no CLI fuss)
================================================
Chạy thẳng: `python -m examples.run_f_manager_quicktest`

Kịch bản:
- Thử lấy augmented từ `./augmented_quicktest.jsonl` (nếu có).
- Nếu không có, đọc `./data/docs` để tự tạo augmented tối thiểu (original=transformed=text).
- Dùng `SimpleFaqGenerator` (heuristic) để tạo FAQ gốc + paraphrase.
- Dùng `EmbedderAdapter` (bọc `embedding.EmbedderAgent`) để encode.
- Upsert vào Milvus bằng `IndexingAgent` (đến collection `*_quickfaq__faq`).
- In summary + stats, rồi dọn collection.
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Any, Dict, List

from src.managers.urag_f_manager import FManager, FManagerConfig
from src.managers.urag_f_manager import AugmentedChunk, FAQItem, FAQWithVec, IFaqGenerator, IEmbedder
from src.indexing.indexing_agent import IndexingAgent, AgentConfig, StatsReq
from pymilvus import utility

# ---- Embedder adapter (🆕 bọc EmbedderAgent để khớp IEmbedder) ----
from embedding.embedding_agent import EmbConfig, EmbedderAgent


class EmbedderAdapter(IEmbedder):  # type: ignore[misc]
    def __init__(self, language: str = "default") -> None:
        self.agent = EmbedderAgent(
            EmbConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                vi_model_name="dangvantuan/vietnamese-embedding",
                language=("vi" if language == "vi" else "default"),
                normalize_for_cosine=True,
                metric="COSINE",
            )
        )

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        return self.agent.encode(texts)  # type: ignore[attr-defined]

    def info(self) -> Dict[str, Any]:
        return {"embedder": "EmbedderAgent", "dim_hint": 384}


# ---- Simple FAQ generator (🆕 heuristic, không cần LLM) ----
class SimpleFaqGenerator(IFaqGenerator):  # type: ignore[misc]
    def generate_roots(self, augmented_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        roots: List[Dict[str, Any]] = []
        for row in augmented_chunks:
            doc_id = str(row.get("doc_id", "doc"))
            chunk_id = str(row.get("chunk_id", "0"))
            # Lấy text từ các khóa phổ biến (transformed/original/text)
            text = str(row.get("text") or row.get("transformed") or row.get("original") or "").strip()
            if not text:
                continue
            q = f"Nội dung chính của đoạn {chunk_id} là gì?"
            a = text if len(text) <= 512 else (text[:512] + "…")
            roots.append({
                "question": q,
                "answer": a,
                "canonical_id": f"{doc_id}::{chunk_id}",
                "source": "faq_src",
                "metadata": {"from_chunk": chunk_id, "doc_id": doc_id}
            })
        return roots

    def enrich_from_roots(self, roots: List[Dict[str, Any]], paraphrase_n: int = 5) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in roots:
            out.append({**r, "is_paraphrase": False})  # giữ root
            base_q = r["question"]
            a = r["answer"]
            # tạo vài paraphrase đơn giản
            templates = [
                "Tóm tắt nội dung đoạn này là gì?",
                "Đoạn văn này nói về điều gì?",
                "Điểm chính của đoạn là gì?",
                "Bạn có thể giải thích đoạn này không?",
                "Nội dung đoạn văn tập trung vào vấn đề nào?",
            ]
            for i, t in enumerate(templates[:paraphrase_n]):
                out.append({
                    **r,
                    "question": t,
                    "answer": a,
                    "is_paraphrase": True,
                })
        return out


# ---- Utilities ----
AUG_PATH = Path("./augmented_quicktest.jsonl")
DOCS_DIR = Path("./data/docs")
LANGUAGE = os.getenv("URAG_LANG", "default")
COLLECTION_BASE = f"ura_rag_quickfaq_{int(time.time())}"
DO_CLEANUP = True


def load_augmented() -> List[Dict[str, Any]]:
    if AUG_PATH.exists():
        data: List[Dict[str, Any]] = []
        with AUG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
        if data:
            print(f"[QuickFAQ] Loaded augmented from {AUG_PATH} -> {len(data)} items")
            return data
    # Fallback: tạo augmented tối thiểu từ docs
    print("[QuickFAQ] No augmented file found, fallback to docs → minimal augmented")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if not any(DOCS_DIR.iterdir()):
        (DOCS_DIR / "hello.txt").write_text("Xin chào, đây là tài liệu mẫu cho FManager.", encoding="utf-8")
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(sorted(DOCS_DIR.glob("**/*"))):
        if p.is_dir():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        out.append({
            "doc_id": p.stem,
            "chunk_id": f"c{i}",
            "original": text,
            "transformed": text,
            "metadata": {"path": str(p)}
        })
    print(f"[QuickFAQ] Built minimal augmented: {len(out)} items")
    return out


def main() -> int:
    print("[QuickFAQ] Starting…")

    # 1) Chuẩn bị components
    faq_gen = SimpleFaqGenerator()
    embedder = EmbedderAdapter(language=LANGUAGE)
    fcfg = FManagerConfig(embed_field="question", l2_normalize=True)
    fm = FManager(faq_gen, embedder, fcfg)

    # 2) Load augmented
    augmented = load_augmented()

    # 3) Run full pipeline (augmented → roots → enrich → embed → index)
    result = fm.run_from_augmented(
        augmented,
        collection_base=COLLECTION_BASE,
        paraphrase_n=3,
        metric="COSINE",
        index_params=None,
        shards=2,
        build_index=True,
    )

    print("\n=== Summary ===")
    print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))

    # 4) Stats nhanh (FAQ collection)
    try:
        ia = IndexingAgent(AgentConfig())
        faq_col = COLLECTION_BASE + "__faq"
        stats = ia.process(StatsReq(op="stats", collection=faq_col).model_dump())
        print("\n=== FAQ Stats ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[WARN] stats failed: {e}")

    # 5) Cleanup
    if DO_CLEANUP:
        try:
            for name in [COLLECTION_BASE + "__doc", COLLECTION_BASE + "__faq", COLLECTION_BASE]:
                if utility.has_collection(name):
                    utility.drop_collection(name)
                    print(f"[Cleanup] Dropped collection: {name}")
        except Exception as e:
            print(f"[WARN] cleanup failed: {e}")

    print("\n[QuickFAQ] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
