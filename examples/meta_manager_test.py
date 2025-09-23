# -*- coding: utf-8 -*-
"""
examples/run_meta_quicktest.py

Quick runner cho MetaManager, tương thích với hai quick-test sẵn có:
- D quick: sinh augmented và (tuỳ chọn) index vào <base>__doc  (tham chiếu logic từ d_manager_test) 
- F quick: nhận augmented (file .jsonl) và index vào <base>__faq (tham chiếu logic từ f_manager_test_2)

Kịch bản hỗ trợ:
  1) --mode doc:   chạy full nhánh DOC, có thể đẩy augmented sang FAQ (mặc định ON)
  2) --mode faq:   chỉ chạy nhánh FAQ từ augmented file hoặc từ docs khi thiếu
  3) --from-aug:   ép dùng augmented_quicktest.jsonl nếu có
  4) --no-faq:     không đẩy augmented sang FManager (doc-only)
  5) --no-augment: patch augment thành NoOp (đi nhanh, không cần LLM)
  6) --real-index: dùng IndexingAgent thật; nếu bỏ, dùng dummy (không cần Milvus)

Chạy mẫu:
  uv run examples/run_meta_quicktest.py --mode doc --docs ./data/docs
  uv run examples/run_meta_quicktest.py --mode doc --docs ./data/docs --real-index --no-augment
  uv run examples/run_meta_quicktest.py --mode faq --from-aug --real-index
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.managers.meta_manager import MetaManager, MetaConfig

AUG_PATH = Path("./augmented_quicktest.jsonl")


# ============= Dummy IndexingAgent (no Milvus) =============
def patch_dummy_indexing_agent() -> None:
    try:
        from src.indexing import indexing_agent as idx_mod
    except Exception as e:
        print(f"[WARN] Không thể import IndexingAgent để mock: {e}")
        return

    class _DummyIndexingAgent:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
        def ensure_collections(self, *a, **k) -> Dict[str, Any]:
            return {"ok": True, "mode": "dummy"}
        def upsert_documents(self, payload: List[Dict[str, Any]], *a, **k) -> Dict[str, Any]:
            return {"ok": True, "n": len(payload), "mode": "dummy-doc"}
        def upsert_faqs(self, payload: List[Dict[str, Any]], *a, **k) -> Dict[str, Any]:
            return {"ok": True, "n": len(payload), "mode": "dummy-faq"}
        def build_indexes(self, *a, **k) -> Dict[str, Any]:
            return {"ok": True, "built": False, "mode": "dummy"}
        def close(self) -> None:
            pass

    idx_mod.IndexingAgent = _DummyIndexingAgent  # type: ignore[attr-defined]
    print("[MOCK] Dummy IndexingAgent enabled (no Milvus required).")


# ============= Helpers =============
def maybe_patch_no_augment() -> None:
    """Patch textGenerate augment -> NoOp để chạy nhanh khi không muốn gọi LLM."""
    try:
        import src.textGenerate as TG  # type: ignore
        if hasattr(TG, "augment"):
            TG.augment = lambda chunks, *a, **k: chunks  # type: ignore
            print("[PATCH] textGenerate.augment -> NoOp")
        if hasattr(TG, "Augmenter"):
            class _NoOpAug:
                def __init__(self, *a, **k): ...
                def run(self, chunks): return chunks
            TG.Augmenter = _NoOpAug  # type: ignore
            print("[PATCH] textGenerate.Augmenter -> NoOp")
    except Exception as e:
        print(f"[WARN] no-augment patch failed: {e}")


def load_augmented_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


# ============= Main runners =============
def run_doc(docs_root: str, *, real_index: bool, no_faq: bool, no_augment: bool) -> None:
    if not real_index:
        patch_dummy_indexing_agent()
    if no_augment:
        maybe_patch_no_augment()

    cfg = MetaConfig(
        collection_base=f"ura_rag_meta_{os.getpid()}",
        language=os.getenv("URAG_LANG", "default"),
        docs_root_dir=docs_root,
        build_index=False,
        shards_num=1,
        faq_paraphrase_n=3,
    )
    mm = MetaManager(cfg)

    print("[Meta] run_auto('doc') …")
    out = mm.run_auto("doc", docs_root_dir=docs_root, also_build_faq=(not no_faq))
    print("[Meta] done. Summary:")
    print(json.dumps(out, ensure_ascii=False, indent=2))


def run_faq(*, real_index: bool, from_aug: bool) -> None:
    if not real_index:
        patch_dummy_indexing_agent()

    cfg = MetaConfig(
        collection_base=f"ura_rag_meta_{os.getpid()}",
        language=os.getenv("URAG_LANG", "default"),
        build_index=False,
        shards_num=1,
        faq_paraphrase_n=3,
    )
    mm = MetaManager(cfg)

    if from_aug:
        augmented = load_augmented_file(AUG_PATH)
        if not augmented:
            print(f"[Meta] Không tìm thấy augmented ở {AUG_PATH}. Sẽ chạy FAQ từ roots tối giản.")
        else:
            out = mm.run_auto("faq", augmented=augmented)
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return

    # Fallback: FAQ từ roots tối giản
    roots = [
        {"question": "Dự án URA-RAG là gì?", "answer": "Pipeline indexing & RAG cho docs/FAQ."},
        {"question": "Milvus dùng làm gì?", "answer": "Lưu vector embeddings và tìm kiếm ANN."},
    ]
    out = mm.run_auto("faq", roots=roots)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["doc", "faq"], default="doc")
    ap.add_argument("--docs", default="./data/docs")
    ap.add_argument("--real-index", action="store_true")
    ap.add_argument("--no-faq", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--from-aug", action="store_true")
    args = ap.parse_args()

    if args.mode == "doc":
        run_doc(args.docs, real_index=args.real_index, no_faq=args.no_faq, no_augment=args.no_augment)
    else:
        run_faq(real_index=args.real_index, from_aug=args.from_aug)
