# -*- coding: utf-8 -*-
"""
Quick & Simple Runner (no CLI fuss)
===================================
Chạy thẳng: `python -m examples.run_urag_d_quicktest`

- Tự set default:
  - docs_root = ./data/docs (tự tạo ví dụ nếu chưa có)
  - collection_base = ura_rag_quicktest_<timestamp>
  - language = "vi" nếu muốn, mặc định "default"
  - metric/index type dùng mặc định trong DManagerConfig
- Kiểm tra Milvus health (không bắt buộc), tạo collection, index, in summary.
- Lưu augmented ra ./augmented_quicktest.jsonl

Bạn có thể chỉnh nhanh các biến ở block CONFIG dưới đây.
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Any, Dict

from managers.urag_d_manager import URagDManager, DManagerConfig
from indexing.indexing_agent import IndexingAgent, AgentConfig, HealthReq, StatsReq
from pymilvus import utility

# ---------------- CONFIG (chỉnh ở đây nếu cần) ----------------
LANGUAGE = os.getenv("URAG_LANG", "default")  # "default" | "vi"
DOCS_DIR = Path("./data/docs")
SAVE_AUG = Path("./augmented_quicktest.jsonl")
COLLECTION_BASE = f"ura_rag_quicktest_{int(time.time())}"
INDEX_PARAMS: Dict[str, Any] = {"M": 32, "efConstruction": 200}  # cho HNSW
SHARDS = 2
BUILD_INDEX = True
DO_CLEANUP = True  # drop created collections at the end
# --------------------------------------------------------------


def _ensure_sample_docs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Tạo file ví dụ nếu thư mục trống
    if not any(root.iterdir()):
        (root / "hello_vi.txt").write_text(
            """Dự án URAG: Đây là tài liệu ví dụ để kiểm tra pipeline chunk → augment → embed → index.""",
            encoding="utf-8",
        )
        (root / "hello_en.txt").write_text(
            """URAG project: This is a small sample document to exercise the pipeline.""",
            encoding="utf-8",
        )


def _print_json(obj: Any) -> None:
    try:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except Exception:
        print(obj)


def main():
    print("[QuickTest] Starting…")

    ia = None  # for Pylance: ensure bound name even if health check fails
    # 0) Optional: kiểm tra Milvus
    try:
        ia = IndexingAgent(AgentConfig())
        health = ia.process(HealthReq(op="health").model_dump())
        ok = bool(health.get("data", {}).get("ok", False))
        print(f"[Milvus] health: {'OK' if ok else 'NOT OK'}")
        if not ok:
            _print_json(health)
    except Exception as e:
        print(f"[WARN] Milvus health check failed: {e}")

    # 1) Chuẩn bị docs
    _ensure_sample_docs(DOCS_DIR)
    print(f"[QuickTest] Using docs: {DOCS_DIR.resolve()}")

    # 2) Cấu hình DManager
    cfg = DManagerConfig(
        docs_root_dir=str(DOCS_DIR),
        collection_base=COLLECTION_BASE,
        language=LANGUAGE,
        # metric/index type & params theo mặc định của DManagerConfig / IndexParams
        index_params=INDEX_PARAMS,
        shards_num=SHARDS,
        build_index=BUILD_INDEX,
    )

    # 3) Run pipeline
    manager = URagDManager(cfg)
    result = manager.run(return_augmented=True)

    # 4) Lưu augmented
    try:
        with SAVE_AUG.open("w", encoding="utf-8") as f:
            for row in result.get("augmented_chunks", []) or []:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[QuickTest] Augmented saved -> {SAVE_AUG.resolve()}")
    except Exception as e:
        print(f"[WARN] Save augmented failed: {e}")

    # 5) In summary
    print("\n=== Summary ===")
    _print_json(result.get("summary"))

    # 6) In stats nhanh (doc collection)
    try:
        if ia is None:
            ia = IndexingAgent(AgentConfig())
        doc_col = COLLECTION_BASE + "__doc"
        stats = ia.process(StatsReq(op="stats", collection=doc_col).model_dump())
        print("=== Doc Stats ===")
        _print_json(stats)
    except Exception as e:
        print(f"[WARN] stats failed: {e}")

    # 7) Cleanup collections if requested
    if DO_CLEANUP:
        try:
            if ia is None:
                ia = IndexingAgent(AgentConfig())
            base = COLLECTION_BASE
            cols = [base + "__doc", base + "__faq"] if getattr(ia.cfg, "dual_collections", True) else [base]
            for name in cols:
                if utility.has_collection(name):
                    utility.drop_collection(name)
                    print(f"[Cleanup] Dropped collection: {name}")
        except Exception as e:
            print(f"[WARN] cleanup failed: {e}")

    print("[QuickTest] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
