# -*- coding: utf-8 -*-
"""
URAG D Manager — Orchestrator for Documents ➜ (Chunk ➜ LLM Augment ➜ Embed ➜ Index)
=================================================================================
Vị trí file (khớp tree của bạn): `src/managers/urag_d_manager.py`

Pipeline:
  Document -> SemanticChunker -> chunks
  chunks -> LLM (textGenerate) -> augmentedChunks
  augmentedChunks -> Embed -> Index (Milvus) qua `indexing.indexing_agent.IndexingAgent`
  augmentedChunks -> có thể trả ra cho MetaManager để chuyển tiếp sang FManager

Phụ thuộc sẵn có theo tree:
- from llm.URag_D.document_loader import DocLoaderLC, DocLoaderConfig
- from llm.URag_D.semantic_chunker import SemanticChunkerLC, LCChunkerConfig
- from llm.URag_D import textGenerate as tg
- from embedding.embedding_agent import EmbConfig, EmbedderAgent, Metric as EmbMetric
- from indexing.indexing_agent import (
    IndexingAgent, AgentConfig, UpsertIndexReq, CreateCollectionReq, Item, IndexParams,
    Metric, IndexType,
)
      IndexingAgent, AgentConfig, UpsertIndexReq, CreateCollectionReq, Item, IndexParams
  )
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast
import json
import time
import hashlib

# ---------- Imports khớp cấu trúc dự án ----------
from src.llm.URag_D.document_loader import DocLoaderLC, DocLoaderConfig
from src.llm.URag_D.semantic_chunker import SemanticChunkerLC, LCChunkerConfig
from src.llm.URag_D import textGenerate as tg
from src.embedding.embedding_agent import EmbConfig, EmbedderAgent, Metric as EmbMetric
from src.indexing.indexing_agent import (
    IndexingAgent, AgentConfig, UpsertIndexReq, CreateCollectionReq, Item, IndexParams,
    Metric, IndexType,
)


# ------------------- Cấu hình -------------------
@dataclass
class DManagerConfig:
    # Nguồn tài liệu
    docs_root_dir: Optional[str] = None

    # Tên base collection (IndexingAgent sẽ tự tách __doc / __faq nếu bật dual)
    collection_base: str = "ura_rag"

    # Ngôn ngữ & embedding
    language: str = "default"  # "vi" | "default"
    vi_model_name: str = "dangvantuan/vietnamese-embedding"
    en_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_device: Optional[str] = None
    normalize_for_cosine: bool = True

    # Chunker
    chunker: Optional[LCChunkerConfig] = None

    # Milvus / Index params
    metric_type: Metric = "COSINE"  # Literal| "IP" | "L2"
    index_type: IndexType = "HNSW"   # Literal| "IVF_FLAT" | "IVF_SQ8" | "IVF_PQ"
    index_params: Optional[Dict[str, Any]] = None  # ví dụ {"M":32, "efConstruction":200}
    shards_num: int = 2
    build_index: bool = True

    # Metadata mặc định
    default_source_doc: str = "doc_src"


class URagDManager:
    """
    Điều phối pipeline Documents-only theo yêu cầu của bạn.
    """

    def __init__(self, cfg: DManagerConfig):
        self.cfg = cfg
        # Loader & Chunker
        self.loader = DocLoaderLC(DocLoaderConfig())
        self.chunker = SemanticChunkerLC(
            cfg.chunker or LCChunkerConfig(
                language=cfg.language,
                vi_model_name=cfg.vi_model_name,
                en_model_name=cfg.en_model_name,
            )
        )
        # Embedder
        self.embedder = EmbedderAgent(
            EmbConfig(
                model_name=cfg.en_model_name,
                vi_model_name=cfg.vi_model_name,
                language=("vi" if cfg.language.lower() == "vi" else "default"),
                device=cfg.emb_device,
                normalize_for_cosine=cfg.normalize_for_cosine,
                metric=cast(EmbMetric, self.cfg.metric_type),
            )
        )
        # Indexing agent (Milvus)
        self.indexer = IndexingAgent(AgentConfig())

    # ------------------- Public API -------------------
    def run(self, docs_root_dir: Optional[str] = None, return_augmented: bool = True) -> Dict[str, Any]:
        root = docs_root_dir or self.cfg.docs_root_dir
        assert root, "docs_root_dir chưa được cấu hình"

        # 1) Load tài liệu -> augmented inputs (doc_id, text, metadata)
        docs = self.loader.load_normal_docs(root)
        aug_inputs = self.loader.to_augmented_inputs(docs, default_source=self.cfg.default_source_doc)

        # 2) Chunk theo ngữ nghĩa
        chunks: List[Dict[str, Any]] = []
        for item in aug_inputs:
            doc_id = item["doc_id"]
            text = item["text"]
            parts = self.chunker.chunk(text, doc_id)
            for p in parts:
                p["source"] = item.get("metadata", {}).get("source", self.cfg.default_source_doc)
                p["metadata"] = item.get("metadata", {})
            chunks.extend(parts)

        # 3) LLM augment từng chunk
        augmented_chunks = self._augment_chunks(chunks)

        # 4) Embed và Index
        ts_now = int(time.time())
        texts = [ac.get("transformed") or ac.get("original") or "" for ac in augmented_chunks]
        vectors = self._encode_texts(texts)
        dim = len(vectors[0]) if vectors else 0

        # 4.1) Đảm bảo collection tồn tại
        self._ensure_collections(dim)

        # 4.2) Build items và upsert
        items: List[Item] = []
        for ac, vec in zip(augmented_chunks, vectors):
            uid = self._make_id(ac)
            items.append(
                Item(
                    id=uid,
                    type="doc",
                    vector=[float(x) for x in vec],
                    text=ac.get("transformed") or ac.get("original") or "",
                    source=ac.get("source", self.cfg.default_source_doc),
                    metadata=ac.get("metadata", {}),
                    ts=ts_now,
                )
            )

        upsert = UpsertIndexReq(
            op="upsert",
            collection=self.cfg.collection_base,
            dim=dim,
            metric_type=self.cfg.metric_type,
            items=items,
            shards_num=self.cfg.shards_num,
            index_params=(IndexParams(index_type=self.cfg.index_type, metric_type=self.cfg.metric_type, params=(self.cfg.index_params or { }))),
            build_index=self.cfg.build_index,
        )
        resp = self.indexer.process(upsert.model_dump())

        summary = {
            "docs_loaded": len(aug_inputs),
            "chunks": len(chunks),
            "augmented_chunks": len(augmented_chunks),
            "doc_embedded": len(items),
            "dim": dim,
            "indexing_status": resp.get("status"),
        }
        out: Dict[str, Any] = {"summary": summary}
        if return_augmented:
            out["augmented_chunks"] = augmented_chunks
        return out

    # ------------------- Helpers -------------------
    def _augment_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        raw_texts = [c.get("text", "") for c in chunks]
        # Chia 2 worker nếu có; nếu chỉ có 1, vẫn OK
        parts: List[Tuple[List[str], Any]] = []
        mid = len(raw_texts) // 2
        if hasattr(tg, "worker1"):
            parts.append((raw_texts[:mid], getattr(tg, "worker1")))
        if hasattr(tg, "worker2"):
            parts.append((raw_texts[mid:], getattr(tg, "worker2")))
        if not parts:
            # Không có worker -> giữ nguyên làm augmented tối thiểu
            return [
                {
                    "doc_id": c.get("doc_id"),
                    "chunk_id": c.get("chunk_id"),
                    "original": c.get("text", ""),
                    "transformed": c.get("text", ""),
                    "source": c.get("source"),
                    "metadata": c.get("metadata", {}),
                }
                for c in chunks
            ]

        augmented: List[Dict[str, Any]] = []
        cursor = 0
        for texts_slice, worker in parts:
            if not texts_slice:
                continue
            deps = getattr(tg, "MyDeps")(listChunks=texts_slice)
            results = worker.run_sync(deps=deps)
            cleaned = getattr(tg, "checking_output")(results.output)
            try:
                parsed = json.loads(cleaned)  # list[{original, transformed}]
            except Exception:
                parsed = []

            for _ in texts_slice:
                base = chunks[cursor]
                cursor += 1
                row = parsed.pop(0) if parsed else {}
                original = str(row.get("original", base.get("text", ""))).strip() if isinstance(row, dict) else base.get("text", "")
                transformed = str(row.get("transformed", original)).strip() if isinstance(row, dict) else original
                augmented.append(
                    {
                        "doc_id": base.get("doc_id"),
                        "chunk_id": base.get("chunk_id"),
                        "original": original,
                        "transformed": transformed,
                        "source": base.get("source"),
                        "metadata": base.get("metadata", {}),
                    }
                )

        # Bổ sung nếu thiếu vì parsing lỗi
        while len(augmented) < len(chunks):
            base = chunks[len(augmented)]
            augmented.append(
                {
                    "doc_id": base.get("doc_id"),
                    "chunk_id": base.get("chunk_id"),
                    "original": base.get("text", ""),
                    "transformed": base.get("text", ""),
                    "source": base.get("source"),
                    "metadata": base.get("metadata", {}),
                }
            )
        return augmented

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Ưu tiên API encode(list[str]) của EmbedderAgent
        if hasattr(self.embedder, "encode"):
            vectors = self.embedder.encode(texts)  # type: ignore[attr-defined]
            if not vectors or not isinstance(vectors[0], (list, tuple)):
                raise RuntimeError("EmbedderAgent.encode trả về không hợp lệ")
            return [list(map(float, v)) for v in vectors]
        # Fallback: nếu có embed_docs, bóc vector ra
        if hasattr(self.embedder, "embed_docs"):
            payload = [
                {"id": f"tmp-{i}", "text": t, "type": "doc", "metadata": {}, "source": self.cfg.default_source_doc}
                for i, t in enumerate(texts)
            ]
            out = self.embedder.embed_docs(payload, default_source=self.cfg.default_source_doc, ts=int(time.time()))  # type: ignore[attr-defined]
            return [list(map(float, row.get("vector", []))) for row in out]
        raise RuntimeError("Không tìm thấy API encode/embed_docs trong EmbedderAgent")

    def _ensure_collections(self, dim: int) -> None:
        """Gọi create_collection một lần để đảm bảo schema/index tồn tại (dual collections)."""
        req = CreateCollectionReq(
            op="create_collection",
            collection=self.cfg.collection_base,
            dim=dim,
            metric_type=self.cfg.metric_type,
            shards_num=self.cfg.shards_num,
            index_params=IndexParams(index_type=self.cfg.index_type, metric_type=self.cfg.metric_type, params=(self.cfg.index_params or {})),
        )
        self.indexer.process(req.model_dump())

    @staticmethod
    def _make_id(ac: Dict[str, Any]) -> str:
        """Sinh id ổn định từ (doc_id, chunk_id, text) để tiện upsert."""
        base = f"{ac.get('doc_id','')}::{ac.get('chunk_id','')}::{(ac.get('transformed') or ac.get('original') or '')[:64]}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()
