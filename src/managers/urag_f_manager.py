# -*- coding: utf-8 -*-
"""
FManager — FAQ pipeline orchestrator
===================================

Chức năng đúng theo yêu cầu:
1) augmentedChunks tài liệu -> FAQ agent -> tạo FAQ gốc -> FAQ agent -> enrich -> embed -> indexing -> faq db
2) FAQ (gốc) -> FAQ agent -> enrich -> embed -> indexing -> faq db

🆕 Bổ sung tiện ích/one-shot để dễ tích hợp với DManager:
- 🆕 `coerce_from_augmented(...)` — chuyển augmentedChunks (original/transformed) sang `AugmentedChunk`
- 🆕 `run_from_augmented(...)` — chạy trọn nhánh (1) và index vào Milvus qua `IndexingAgent`
- 🆕 `run_from_roots(...)` — chạy trọn nhánh (2) và index
- 🆕 `index(...)` — tiện ích upsert vào Milvus (dual collections) bằng `IndexingAgent`
- 🆕 `info()` — trả thông tin cấu hình + embedder
- 🔧 `make_index_payload(...)` — thêm tham số `metric` (không còn hardcode "COSINE")
- 🔧 `_mk_id(...)` — đổi sang id ổn định bằng MD5 (tránh phụ thuộc PYTHONHASHSEED)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
import math
import json
import time
import hashlib  # 🆕 stable id

# 🆕 Dùng trực tiếp IndexingAgent để upsert
from indexing.indexing_agent import (
    IndexingAgent, AgentConfig, UpsertIndexReq, Item, IndexParams, Metric,
)

# ======================= Interfaces (Protocol) =======================

class IFaqGenerator(Protocol):
    """
    Generator interface kỳ vọng:
    - generate_roots(chunks) -> List[Dict]: mỗi dict có {question, answer, canonical_id, ...}
    - enrich_from_roots(roots, paraphrase_n=...) -> List[Dict]:
        trả về list gồm cả root + paraphrases (mỗi item có question/answer; nếu là paraphrase nên có is_paraphrase=True)
    """
    def generate_roots(self, augmented_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...
    def enrich_from_roots(self, roots: List[Dict[str, Any]], paraphrase_n: int = 5) -> List[Dict[str, Any]]: ...


class IEmbedder(Protocol):
    """
    Embedder interface kỳ vọng:
    - info() -> Dict[str, Any]  (tùy chọn)
    - encode_texts(texts: List[str]) -> List[List[float]]  (float vectors, đã normalize hay chưa tùy bạn)
    """
    def encode_texts(self, texts: List[str]) -> List[List[float]]: ...
    def info(self) -> Dict[str, Any]: ...


# ======================= Models =======================

class AugmentedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id", "chunk_id", "text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must be non-empty")
        return v


class FAQItem(BaseModel):
    question: str
    answer: str
    canonical_id: Optional[str] = None
    is_paraphrase: bool = False
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("question", "answer")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question/answer must be non-empty")
        return v


class FAQWithVec(FAQItem):
    vector: List[float]
    ts: int = 0  # để khớp schema indexing nếu bạn muốn


# ======================= Config =======================

@dataclass
class FManagerConfig:
    # embed question hay answer? Thông thường embed QUESTION cho FAQ retrieval
    embed_field: str = "question"

    # default source to fill nếu thiếu
    default_source: str = "faq_src"

    # có L2-normalize vector không (hữu ích khi metric COSINE)
    l2_normalize: bool = True


# ======================= Manager =======================

class FManager:
    """
    Quản lý luồng FAQ:
    1) nhận augmented chunks (đã sinh từ Document pipeline)
    2) dùng FAQ generator -> tạo FAQ gốc (root), rồi enrich (paraphrase)
    3) embed các câu hỏi (hoặc theo config.embed_field)
    4) trả về payload đã kèm vector, sẵn sàng gửi qua indexing_agent (op=index/upsert, type='faq')
    """

    def __init__(self, faq_generator: IFaqGenerator, embedder: IEmbedder, cfg: Optional[FManagerConfig] = None):
        self.gen = faq_generator
        self.embedder = embedder
        self.cfg = cfg or FManagerConfig()

    # --------- public API ---------

    def generate(self, chunks: List[AugmentedChunk], paraphrase_n: int = 5) -> Dict[str, List[FAQItem]]:
        """
        Tạo FAQ: root -> enrich. Chỉ build dữ liệu (chưa embed).
        return: {"roots": [...], "faqs": [...]}
        """
        # Convert về dict để không phụ thuộc model lớp ngoài
        chunk_dicts = [c.model_dump() for c in chunks]

        roots_raw = self.gen.generate_roots(chunk_dicts)
        roots = [self._coerce_faq_item(r, is_paraphrase=False) for r in roots_raw]

        enriched_raw = self.gen.enrich_from_roots([r.model_dump() for r in roots], paraphrase_n=paraphrase_n)
        faqs = [self._coerce_faq_item(e) for e in enriched_raw]

        return {"roots": roots, "faqs": faqs}

    def embed(self, faqs: List[FAQItem]) -> List[FAQWithVec]:
        """
        Embed danh sách FAQ theo config.embed_field (mặc định là question).
        """
        if not faqs:
            return []
        texts = [(f.question if self.cfg.embed_field == "question" else f.answer) for f in faqs]
        vecs = self.embedder.encode_texts(texts)

        if self.cfg.l2_normalize:
            for v in vecs:
                self._l2_normalize_inplace(v)

        ts_now = int(time.time())
        out: List[FAQWithVec] = []
        for f, v in zip(faqs, vecs):
            out.append(FAQWithVec(
                question=f.question,
                answer=f.answer,
                canonical_id=f.canonical_id,
                is_paraphrase=f.is_paraphrase,
                source=f.source or self.cfg.default_source,
                metadata=f.metadata or {},
                vector=v,
                ts=ts_now
            ))
        return out

    # 🔧 thêm tham số metric (trước đây hardcode COSINE)
    def make_index_payload(self, collection_base: str, items_with_vec: List[FAQWithVec], *, metric: Metric = "COSINE") -> Dict[str, Any]:
        """
        Tạo payload (dict) tương thích IndexingAgent.upsert (dual-collection mode).
        Lưu ý: IndexingAgent sẽ tách sang __faq dựa trên field 'type' = 'faq'.
        """
        return {
            "op": "upsert",
            "collection": collection_base,
            "metric_type": metric,
            "items": [
                {
                    "id": self._mk_id(it),
                    "type": "faq",
                    "vector": it.vector,
                    "question": it.question,
                    "answer": it.answer,
                    "source": it.source or self.cfg.default_source,
                    "metadata": {
                        **(it.metadata or {}),
                        "canonical_id": it.canonical_id or "",
                        "is_paraphrase": it.is_paraphrase
                    },
                    "ts": it.ts,
                }
                for it in items_with_vec
            ]
        }

    # 🆕 Adapter: nhận augmentedChunks từ D manager (original/transformed)
    def coerce_from_augmented(self, augmented: List[Dict[str, Any]]) -> List[AugmentedChunk]:
        out: List[AugmentedChunk] = []
        for row in augmented:
            text = str(row.get("transformed") or row.get("original") or row.get("text") or "").strip()
            if not text:
                continue
            doc_id = str(row.get("doc_id") or "").strip() or "doc"
            chunk_id = str(row.get("chunk_id") or "").strip() or str(len(out))
            meta = row.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {"raw_meta": meta}
            out.append(AugmentedChunk(doc_id=doc_id, chunk_id=chunk_id, text=text, metadata=meta))
        return out

    # 🆕 One-shot: Nhánh (1) — từ augmentedChunks tới index
    def run_from_augmented(
        self,
        augmented: List[Dict[str, Any]],
        *,
        collection_base: str,
        paraphrase_n: int = 5,
        metric: Metric = "COSINE",
        index_params: Optional[IndexParams] = None,
        shards: int = 2,
        build_index: bool = True,
    ) -> Dict[str, Any]:
        chunks = self.coerce_from_augmented(augmented)
        built = self.generate(chunks, paraphrase_n=paraphrase_n)
        vec_items = self.embed(built["faqs"])  # embed all enriched (gồm cả roots nếu generator trả về)
        payload = self.make_index_payload(collection_base, vec_items, metric=metric)
        # Upsert bằng IndexingAgent
        ia = IndexingAgent(AgentConfig())
        req = UpsertIndexReq(
            op="upsert", collection=collection_base, dim=len(vec_items[0].vector) if vec_items else None,
            metric_type=metric, items=[Item(**it) if not isinstance(it, Item) else it for it in payload["items"]],
            shards_num=shards, index_params=index_params, build_index=build_index,
        )
        resp = ia.process(req.model_dump())
        return {
            "summary": {
                "chunks_in": len(chunks),
                "roots": len(built["roots"]),
                "faqs": len(built["faqs"]),
                "embedded": len(vec_items),
            },
            "resp": resp,
        }

    # 🆕 One-shot: Nhánh (2) — từ FAQ gốc tới index
    def run_from_roots(
        self,
        roots: List[Dict[str, Any]],
        *,
        collection_base: str,
        paraphrase_n: int = 5,
        metric: Metric = "COSINE",
        index_params: Optional[IndexParams] = None,
        shards: int = 2,
        build_index: bool = True,
    ) -> Dict[str, Any]:
        enriched_raw = self.gen.enrich_from_roots(roots, paraphrase_n=paraphrase_n)
        faqs = [self._coerce_faq_item(e) for e in enriched_raw]
        vec_items = self.embed(faqs)
        payload = self.make_index_payload(collection_base, vec_items, metric=metric)
        ia = IndexingAgent(AgentConfig())
        req = UpsertIndexReq(
            op="upsert", collection=collection_base, dim=len(vec_items[0].vector) if vec_items else None,
            metric_type=metric, items=[Item(**it) if not isinstance(it, Item) else it for it in payload["items"]],
            shards_num=shards, index_params=index_params, build_index=build_index,
        )
        resp = ia.process(req.model_dump())
        return {
            "summary": {
                "roots_in": len(roots),
                "faqs": len(faqs),
                "embedded": len(vec_items),
            },
            "resp": resp,
        }

    # 🆕 Tiện ích index trực tiếp (đã có vector)
    def index(
        self,
        items_with_vec: List[FAQWithVec],
        *,
        collection_base: str,
        metric: Metric = "COSINE",
        index_params: Optional[IndexParams] = None,
        shards: int = 2,
        build_index: bool = True,
    ) -> Dict[str, Any]:
        payload = self.make_index_payload(collection_base, items_with_vec, metric=metric)
        ia = IndexingAgent(AgentConfig())
        req = UpsertIndexReq(
            op="upsert", collection=collection_base, dim=len(items_with_vec[0].vector) if items_with_vec else None,
            metric_type=metric, items=[Item(**it) if not isinstance(it, Item) else it for it in payload["items"]],
            shards_num=shards, index_params=index_params, build_index=build_index,
        )
        return ia.process(req.model_dump())

    # 🆕 Expose thông tin cấu hình & embedder
    def info(self) -> Dict[str, Any]:
        return {
            "cfg": {
                "embed_field": self.cfg.embed_field,
                "default_source": self.cfg.default_source,
                "l2_normalize": self.cfg.l2_normalize,
            },
            "embedder": (self.embedder.info() if hasattr(self.embedder, "info") else {}),
        }

    # --------- helpers ---------

    def _coerce_faq_item(self, r: Dict[str, Any], is_paraphrase: Optional[bool] = None) -> FAQItem:
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        can = r.get("canonical_id")
        src = r.get("source")
        meta = r.get("metadata") or {}

        ip = bool(r.get("is_paraphrase")) if is_paraphrase is None else bool(is_paraphrase)

        return FAQItem(
            question=q,
            answer=a,
            canonical_id=(str(can) if can else None),
            is_paraphrase=ip,
            source=(str(src) if src else None),
            metadata=(meta if isinstance(meta, dict) else {})
        )

    def _l2_normalize_inplace(self, v: List[float]) -> None:
        s = 0.0
        for x in v:
            s += x * x
        if s <= 0:
            return
        inv = 1.0 / math.sqrt(s)
        for i in range(len(v)):
            v[i] *= inv

    # 🔧 dùng MD5 để có id ổn định (deterministic)
    def _mk_id(self, it: FAQWithVec) -> str:
        base = f"{(it.canonical_id or 'c')[:32]}|{int(it.is_paraphrase)}|{(it.question or '')[:256]}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()
