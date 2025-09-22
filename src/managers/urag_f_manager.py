# src/managers/f_manager.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Protocol
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
import math
import json
import time


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

    def make_index_payload(self, collection_base: str, items_with_vec: List[FAQWithVec]) -> Dict[str, Any]:
        """
        Tạo payload (dict) tương thích IndexingAgent.upsert (dual-collection mode).
        Lưu ý: IndexingAgent sẽ tách sang __faq dựa trên field 'type' = 'faq'.
        """
        return {
            "op": "index",
            "collection": collection_base,
            "metric_type": "COSINE",
            "items": [
                {
                    "id": self._mk_id(it),      # hoặc để IndexingAgent tự upsert theo id do bạn tạo ngoài
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

    def _mk_id(self, it: FAQWithVec) -> str:
        """
        Nếu bạn muốn tự phát sinh id cho FAQ trước khi indexing. Ở đây mình tạo deterministic id nhẹ nhàng.
        Có thể thay bằng UUID hoặc id do canonical mapping quản lý.
        """
        base = f"{(it.canonical_id or 'c')[:12]}::{('p' if it.is_paraphrase else 'r')}::{hash(it.question) & 0xfffffff}"
        return base
