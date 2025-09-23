# src/managers/urag_f_manager.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
import math
import json
import time
import hashlib

# Dùng trực tiếp IndexingAgent để upsert
from src.indexing.indexing_agent import (
    IndexingAgent, AgentConfig, UpsertIndexReq, Item, IndexParams, Metric,
)

# ======================= Interfaces (Protocol) =======================

class IFaqGenerator(Protocol):
    def generate_roots(self, augmented_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]: ...
    def enrich_from_roots(self, roots: List[Dict[str, Any]], paraphrase_n: int = 5) -> List[Dict[str, Any]]: ...

class IEmbedder(Protocol):
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
    ts: int = 0

# ======================= Config =======================

@dataclass
class FManagerConfig:
    embed_field: str = "question"
    default_source: str = "faq_src"
    l2_normalize: bool = True

# ======================= Manager =======================

class FManager:
    def __init__(self, faq_generator: IFaqGenerator, embedder: IEmbedder, cfg: Optional[FManagerConfig] = None):
        self.gen = faq_generator
        self.embedder = embedder
        self.cfg = cfg or FManagerConfig()
        # --- MODIFIED THIS BLOCK TO USE THE NEW info() METHOD ---
        try:
            embedder_info = self.embedder.info()
            print(f"[FManager] Initialized with Embedder info: {json.dumps(embedder_info)}")
        except (AttributeError, TypeError):
            print("[FManager] Initialized. (Embedder has no info() method).")
        # --- END OF MODIFICATION ---

    # --------- public API ---------

    def generate(self, chunks: List[AugmentedChunk], paraphrase_n: int = 5) -> Dict[str, List[FAQItem]]:
        chunk_dicts = [c.model_dump() for c in chunks]
        roots_raw = self.gen.generate_roots(chunk_dicts)
        roots = [self._coerce_faq_item(r, is_paraphrase=False) for r in roots_raw]
        enriched_raw = self.gen.enrich_from_roots([r.model_dump() for r in roots], paraphrase_n=paraphrase_n)
        faqs = [self._coerce_faq_item(e) for e in enriched_raw]
        return {"roots": roots, "faqs": faqs}

    def embed(self, faqs: List[FAQItem]) -> List[FAQWithVec]:
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
                **f.model_dump(),
                vector=v,
                ts=ts_now
            ))
        return out

    def make_index_payload(self, collection_base: str, items_with_vec: List[FAQWithVec], *, metric: Metric = "COSINE") -> Dict[str, Any]:
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
        if not chunks:
            return {"summary": {"chunks_in": 0, "status": "skipped_empty"}, "resp": {}}
            
        built = self.generate(chunks, paraphrase_n=paraphrase_n)
        vec_items = self.embed(built["faqs"])
        if not vec_items:
            return {"summary": {"chunks_in": len(chunks), "faqs": len(built["faqs"]), "status": "skipped_no_vectors"}, "resp": {}}
            
        payload = self.make_index_payload(collection_base, vec_items, metric=metric)
        ia = IndexingAgent(AgentConfig())
        req = UpsertIndexReq(
            op="upsert", collection=collection_base, dim=len(vec_items[0].vector),
            metric_type=metric, items=[Item(**it) for it in payload["items"]],
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
        if not roots:
             return {"summary": {"roots_in": 0, "status": "skipped_empty"}, "resp": {}}

        enriched_raw = self.gen.enrich_from_roots(roots, paraphrase_n=paraphrase_n)
        faqs = [self._coerce_faq_item(e) for e in enriched_raw]
        vec_items = self.embed(faqs)
        if not vec_items:
            return {"summary": {"roots_in": len(roots), "faqs": len(faqs), "status": "skipped_no_vectors"}, "resp": {}}
            
        payload = self.make_index_payload(collection_base, vec_items, metric=metric)
        ia = IndexingAgent(AgentConfig())
        req = UpsertIndexReq(
            op="upsert", collection=collection_base, dim=len(vec_items[0].vector),
            metric_type=metric, items=[Item(**it) for it in payload["items"]],
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

    def _coerce_faq_item(self, r: Dict[str, Any], is_paraphrase: Optional[bool] = None) -> FAQItem:
        q = str(r.get("question", "")).strip()
        a = str(r.get("answer", "")).strip()
        can = r.get("canonical_id")
        src = r.get("source")
        meta = r.get("metadata") or {}
        ip = bool(r.get("is_paraphrase")) if is_paraphrase is None else bool(is_paraphrase)
        return FAQItem(question=q, answer=a, canonical_id=(str(can) if can else None), is_paraphrase=ip, source=(str(src) if src else None), metadata=(meta if isinstance(meta, dict) else {}))

    def _l2_normalize_inplace(self, v: List[float]) -> None:
        s = sum(x * x for x in v)
        if s <= 1e-9: return
        inv = 1.0 / math.sqrt(s)
        for i in range(len(v)): v[i] *= inv

    def _mk_id(self, it: FAQWithVec) -> str:
        base = f"{(it.canonical_id or 'c')[:32]}|{int(it.is_paraphrase)}|{(it.question or '')[:256]}"
        return hashlib.md5(base.encode("utf-8")).hexdigest()