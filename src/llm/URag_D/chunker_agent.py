# -*- coding: utf-8 -*-
"""
SemanticChunkerAgent — Pydantic Agent wrapper cho LangChain SemanticChunker

- Cho phép chọn backend embedding:
    + Dùng EmbedderAgent nội bộ của bạn (khuyến nghị, đồng bộ metric COSINE + normalize)
    + Hoặc dùng HuggingFaceEmbeddings (LangChain community)

- API:
    cfg = SemanticChunkerAgentConfig(language="vi", use_agent_embedder=True)
    agent = SemanticChunkerAgent(cfg)
    resp = agent.chunk(SemanticChunkerRequest(text="...", doc_id="doc_123"))
    print([c.text for c in resp.chunks])

Phụ thuộc:
- langchain_experimental.text_splitter.SemanticChunker
- langchain_community.embeddings.HuggingFaceEmbeddings
- pydantic v2
- src/embedding/embedding_agent.EmbedderAgent, EmbConfig  (dự án của bạn)

Ghi chú:
- Nếu bạn cần giữ y hệt output cũ: list[{doc_id, chunk_id, text}], hãy dùng
  resp.to_legacy_list().
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
import uuid
import re

from pydantic import BaseModel, Field, field_validator

# LangChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings  

from src.embedding.embedding_agent import EmbConfig, EmbedderAgent
# --- Adapter: biến EmbedderAgent thành LangChain Embeddings interface ---
class _AgentAsLCEmbeddings(Embeddings):
    def __init__(self, agent: EmbedderAgent):
        self.agent = agent

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # agent._encode: ndarray float32 (đã L2-norm nếu COSINE)
        vecs = self.agent._encode(texts)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ===================== Pydantic Models =====================

class SemanticChunkerAgentConfig(BaseModel):
    # Chọn model theo ngôn ngữ
    language: Literal["default", "vi", "en"] = "default"
    vi_model_name: str = "dangvantuan/vietnamese-embedding"
    en_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Tham số SemanticChunker
    buffer_size: int = 1
    add_start_index: bool = False
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = "percentile"
    breakpoint_threshold_amount: Optional[float] = None  # ví dụ 95 khi dùng 'percentile'
    number_of_chunks: Optional[int] = None
    sentence_split_regex: str = r'(?<=[.?!])\s+'
    min_chunk_size: Optional[int] = None  # độ dài ký tự tối thiểu cho mỗi chunk (None = tắt)

    # Backend embeddings
    use_agent_embedder: bool = True
    agent_device: Optional[str] = None  # 'cpu' | 'cuda' | None

    # Normalize + metric cho Agent nội bộ
    normalize_for_cosine: bool = True
    metric: Literal["COSINE", "IP", "L2"] = "COSINE"

    # Guardrails nhẹ
    @field_validator("buffer_size")
    @classmethod
    def _buf_pos(cls, v: int) -> int:
        if v < 0:
            raise ValueError("buffer_size must be >= 0")
        return v

    @field_validator("breakpoint_threshold_amount")
    @classmethod
    def _percentile_range(cls, v: Optional[float], info) -> Optional[float]:
        # Chỉ kiểm tra khi dùng percentile
        values = info.data
        btype = values.get("breakpoint_threshold_type", "percentile")
        if btype == "percentile" and v is not None:
            if not (0.0 <= v <= 100.0):
                raise ValueError("breakpoint_threshold_amount must be in [0, 100] when type='percentile'")
        return v

    @field_validator("sentence_split_regex")
    @classmethod
    def _valid_regex(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid sentence_split_regex: {e}")
        return v


class SemanticChunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str


class SemanticChunkerRequest(BaseModel):
    text: str = Field(..., description="Đoạn văn bản đầu vào để chunk")
    doc_id: str = Field(..., description="Mã tài liệu gốc")


class SemanticChunkerResponse(BaseModel):
    chunks: List[SemanticChunk]
    meta: Dict[str, Any] = Field(default_factory=dict)

    # Helper để trả về format cũ list[{doc_id, chunk_id, text}]
    def to_legacy_list(self) -> List[Dict[str, Any]]:
        return [c.model_dump() for c in self.chunks]


# ===================== Agent =====================

class SemanticChunkerAgent:
    """
    Agent đóng gói LangChain SemanticChunker dưới dạng Pydantic-friendly API.
    """
    def __init__(self, cfg: Optional[SemanticChunkerAgentConfig] = None):
        self.cfg = cfg or SemanticChunkerAgentConfig()
        self._embeddings: Embeddings = self._build_embeddings()
        self._splitter = SemanticChunker(
            embeddings=self._embeddings,
            buffer_size=self.cfg.buffer_size,
            add_start_index=self.cfg.add_start_index,
            breakpoint_threshold_type=self.cfg.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.cfg.breakpoint_threshold_amount,
            number_of_chunks=self.cfg.number_of_chunks,
            sentence_split_regex=self.cfg.sentence_split_regex,
            min_chunk_size=self.cfg.min_chunk_size,
        )

    # ---------- Private helpers ----------
    def _choose_model_name(self) -> str:
        if self.cfg.language == "vi":
            return self.cfg.vi_model_name
        elif self.cfg.language == "en":
            return self.cfg.en_model_name
        else:
            # "default": ưu tiên EN làm base phổ thông
            return self.cfg.en_model_name

    def _build_embeddings(self) -> Embeddings:
        if self.cfg.use_agent_embedder:
            emb_cfg = EmbConfig(
                model_name=self._choose_model_name(),
                vi_model_name=self.cfg.vi_model_name,
                language=("vi" if self.cfg.language == "vi" else "default"),
                device=self.cfg.agent_device,
                normalize_for_cosine=self.cfg.normalize_for_cosine,
                metric=self.cfg.metric,
            )
            agent = EmbedderAgent(emb_cfg)
            return _AgentAsLCEmbeddings(agent)
        else:
            model_name = self._choose_model_name()
            return HuggingFaceEmbeddings(model_name=model_name)

    # ---------- Public API ----------
    def chunk(self, req: SemanticChunkerRequest) -> SemanticChunkerResponse:
        """
        Trả về danh sách chunk (doc_id, chunk_id, text).
        """
        text = (req.text or "").strip()
        if not text:
            return SemanticChunkerResponse(chunks=[], meta={"reason": "empty_input"})

        parts = self._splitter.split_text(text)
        chunks: List[SemanticChunk] = []
        for p in parts:
            t = p.strip()
            if not t:
                continue
            chunks.append(
                SemanticChunk(
                    doc_id=req.doc_id,
                    chunk_id=f"{req.doc_id}__{uuid.uuid4().hex[:12]}",
                    text=t,
                )
            )

        return SemanticChunkerResponse(
            chunks=chunks,
            meta={
                "language": self.cfg.language,
                "backend": "agent_embedder" if self.cfg.use_agent_embedder else "hf_embeddings",
                "buffer_size": self.cfg.buffer_size,
                "breakpoint_type": self.cfg.breakpoint_threshold_type,
                "number_of_chunks_cfg": self.cfg.number_of_chunks,
                "min_chunk_size": self.cfg.min_chunk_size,
            },
        )

    # Nếu bạn muốn API "thuần" như hàm cũ:
    def chunk_legacy(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        resp = self.chunk(SemanticChunkerRequest(text=text, doc_id=doc_id))
        return resp.to_legacy_list()
