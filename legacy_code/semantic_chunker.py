# src/llm/URag_D/semantic_chunker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
import uuid

# LangChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings  # <-- quan trọng

# Thử 2 đường import để phù hợp layout dự án của bạn
from embedding.embedding_agent import EmbConfig, EmbedderAgent  # src/embedding/embedding_agent.py

# --- Adapter: biến EmbedderAgent thành LangChain Embeddings interface ---
class _AgentAsLCEmbeddings(Embeddings):  # <-- kế thừa Embeddings để Pylance hài lòng
    def __init__(self, agent: EmbedderAgent):
        self.agent = agent

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.agent._encode(texts)  # ndarray float32 (đã L2-norm nếu COSINE)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

@dataclass
class LCChunkerConfig:
    # chọn model theo ngôn ngữ
    language: str = "default"  # "vi" để ưu tiên tiếng Việt
    vi_model_name: str = "dangvantuan/vietnamese-embedding"
    en_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Tham số SemanticChunker
    buffer_size: int = 1
    add_start_index: bool = False
    breakpoint_threshold_type: Literal["percentile","standard_deviation","interquartile","gradient"] = "percentile"
    breakpoint_threshold_amount: Optional[float] = None  # ví dụ 95 khi dùng 'percentile'
    number_of_chunks: Optional[int] = None
    sentence_split_regex: str = r'(?<=[.?!])\s+'
    min_chunk_size: Optional[int] = None  # tối thiểu độ dài chunk (ký tự)

    # Chọn backend embeddings
    use_agent_embedder: bool = True
    agent_device: Optional[str] = None  # 'cpu' | 'cuda' | None

class SemanticChunkerLC:
    """Chunk văn bản theo ngữ nghĩa bằng LangChain SemanticChunker."""

    def __init__(self, cfg: Optional[LCChunkerConfig] = None):
        self.cfg = cfg or LCChunkerConfig()

        if self.cfg.use_agent_embedder:
            emb_cfg = EmbConfig(
                model_name=self.cfg.en_model_name,
                vi_model_name=self.cfg.vi_model_name,
                language=("vi" if self.cfg.language.lower() == "vi" else "default"),
                device=self.cfg.agent_device,
                normalize_for_cosine=True,
                metric="COSINE",
            )
            agent = EmbedderAgent(emb_cfg)
            embeddings: Embeddings = _AgentAsLCEmbeddings(agent)  # <-- annotate đúng type
        else:
            model_name = self.cfg.vi_model_name if self.cfg.language.lower() == "vi" else self.cfg.en_model_name
            embeddings: Embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=self.cfg.buffer_size,
            add_start_index=self.cfg.add_start_index,
            breakpoint_threshold_type=self.cfg.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.cfg.breakpoint_threshold_amount,
            number_of_chunks=self.cfg.number_of_chunks,
            sentence_split_regex=self.cfg.sentence_split_regex,
            min_chunk_size=self.cfg.min_chunk_size,
        )

    def chunk(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Trả về list[{doc_id, chunk_id, text}]"""
        if not text or not text.strip():
            return []
        parts = self.splitter.split_text(text)
        out: List[Dict[str, Any]] = []
        for p in parts:
            t = p.strip()
            if not t:
                continue
            out.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}__{uuid.uuid4().hex[:12]}",
                "text": t,
            })
        return out
