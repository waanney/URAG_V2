# -*- coding: utf-8 -*-
"""
SearchAgent (Pydantic) — URAG Inference Pipeline (Kernel-based LLM)
Tier-1: FAQ -> trả answer trực tiếp nếu vượt ngưỡng
Tier-2: Doc  -> xây prompt từ doc, gửi LLM (qua KERNEL) -> trả lời
Tier-3: Fallback nếu không có gì liên quan

Phụ thuộc:
- /src/embedding/embedding_agent.py  (EmbedderAgent)
- /src/indexing/indexing_agent.py    (IndexingAgent, SearchReq)
- /src/llm/llm_kernel.py             (KERNEL: UI/ENV chọn provider/model)

Ghi chú:
- LLM được lấy từ KERNEL.get_active_model(); provider/model do UI/ENV quyết định.
- Không phụ thuộc GEMINI SDK trực tiếp ở đây.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal, Tuple, Sequence
import time

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

# === Agents của bạn ===
from src.embedding.embedding_agent import EmbedderAgent, EmbConfig
from src.indexing.indexing_agent import IndexingAgent, AgentConfig, SearchReq
from src.llm.llm_kernel import KERNEL

from rapidfuzz import fuzz

import torch
from sentence_transformers import CrossEncoder
# ======================= Search agent (Pydantic) =======================

class CrossEncoderReranker:
    """
    Reranker cross-encoder: nhận (query, candidate_text) -> score ∈ ℝ (càng cao càng liên quan).
    Hỗ trợ batch, tự chọn device (cuda/mps/cpu).
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512):
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
        self.model = CrossEncoder(model_name, max_length=max_length, device=self.device)

    @torch.inference_mode()
    def score(self, query: str, cands: Sequence[str], batch_size: int = 32) -> List[float]:
        """
        Trả về list score (float) tương ứng với từng candidate trong cands.
        """
        if not cands:
            return []
        pairs: List[Tuple[str, str]] = [(query, c) for c in cands]
        scores = self.model.predict(pairs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        return [float(s) for s in scores]

def _entity_overlap(a: str, b: str, entities: List[str]) -> int:
    aL, bL = a.lower(), b.lower()
    return sum(1 for e in entities if e and e.lower() in aL and e.lower() in bL)

def _extract_simple_entities(q: str) -> List[str]:
    # Nhanh-gọn: lấy chuỗi có chữ cái đầu viết hoa (tên riêng), hoặc tự liệt kê từ khóa tay
    toks = [t.strip(",.?;:()[]") for t in q.split()]
    cands = [t for t in toks if t[:1].isupper() and len(t) > 1]
    # ví dụ: gom lại cặp tên ghép 2 từ
    join2 = []
    i = 0
    while i < len(cands)-1:
        if cands[i][0].isupper() and cands[i+1][0].isupper():
            join2.append(cands[i] + " " + cands[i+1])
            i += 2
        else:
            i += 1
    return list(set(cands + join2))

def _accept_faq(
    query: str,
    cand_q: str,
    cos_sim: float,
    *,
    require_entity: bool = True,
    min_lexical: int = 60,
    rerank_score: Optional[float] = None,
    rerank_min: float = 0.50,
    must_entities: Optional[List[str]] = None,
) -> bool:
    # 1) cosine gate
    if cos_sim < 0.70:  # <-- tăng ngưỡng an toàn
        return False
    # 2) lexical sanity via partial_ratio (rẻ, tránh paraphrase lệch nghĩa)
    if fuzz.partial_ratio(query, cand_q) < min_lexical:
        return False
    # 3) entity overlap (nếu query có tên riêng)
    if require_entity and must_entities:
        if _entity_overlap(query, cand_q, must_entities) == 0:
            return False
    # 4) cross-encoder rerank (nếu có)
    if rerank_score is not None and rerank_score < rerank_min:
        return False
    return True

Metric = Literal["COSINE", "IP", "L2"]

class SearchConfig(BaseModel):
    # Collection cơ sở (đã lập chỉ mục): tên gốc, suffix do IndexingAgent cấu hình
    collection_base: str

    # Retrieval
    faq_top_k: int = 5
    doc_top_k: int = 5

    # Ngưỡng similarity (0..1) sau chuẩn hoá từ distance:
    tFAQ: float = 0.70
    tDOC: float = 0.60

    metric: Metric = "COSINE"

    # Milvus search params (tuỳ index HNSW/IVF)
    faq_search_params: Optional[Dict[str, Any]] = None
    doc_search_params: Optional[Dict[str, Any]] = None

    # LLM
    max_ctx_docs: int = 4
    disclaimer: Optional[str] = "Lưu ý: Câu trả lời được tổng hợp từ tài liệu hệ thống."
    llm_system_instruction: str = (
        "Bạn là một trợ lý AI chuyên phân tích và tổng hợp thông tin từ các tài liệu được cung cấp trong thẻ <CONTEXT>."
        "Nhiệm vụ của bạn là trả lời câu hỏi trong thẻ <QUESTION> một cách chính xác, tuân thủ nghiêm ngặt các quy tắc sau:\n"
        "1. **Phân tích & Đối chiếu:** Chỉ được phép sử dụng thông tin nằm trong <CONTEXT>. Tuyệt đối không dùng kiến thức bên ngoài.\n"
        "2. **Tổng hợp & Trích dẫn:** Luôn trích dẫn nguồn của thông tin bằng cách dùng định dạng `` ngay sau thông tin đó. Ví dụ: 'Tuổi thọ trung bình của khu vực Hà Nội cũ là 79 tuổi .'\n"
        "3. **Xử lý thiếu thông tin:** Nếu không có tài liệu nào trong <CONTEXT> chứa câu trả lời, hãy trả lời dứt khoát: 'Không có thông tin trong tài liệu cung cấp để trả lời câu hỏi này.'"
    )

    accept_require_entity: bool = True
    accept_min_lexical: int = 60  # rapidfuzz.partial_ratio
    reranker_name: Optional[str] = None  # "bge-reranker-base" / "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_min_score: float = 0.50
    prefer_source_order: Optional[str] = None  # ["doc_qa_generated", "paraphrase_generated"]

    @field_validator("tFAQ", "tDOC")
    @classmethod
    def _clamp_thresholds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Thresholds must be in [0,1]")
        return v

class Hit(BaseModel):
    id: str
    score: float
    similarity: float
    source: Optional[str] = None
    text: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchTrace(BaseModel):
    latency_ms: int
    params: Dict[str, Any]
    faq_hits: Optional[List[Hit]] = None
    faq_best: Optional[Hit] = None
    doc_hits: Optional[List[Hit]] = None
    ctx_used: Optional[List[Hit]] = None
    prompt: Optional[str] = None

class SearchResponse(BaseModel):
    tier: Literal["faq", "doc", "none"]
    final_answer: str
    trace: SearchTrace

class SearchAgent:
    def __init__(
        self,
        s_cfg: SearchConfig,
        e_cfg: Optional[EmbConfig] = None,
        i_cfg: Optional[AgentConfig] = None,
        embedder: Optional[EmbedderAgent] = None,
        indexer: Optional[IndexingAgent] = None,
    ):
        self.s_cfg = s_cfg
        self.embedder = embedder or EmbedderAgent(e_cfg or EmbConfig())
        self.indexer = indexer or IndexingAgent(i_cfg or AgentConfig())

        # Lấy model do UI/ENV chọn qua KERNEL, tạo pydantic-ai Agent với system prompt cố định
        llm_model = KERNEL.get_active_model(model_name="gpt-4o-mini")
        self._llm_agent = Agent(llm_model, system_prompt=self.s_cfg.llm_system_instruction)

        # khớp suffix từ IndexingAgent để tránh lệch tên
        self.doc_collection = f"{self.s_cfg.collection_base}{self.indexer.cfg.doc_suffix}"
        self.faq_collection = f"{self.s_cfg.collection_base}{self.indexer.cfg.faq_suffix}"

        self._reranker = None
        if self.s_cfg.reranker_name:  # ví dụ: "BAAI/bge-reranker-base" hoặc "cross-encoder/ms-marco-MiniLM-L-6-v2"
            try:
                self._reranker = CrossEncoderReranker(self.s_cfg.reranker_name)
            except Exception as e:
                # Không chặn pipeline nếu load thất bại
                print(f"[Reranker] Load failed ({self.s_cfg.reranker_name}): {e}")
                self._reranker = None

    # ---------------- public API ----------------
    def answer(self, query: str) -> SearchResponse:
        t0 = time.time()
        self._last_query_text = query
        # 1) embed query
        q_vec = self.embedder.encode([query])[0]  # List[float]

        # 2) FAQ
        faq_hits, faq_best = self._search_faq(q_vec)
        if faq_best and faq_best.similarity >= self.s_cfg.tFAQ:
            trace = SearchTrace(
                latency_ms=int((time.time() - t0) * 1000),
                params={"tFAQ": self.s_cfg.tFAQ, "faq_top_k": self.s_cfg.faq_top_k, "metric": self.s_cfg.metric},
                faq_hits=faq_hits, faq_best=faq_best,
            )
            return SearchResponse(tier="faq", final_answer=faq_best.answer or "", trace=trace)

        # 3) DOC + LLM
        doc_hits = self._search_doc(q_vec)
        doc_rel = [h for h in doc_hits if h.similarity >= self.s_cfg.tDOC]
        if doc_rel:
            ctx = doc_rel[: self.s_cfg.max_ctx_docs]
            prompt = self._build_prompt(query, ctx)
            llm_answer = self._llm_agent.run_sync(prompt).output.strip()
            if self.s_cfg.disclaimer and llm_answer:
                llm_answer += f"\n\n{self.s_cfg.disclaimer}"
            trace = SearchTrace(
                latency_ms=int((time.time() - t0) * 1000),
                params={
                    "tDOC": self.s_cfg.tDOC,
                    "doc_top_k": self.s_cfg.doc_top_k,
                    "metric": self.s_cfg.metric,
                    "max_ctx_docs": self.s_cfg.max_ctx_docs,
                },
                doc_hits=doc_hits, ctx_used=ctx, prompt=prompt
            )
            return SearchResponse(tier="doc", final_answer=llm_answer or "", trace=trace)

        # 4) Fallback
        trace = SearchTrace(
            latency_ms=int((time.time() - t0) * 1000),
            params={"tFAQ": self.s_cfg.tFAQ, "tDOC": self.s_cfg.tDOC},
            faq_hits=faq_hits, doc_hits=doc_hits
        )
        return SearchResponse(
            tier="none",
            final_answer="Xin lỗi, tôi không tìm thấy thông tin liên quan trong kho dữ liệu.",
            trace=trace
        )

    # ---------------- internals ----------------
    def _search_faq(self, q_vec: List[float]) -> Tuple[List[Hit], Optional[Hit]]:
        req = SearchReq(
            op="search",
            collection=self.faq_collection,
            search_vector=q_vec,
            top_k=self.s_cfg.faq_top_k,
            metric_type=self.s_cfg.metric,
            output_fields=["id", "question", "answer", "source", "metadata"],
            search_params=self.s_cfg.faq_search_params or {},
        )
        res = self.indexer.process(req)
        if res.get("status") != "ok":
            return [], None
        hits = self._pp_hits(res["data"].get("results", []), is_faq=True)
        if self._reranker and hits:
            # Ghép text ứng viên: ưu tiên question; có thể nối answer để tăng tín hiệu
            cand_texts = []
            for h in hits:
                q = h.question or ""
                a = h.answer or ""
                # Bạn có thể thay đổi template bên dưới:
                cand_texts.append(f"Q: {q}\nA: {a}".strip())

            r_scores = self._reranker.score(self._last_query_text, cand_texts, batch_size=32) \
                       if hasattr(self, "_last_query_text") else \
                       self._reranker.score("", cand_texts, batch_size=32)

            # Lưu điểm vào metadata và áp gate
            for h, rs in zip(hits, r_scores):
                if h.metadata is None:
                    h.metadata = {}
                h.metadata["rerank_score"] = rs

            # Lọc theo _accept_faq (cosine gate + lexical + entity + rerank_min)
            ents = _extract_simple_entities(self._last_query_text) if hasattr(self, "_last_query_text") else []
            filtered = []
            for h in hits:
                ok = _accept_faq(
                    self._last_query_text,
                    h.question or "",
                    h.similarity,
                    require_entity=self.s_cfg.accept_require_entity,
                    min_lexical=self.s_cfg.accept_min_lexical,
                    rerank_score=h.metadata.get("rerank_score"),
                    rerank_min=self.s_cfg.reranker_min_score,
                    must_entities=ents,
                )
                if ok:
                    filtered.append(h)
            hits = filtered or hits  # nếu lọc sạch, giữ nguyên top-K để tránh "mất tiếng"

            # Re-rank: blend similarity (Milvus) + rerank_score (CrossEncoder)
            # Chuẩn hoá mềm: sim_norm ∈ [0,1]; rr_norm dùng sigmoid nếu model ko trả về [0,1]
            def _sigmoid(x: float) -> float:
                import math
                return 1.0 / (1.0 + math.exp(-x))

            def _norm_rr(v: float) -> float:
                # Heuristic: nếu điểm đã ở [0,1] thì dùng trực tiếp; nếu >1 thì sigmoid
                return v if 0.0 <= v <= 1.0 else _sigmoid(v)

            alpha = 0.5  # trọng số cho Milvus similarity (0.0..1.0). Bạn có thể cho vào config
            for h in hits:
                rr = _norm_rr(float(h.metadata.get("rerank_score", 0.0)))
                score_mix = alpha * h.similarity + (1.0 - alpha) * rr
                h.metadata["score_mix"] = score_mix

            hits.sort(key=lambda x: x.metadata.get("score_mix", x.similarity), reverse=True)

        # best là hits[0] (sau (re)rank)
        return hits, (hits[0] if hits else None)

    def _search_doc(self, q_vec: List[float]) -> List[Hit]:
        req = SearchReq(
            op="search",
            collection=self.doc_collection,
            search_vector=q_vec,
            top_k=self.s_cfg.doc_top_k,
            metric_type=self.s_cfg.metric,
            output_fields=["id", "text", "source", "metadata"],
            search_params=self.s_cfg.doc_search_params or {},
        )
        res = self.indexer.process(req)
        if res.get("status") != "ok":
            return []
        return self._pp_hits(res["data"].get("results", []), is_faq=False)

    def _pp_hits(self, raw: List[Dict[str, Any]], is_faq: bool) -> List[Hit]:
        out: List[Hit] = []
        for h in raw:
            # pymilvus trả "score" = distance/score tuỳ metric; chuẩn hoá về similarity
            distance = float(h.get("score", 0.0))
            sim = self._distance_to_similarity(distance, self.s_cfg.metric)
            item: Dict[str, Any] = {
                "id": h.get("id") or "",
                "score": distance,
                "similarity": sim,
                "source": h.get("source"),
                "metadata": h.get("metadata"),
            }
            if is_faq:
                item["question"] = h.get("question")
                item["answer"] = h.get("answer")
            else:
                item["text"] = h.get("text")
            out.append(Hit(**item))
        out.sort(key=lambda x: x.similarity, reverse=True)
        return out

    @staticmethod
    def _distance_to_similarity(distance: float, metric: Metric) -> float:
        if metric == "COSINE":
            sim = 1.0 - distance
        elif metric == "IP":
            sim = distance
        else:  # L2
            sim = 1.0 / (1.0 + max(distance, 0.0))
        return max(0.0, min(1.0, sim))

    def _build_prompt(self, query: str, docs: List[Hit]) -> str:
        blocks = []
        for i, d in enumerate(docs, 1):
            src = d.source or ""
            txt = d.text or ""
            blocks.append(f"[DOC {i} | source: {src}]\n{txt}\n")
        ctx = "\n".join(blocks)
        return (
            f"<QUESTION>\n{query}\n</QUESTION>\n\n"
            f"<CONTEXT>\n{ctx}\n</CONTEXT>\n\n"
            f"Chỉ dùng CONTEXT để trả lời. Nếu không đủ dữ liệu, nói rõ không có thông tin."
        )
