# -*- coding: utf-8 -*-
"""
SearchAgent (Pydantic) — URAG Inference Pipeline
Tier-1: FAQ -> trả answer trực tiếp nếu vượt ngưỡng
Tier-2: Doc  -> xây prompt từ doc, gửi LLM (Gemini) -> trả lời
Tier-3: Fallback nếu không có gì liên quan

Phụ thuộc:
- /src/embedding/embedding_agent.py  (EmbedderAgent)
- /src/indexing/indexing_agent.py    (IndexingAgent, SearchReq)
- Gemini API: ưu tiên SDK mới "google.genai" (GA, 2025); fallback "google.generativeai"

ENV:
- GEMINI_API_KEY=xxx   (SDK mới auto đọc; vẫn hỗ trợ truyền thủ công)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal, Tuple
import os, time, math, json

from pydantic import BaseModel, Field, field_validator
# === Agents của bạn ===
from src.embedding.embedding_agent import EmbedderAgent, EmbConfig
from src.indexing.indexing_agent import IndexingAgent, AgentConfig, SearchReq

# ======================= LLM (Gemini) adapter =======================

class GeminiConfig(BaseModel):
    model: str = Field(default="gemini-1.5-flash")
    api_key: Optional[str] = None
    system_instruction: Optional[str] = Field(
        default="Bạn là trợ lý chỉ trả lời dựa trên CONTEXT. Nếu thiếu dữ liệu, hãy nói 'không có thông tin'."
    )
    temperature: float = 0.2
    top_p: float = 0.9
    candidate_count: int = 1

class GeminiLLM:
    """
    Supports both SDKs:
      - New (preferred):   from google import genai
      - Legacy (fallback): import google.generativeai as genai
    Typed as Any to keep Pylance happy across versions.
    """
    def __init__(self, cfg: GeminiConfig):
        self.cfg = cfg
        self._sdk_kind: str = "none"
        self._client: Any = None
        self._model_obj: Any = None   # legacy GenerativeModel instance
        self._init_client()

    def _init_client(self) -> None:
        api_key = self.cfg.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu API key cho Gemini (đặt GEMINI_API_KEY hoặc GOOGLE_API_KEY).")

        # Try NEW SDK first
        try:
            # New SDK (GA 2025): from google import genai
            from google import genai as genai_new  # type: ignore[import-not-found]
            client: Any = genai_new.Client(api_key=api_key)  # type: ignore[call-arg]
            # Smoke test a lightweight attribute to confirm it's the new SDK
            _ = getattr(client, "models", None)
            self._sdk_kind = "new"
            self._client = client
            return
        except Exception:
            pass

        # Fallback to LEGACY SDK
        import google.generativeai as genai_old  # type: ignore[import-not-found]
        # Pylance sometimes flags these as “not exported”; treat module as Any.
        genai_old = genai_old  # type: ignore[no-redef]
        # configure + GenerativeModel exist at runtime; typing: Any
        genai_old.configure(api_key=api_key)  # type: ignore[attr-defined]
        model_obj: Any = genai_old.GenerativeModel(self.cfg.model)  # type: ignore[attr-defined]
        self._sdk_kind = "old"
        self._client = genai_old
        self._model_obj = model_obj

    # --- replace inside class GeminiLLM ---
    def generate(self, prompt: str) -> str:
        sys_inst = self.cfg.system_instruction

        if self._sdk_kind == "new":
            # NEW SDK: contents should be a string (or Parts), not role dicts.
            base_cfg = {
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "candidate_count": self.cfg.candidate_count,
            }
            try:
                cfg = dict(base_cfg)
                if sys_inst:
                    # Preferred way for new SDK: system_instruction in config
                    cfg["system_instruction"] = sys_inst
                resp: Any = self._client.models.generate_content(  # type: ignore[attr-defined]
                    model=self.cfg.model,
                    contents=prompt,   # <--- string, not list of role dicts
                    config=cfg,
                )
            except Exception:
                # Fallback: prepend system text into the prompt and retry without system_instruction
                fused_prompt = f"{sys_inst}\n\n{prompt}" if sys_inst else prompt
                resp: Any = self._client.models.generate_content(  # type: ignore[attr-defined]
                    model=self.cfg.model,
                    contents=fused_prompt,
                    config=base_cfg,
                )

            # Robust text extraction
            text = getattr(resp, "text", None)
            if isinstance(text, str) and text.strip():
                return text

            cand = getattr(resp, "candidates", None)
            if cand and len(cand) > 0:
                content = getattr(cand[0], "content", None)
                parts = getattr(content, "parts", None)
                if parts:
                    for p in parts:
                        t = getattr(p, "text", None)
                        if t:
                            return t
            return ""

        # --- legacy SDK path unchanged ---
        try:
            from google.generativeai.types import GenerationConfig  # type: ignore[import-not-found]
            gen_cfg = GenerationConfig(  # type: ignore[call-arg]
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                candidate_count=self.cfg.candidate_count,
            )
        except Exception:
            gen_cfg = None

        contents_legacy: list[str] = []
        if sys_inst:
            contents_legacy.append(sys_inst)
        contents_legacy.append(prompt)

        resp: Any = self._model_obj.generate_content(  # type: ignore[attr-defined]
            contents_legacy,
            generation_config=gen_cfg,
        )
        return getattr(resp, "text", "") or ""


# ======================= Search agent (Pydantic) =======================

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
    metadata: Optional[Dict[str, Any]] = None

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
        g_cfg: Optional[GeminiConfig] = None,
        embedder: Optional[EmbedderAgent] = None,
        indexer: Optional[IndexingAgent] = None,
        llm: Optional[GeminiLLM] = None,
    ):
        self.s_cfg = s_cfg
        self.embedder = embedder or EmbedderAgent(e_cfg or EmbConfig())
        self.indexer = indexer or IndexingAgent(i_cfg or AgentConfig())
        self.llm = llm or GeminiLLM(g_cfg or GeminiConfig())

        # khớp suffix từ IndexingAgent để tránh lệch tên
        self.doc_collection = f"{self.s_cfg.collection_base}{self.indexer.cfg.doc_suffix}"
        self.faq_collection = f"{self.s_cfg.collection_base}{self.indexer.cfg.faq_suffix}"

    # ---------------- public API ----------------
    def answer(self, query: str) -> SearchResponse:
        t0 = time.time()
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
            llm_answer = self.llm.generate(prompt).strip()
            if self.s_cfg.disclaimer and llm_answer:
                llm_answer += f"\n\n{self.s_cfg.disclaimer}"
            trace = SearchTrace(
                latency_ms=int((time.time() - t0) * 1000),
                params={"tDOC": self.s_cfg.tDOC, "doc_top_k": self.s_cfg.doc_top_k, "metric": self.s_cfg.metric, "max_ctx_docs": self.s_cfg.max_ctx_docs},
                doc_hits=doc_hits, ctx_used=ctx, prompt=prompt
            )
            return SearchResponse(tier="doc", final_answer=llm_answer or "", trace=trace)

        # 4) Fallback
        trace = SearchTrace(
            latency_ms=int((time.time() - t0) * 1000),
            params={"tFAQ": self.s_cfg.tFAQ, "tDOC": self.s_cfg.tDOC},
            faq_hits=faq_hits, doc_hits=doc_hits
        )
        return SearchResponse(tier="none", final_answer="Xin lỗi, tôi không tìm thấy thông tin liên quan trong kho dữ liệu.", trace=trace)

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
            item = {
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
