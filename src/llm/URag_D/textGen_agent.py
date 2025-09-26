# src/llm/URag_D/chunk_rewriter_agent.py
# -*- coding: utf-8 -*-
"""
ChunkRewriterAgent — URAG-D (post-chunk) processor
---------------------------------------------------
Nhiệm vụ (không làm chunking):
  1) Context Extraction: gi = L(oi)
  2) Chunk Reconstruction:
       tij = L(cij | gi)        # rewrite chunk theo general context
       hij = L(tij)             # condense thành 1 câu headline
       dij = concat(hij, tij)   # nếu muốn, concat để index/hiển thị
  3) Trả về augmented chunks cho downstream (FManager / indexing)

Phụ thuộc:
- pydantic, pydantic-ai
- src/llm/llm_kernel.py  (KERNEL chọn LLM runtime; ví dụ Gemini)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable
from dataclasses import dataclass
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
load_dotenv()
# LLM kernel (bạn đã có)
from src.llm.llm_kernel import KERNEL, GoogleConfig

# ===================== Schemas =====================

class ChunkIn(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChunkOut(BaseModel):
    doc_id: str
    chunk_id: str
    original: str
    transformed: str
    headline: str
    # tiện cho index/hiển thị: phần concat (headline + \n + transformed)
    combined: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocWithChunks(BaseModel):
    doc_id: str
    full_text: str
    chunks: List[ChunkIn]

class RewriteSummary(BaseModel):
    doc_id: str
    num_chunks_in: int
    num_chunks_out: int
    general_context_len: int

class RewriteResponse(BaseModel):
    general_context: str
    items: List[ChunkOut]
    summary: RewriteSummary

# ===================== Prompt builders =====================

def _strip_code_fences(s: str) -> str:
    for fence in ("```json", "```", "'''json", "'''"):
        s = s.replace(fence, "")
    return s.strip()

def _json_loads_strict(s: str) -> Any:
    return json.loads(_strip_code_fences(s))

def prompt_general_context(full_text: str) -> str:
    return f"""
Bạn là chuyên gia phân tích. Hãy đọc DOCUMENT và trích xuất **general context** (tổng quan phạm vi, chủ đề, mốc thời gian/điều kiện quan trọng).
- Chỉ dựa trên văn bản, không bịa.
- Viết tiếng Việt, 3–8 gạch đầu dòng.

DOCUMENT:
\"\"\"{full_text}\"\"\"

Trả về đúng JSON, không kèm code fence:
{{
  "context": "- ý 1\\n- ý 2\\n- ..."
}}
""".strip()

@dataclass
class Deps:
    general_context: str
    chunk_text: str

def prompt_rewrite(ctx: RunContext[Deps]) -> str:
    return f"""
# ROLE
Bạn **rewrite** đoạn CHUNK dựa trên **GENERAL CONTEXT** để mạch lạc, nhất quán và KHÔNG bịa. Sau đó **tóm tắt thành 1 câu headline** (10–25 từ).

# GENERAL CONTEXT
{ctx.deps.general_context}

# CHUNK (original)
{ctx.deps.chunk_text}

# INSTRUCTIONS
- Giữ nguyên ý chính; chỉ dùng thông tin có trong GENERAL CONTEXT (nếu cần làm rõ).
- Không thêm dữ kiện mới; không suy diễn; không emoji/dấu ngoặc trong headline.
- Văn phong rõ ràng, tiếng Việt chuẩn.

# OUTPUT (JSON STRICT)
[
  {{
    "original": "<nguyên văn chunk>",
    "transformed": "<chunk đã rewrite>",
    "headline": "<1 câu headline>"
  }}
]
""".strip()

# ===================== LLM Agents =====================

class _ContextAgent:
    """Sinh general context từ full_text (1 lần/tài liệu)."""
    def __init__(self, model_name: Optional[str] = None):
        if KERNEL.load_active_config() is None:
            KERNEL.set_active_config(GoogleConfig(model=model_name or "gemini-2.0-flash"))
        self.model = KERNEL.get_active_model(model_name=model_name or "gemini-2.0-flash")
        self.agent = Agent(self.model)

    def run(self, full_text: str) -> str:
        out = self.agent.run_sync(prompt_general_context(full_text)).output
        try:
            obj = _json_loads_strict(out)
            ctx = str(obj.get("context", "")).strip()
            return ctx
        except Exception:
            return out.strip()

class _RewriteAgent:
    """Rewrite từng chunk theo general context + sinh headline."""
    def __init__(self, model_name: Optional[str] = None):
        if KERNEL.load_active_config() is None:
            KERNEL.set_active_config(GoogleConfig(model=model_name or "gemini-2.0-flash"))
        self.model = KERNEL.get_active_model(model_name=model_name or "gemini-2.0-flash")

        self.agent = Agent(model=self.model, deps_type=Deps)

        @self.agent.system_prompt
        def _sys(ctx: RunContext[Deps]) -> str:
            return prompt_rewrite(ctx)

    def run(self, general_context: str, chunk_text: str) -> Dict[str, str]:
        deps = Deps(general_context=general_context, chunk_text=chunk_text)
        out = self.agent.run_sync(deps=deps).output
        data = _json_loads_strict(out)[0]
        return {
            "original": str(data.get("original", "")).strip() or chunk_text,
            "transformed": str(data.get("transformed", "")).strip() or chunk_text,
            "headline": str(data.get("headline", "")).strip() or "Tổng hợp nội dung",
        }

# ===================== Orchestrator =====================

class ChunkRewriterAgent:
    """
    Nhận: 1 tài liệu đã được chunk sẵn (DocWithChunks)
    Trả:  general_context + danh sách ChunkOut (original, transformed, headline, combined)
    """
    def __init__(self, model_name: Optional[str] = None, concat_headline: bool = True):
        self.concat_headline = concat_headline
        self.ctx_agent = _ContextAgent(model_name=model_name)
        self.rw_agent = _RewriteAgent(model_name=model_name)

    def process_one(self, doc: DocWithChunks) -> RewriteResponse:
        if not doc.full_text.strip() or not doc.chunks:
            return RewriteResponse(
                general_context="",
                items=[],
                summary=RewriteSummary(
                    doc_id=doc.doc_id, num_chunks_in=len(doc.chunks),
                    num_chunks_out=0, general_context_len=0
                ),
            )

        # 1) context
        general = self.ctx_agent.run(doc.full_text)

        # 2) rewrite từng chunk
        outs: List[ChunkOut] = []
        for ch in doc.chunks:
            try:
                rw = self.rw_agent.run(general_context=general, chunk_text=ch.text)
            except Exception:
                rw = {"original": ch.text, "transformed": ch.text, "headline": "Tổng hợp nội dung"}

            combined = f"{rw['headline']}\n{rw['transformed']}" if self.concat_headline else rw["transformed"]
            outs.append(ChunkOut(
                doc_id=ch.doc_id,
                chunk_id=ch.chunk_id,
                original=rw["original"],
                transformed=rw["transformed"],
                headline=rw["headline"],
                combined=combined,
                metadata=ch.metadata or {}
            ))

        return RewriteResponse(
            general_context=general,
            items=outs,
            summary=RewriteSummary(
                doc_id=doc.doc_id,
                num_chunks_in=len(doc.chunks),
                num_chunks_out=len(outs),
                general_context_len=len(general)
            ),
        )

    def process_many(self, docs: Iterable[DocWithChunks]) -> List[RewriteResponse]:
        return [self.process_one(d) for d in docs]

# ===================== Demo =====================
if __name__ == "__main__":
    demo = DocWithChunks(
        doc_id="doc_001",
        full_text=(
            "Quy chế tuyển sinh nêu mốc thời gian đăng ký và ngưỡng đảm bảo chất lượng đầu vào (điểm sàn). "
            "Sinh viên xuất sắc có thể được xét học bổng theo quy định."
        ),
        chunks=[
            ChunkIn(doc_id="doc_001", chunk_id="doc_001__a1", text="Mốc thời gian đăng ký xét tuyển sẽ được công bố trong kế hoạch."),
            ChunkIn(doc_id="doc_001", chunk_id="doc_001__a2", text="Điểm sàn áp dụng theo thông báo từng năm."),
        ],
    )
    agent = ChunkRewriterAgent(model_name="gemini-1.5-flash", concat_headline=True)
    res = agent.process_one(demo)
    print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
