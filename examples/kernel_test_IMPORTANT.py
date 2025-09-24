# -*- coding: utf-8 -*-
"""
Smoke test cho LLMKernel với Google Gemini (pydantic-ai)

Yêu cầu:
- pip/uv: pydantic-ai, python-dotenv
- ENV: GEMINI_API_KEY (hoặc GOOGLE_API_KEY)

Chạy:
    uv run python examples/kernel_gemini_smoketest.py
hoặc:
    python examples/kernel_gemini_smoketest.py
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# import kernel (đặt đúng path tới file của bạn)
from src.llm.llm_kernel import LLMKernel, GoogleConfig

load_dotenv()

def make_kernel_model():
    # Kernel deferred-binding: chỉ cần config Google (key đọc từ ENV khi build model)
    kernel = LLMKernel()
    kernel.configure(GoogleConfig())  # không cần truyền api_key ở đây

    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    model = kernel.get_model(
        model_name,
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    )
    return model_name, model

# ---- Test 1: text generation đơn giản qua Agent ----
def simple_text_test():
    model_name, model = make_kernel_model()

    agent = Agent(model=model)

    @agent.system_prompt
    def sys_prompt(_: RunContext[None]) -> str:
        return "You are a concise assistant. Answer in Vietnamese."

    print(f"[Test1] Using model: {model_name}")
    res = agent.run_sync("Chào một câu ngắn gọn và lịch sự.")
    print("[Test1 Output]")
    print(res.output.strip())
    print("-" * 60)

# ---- Test 2: ép JSON (không markdown), chia nhỏ câu ----
@dataclass
class MyDeps:
    chunks: List[str]

def build_prompt(ctx: RunContext[MyDeps]) -> str:
    data_json = json.dumps({"chunks": ctx.deps.chunks}, ensure_ascii=False)
    # Quy tắc: trả về JSON THUẦN (không ```), mảng các dict {original, transformed}
    return f"""
Bạn là chuyên gia chuyển đổi câu.
Yêu cầu:
- Trả về **JSON THUẦN** (không markdown, không ```).
- Dạng: [{{
    "original": "...",
    "transformed": "..."
}}]
- Giữ nguyên ngữ nghĩa, viết lại câu rõ ràng hơn.

Input JSON:
{data_json}
"""

def strict_json_test():
    model_name, model = make_kernel_model()

    agent = Agent(model=model, deps_type=MyDeps)

    @agent.system_prompt
    def sys_prompt(ctx: RunContext[MyDeps]) -> str:
        return build_prompt(ctx)

    chunks = [
        "Hệ Mặt Trời gồm Mặt Trời",
        "và các thiên thể quay quanh nó,",
        "bao gồm hành tinh, vệ tinh, tiểu hành tinh và sao chổi."
    ]

    print(f"[Test2] Using model: {model_name}")
    res = agent.run_sync(deps=MyDeps(chunks=chunks))
    raw = res.output

    # làm sạch nếu lỡ bọc fence
    for fence in ("```json", "```", "'''json", "'''"):
        raw = raw.replace(fence, "")
    raw = raw.strip()

    print("[Test2 Raw JSON]")
    print(raw)
    try:
        data = json.loads(raw)
        assert isinstance(data, list), "Kỳ vọng một mảng JSON."
        assert all(isinstance(x, dict) for x in data), "Mỗi phần tử phải là dict."
        print("[Test2 Parsed OK] Số phần tử:", len(data))
    except Exception as e:
        print("[Test2 Parse ERROR]", e)
    print("-" * 60)

if __name__ == "__main__":
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Thiếu GEMINI_API_KEY/GOOGLE_API_KEY. Vui lòng export trước khi chạy.")
    simple_text_test()
    strict_json_test()
