# -*- coding: utf-8 -*-
"""
Smoke test: biến đổi list chunks -> JSON chuẩn qua pydantic-ai + KERNEL

Chạy:
    # Cách 1: đặt key qua ENV
    #   PowerShell:  $env:GEMINI_API_KEY="sk-..."
    #   Bash:        export GEMINI_API_KEY=sk-...
    uv run python -m examples.transform_smoketest

    # Cách 2: dùng .env (tuỳ chọn)
    #   Tạo file .env ở gốc project: GEMINI_API_KEY=sk-...
    #   rồi chạy như trên
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# kernel
from src.llm.llm_kernel import KERNEL, GoogleConfig

load_dotenv()

# --- Ép dùng Google (Gemini); nếu muốn dựa ENV/UI thì comment 2 dòng dưới ---
KERNEL.set_active_config(GoogleConfig(model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")))
print("[KERNEL] Active=Google (Gemini)")

# --- Lấy model đang active, cho phép override api_key qua ENV ---
model = KERNEL.get_active_model(
    model_name=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
)

@dataclass
class MyDeps:
    listChunks: List[str]

def build_prompt(ctx: RunContext[MyDeps]) -> str:
    data_json = json.dumps({"chunks": ctx.deps.listChunks}, ensure_ascii=False, indent=2)
    return f"""You are a specialist in generating data, you can change a json file including
chunks to another json file with a meaningful sentence from each given chunk following these rules:

**RULES**
1. The output must have json schema (list of dictionaries).
2. Each item is a dictionary with keys "original" and "transformed".
3. Return strictly JSON, without any Markdown code fences.

**Example**
Given input:
{{
  "chunks": [
    "The solar system consists of the Sun",
    "and all the celestial bodies that orbit it,"
  ]
}}
Correct output:
[
  {{
    "original": "The solar system consists of the Sun",
    "transformed": "The solar system is made up of the Sun as its central star."
  }},
  {{
    "original": "and all the celestial bodies that orbit it,",
    "transformed": "It also includes all the celestial bodies that revolve around the Sun."
  }}
]

Your input is:
{data_json}
"""

def strip_fences(s: str) -> str:
    for f in ("```json", "```", "'''json", "'''"):
        s = s.replace(f, "")
    return s.strip()

def main():
    # dữ liệu mẫu (bạn có thể thay bằng file thật nếu muốn)
    chunks = [
        "The solar system consists of the Sun",
        "and all the celestial bodies that orbit it,",
        "including planets, moons, asteroids, and comets.",
        "Scientists study these objects",
        "to understand the formation and evolution",
        "of our cosmic neighborhood."
    ]

    # tạo 2 worker giống code gốc
    agent1 = Agent(model=model, deps_type=MyDeps)
    agent2 = Agent(model=model, deps_type=MyDeps)

    @agent1.system_prompt
    def sys1(ctx: RunContext[MyDeps]) -> str:
        return build_prompt(ctx)

    @agent2.system_prompt
    def sys2(ctx: RunContext[MyDeps]) -> str:
        return build_prompt(ctx)

    # chia đôi danh sách
    size = len(chunks) // 2
    dep1 = MyDeps(listChunks=chunks[:size])
    dep2 = MyDeps(listChunks=chunks[size:])

    # chạy
    res1 = agent1.run_sync(deps=dep1).output
    res2 = agent2.run_sync(deps=dep2).output

    res1 = strip_fences(res1)
    res2 = strip_fences(res2)

    print("\n=== Worker1 Raw ===")
    print(res1)
    print("\n=== Worker2 Raw ===")
    print(res2)

    # parse + gộp lại
    try:
        j1 = json.loads(res1)
        j2 = json.loads(res2)
        assert isinstance(j1, list) and isinstance(j2, list)
        merged = j1 + j2
    except Exception as e:
        print("[ERROR] JSON parse failed:", e)
        return

    # xuất ra file (như logic gốc), và in preview
    out_path = "src/llm/URag_D/received_output.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existed = json.load(f)
        except Exception:
            existed = []

        if isinstance(existed, list):
            existed.extend(merged)
        else:
            existed = [existed, merged]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(existed, f, ensure_ascii=False, indent=2)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Wrote {len(merged)} items to {out_path}")
    print("=== Preview (first 2) ===")
    for i, it in enumerate(merged[:2], 1):
        print(f"{i}. {it.get('original','?')} -> {it.get('transformed','?')}")

if __name__ == "__main__":
    # nhanh gọn: kiểm tra có key chưa
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise SystemExit("GEMINI_API_KEY/GOOGLE_API_KEY is required for Gemini test.")
    main()
