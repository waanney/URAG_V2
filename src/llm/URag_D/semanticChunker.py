from typing import List
from pydantic import BaseModel, Field, ValidationError
import os
import json
from dotenv import load_dotenv
from google import genai

# ========================
# Setup
# ========================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env")

client = genai.Client(api_key=API_KEY)


# ========================
# Data Models
# ========================
class Chunk(BaseModel):
    text: str

class LLMResponse(BaseModel):
    title_or_quick_description: str = Field(..., min_length=3)


# ========================
# Prompt Builder
# ========================
def build_prompt(chunk: Chunk) -> str:
    return f"""
You are a semantic chunker.
Your task:
1. Read the following text chunk.
2. Write a single, short, one-sentence summary in Vietnamese.
3. Return the result strictly as a JSON object in this format:

{{
  "title_or_quick_description": "..."
}}
{{
  "title_or_quick_description": "..."
}}

## Đoạn văn cần tóm tắt:
\"\"\"{chunk.text}\"\"\"
""".strip()


# ========================
# LLM Call
# ========================
def call_llm(prompt: str) -> str:
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},  # force JSON output
    )

    text = resp.text
    if not text:
        raise ValueError(f"No text in LLM response: {resp}")
    return text.strip()


def safe_json_parse(raw: str):
    """
    Try to extract JSON if Gemini adds extra text around it.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def process_chunk(chunk: Chunk) -> str:
    if not chunk.text.strip():
        raise ValueError("Empty chunk passed to LLM")

    prompt = build_prompt(chunk)
    raw = call_llm(prompt)

    try:
        parsed = safe_json_parse(raw)
        validated = LLMResponse(**parsed)
        return validated.title_or_quick_description
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid LLM response: {raw}") from e


# ========================
# Semantic Chunking
# ========================
def semantic_chunk(text: str) -> List[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return [Chunk(text=p) for p in paragraphs]


def semantic_chunker_pipeline(text: str) -> List[str]:
    chunks = semantic_chunk(text)
    return [process_chunk(chunk) for chunk in chunks]

# ========================
# Save Outputs
# ========================
def save_outputs_to_json(outputs: List[str], filename: str = "chunk_outputs.json"):
    """
    Save pipeline outputs to a JSON file.
    Each summary is stored as an element in a list.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=2)
    print(f"✅ Outputs saved to {filename}")

# ========================
# Example Usage
# ========================
if __name__ == "__main__":
    input_text = """
    Các mốc thời gian quan trọng

Năm 1957: Trung tâm Quốc gia Kỹ thuật được thành lập theo Sắc lệnh 213-GD ngày 29/06/1957, gồm 4 trường kỹ thuật, công nghệ và chuyên nghiệp: Trường Cao Đẳng Công Chánh, Trường Vô tuyến Điện, Trường Hàng Hải Thương Thuyền và Trường Thương Mại.

1972: Trung tâm Quốc gia Kỹ thuật được đổi tên thành Học viện Quốc gia Kỹ thuật theo Sắc lệnh 135SL/GD ngày 15/9/1972 và Sắc lệnh số 53-SL/GD ngày 21/3/1973, gồm 6 trường thành viên.

1974: Học viện Quốc gia Kỹ thuật được đổi tên thành Trường Đại học Kỹ thuật và là thành viên của Viện Đại học Bách khoa Thủ Đức.

1976: Trường được mang tên Trường Đại học Bách khoa theo Quyết định số 426/TTg ngày 27/10/1976.

1996: Trường được mang tên Trường Đại học Kỹ thuật trực thuộc Đại học Quốc gia TP. Hồ Chí Minh theo Quyết định số 1235/GD-ĐT ngày 30/3/1996.

2001: Trường được mang tên Trường Đại học Bách khoa theo Quyết định số 15/2001/QĐ-TTg của Thủ tướng Chính phủ về việc tổ chức lại Đại học Quốc gia thành phố Hồ Chí Minh.
    """
    outputs = semantic_chunker_pipeline(input_text)
    print(outputs)

    save_outputs_to_json(outputs, "semantic_chunks.json")