# Updated semanticChunker.py
from typing import List
from pydantic import BaseModel, Field, ValidationError
import os
import json
from dotenv import load_dotenv
from google import genai
import re

# ========================
# Setup
# ========================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in .env")

client = genai.Client(api_key=API_KEY)

# ========================
# Data Models (kept minimal)
# ========================
class Chunk(BaseModel):
    text: str

# keep an optional model for backward compatibility if LLM returns objects with this key
class LLMResponse(BaseModel):
    title_or_quick_description: str = Field(..., min_length=1)

# ========================
# Prompt Builder (updated)
# ========================
def build_prompt(chunk: Chunk) -> str:
    """
    NOTE: We now request a single JSON ARRAY of STRINGS.
    Each string is a semantic chunk (may contain multiple sentences).
    """
    return f"""
# TASK: Hybrid Semantic Chunking of Text

## ROLE
You are an expert in semantic chunking — the process of dividing text into coherent, self-contained units based on meaning.

## GOAL
1. Analyze the provided text.
2. Segment it into **semantic chunks**. Each chunk should be a coherent unit of meaning. A chunk **may contain multiple sentences** if they belong together conceptually.
3. Avoid splitting an idea across two chunks. Prefer grouping related sentences (definition, example, enumeration, consequences) into the same chunk.

## GUIDELINES
- Treat each paragraph (separated by blank lines) as one semantic chunk, unless the paragraph is excessively long.
- Do NOT split every sentence individually. Keep multiple related sentences together inside the same chunk.
- **If a paragraph contains a heading and list items, keep the heading and the whole list in ONE single chunk. Do NOT split list items into separate chunks.**
- If a paragraph is too long (exceeds reasonable chunk size), then and only then split it into smaller coherent chunks.
- Maintain the original language and phrasing as much as possible.

## OUTPUT FORMAT (STRICT JSON)
Return exactly **one** JSON array of strings. Each element in the array should be one semantic chunk (a string).
Example:
[
  "Artificial Intelligence (AI) is a branch of computer science. It focuses on creating machines that can perform tasks that normally require human intelligence.",
  "Examples include problem-solving, language understanding, and learning from experience."
]

## INPUT:
\"\"\"{chunk.text}\"\"\" 
""".strip()

# ========================
# LLM Call (unchanged)
# ========================
def call_llm(prompt: str) -> str:
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    text = resp.text
    if not text:
        raise ValueError(f"No text in LLM response: {resp}")
    return text.strip()

# ========================
# Safe JSON parse (improved to find arrays too)
# ========================
def safe_json_parse(raw: str):
    """
    Try to extract JSON if Gemini adds extra text around it.
    Will attempt to find a JSON array first, then object.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # try to find an array [...] or object {...} inside raw
        arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if arr_match:
            return json.loads(arr_match.group(0))
        obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(0))
        raise

# ========================
# Process chunk (more flexible parsing)
# ========================
def process_chunk(chunk: Chunk) -> List[str]:
    if not chunk.text.strip():
        raise ValueError("Empty chunk passed to LLM")

    prompt = build_prompt(chunk)
    raw = call_llm(prompt)

    try:
        parsed = safe_json_parse(raw)

        # Case 1: JSON array of strings
        if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
            return [s.strip() for s in parsed if s and s.strip()]

        # Case 2: JSON array of objects (try common keys)
        if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
            results = []
            for item in parsed:
                if "title_or_quick_description" in item:
                    results.append(str(item["title_or_quick_description"]).strip())
                elif "text" in item:
                    results.append(str(item["text"]).strip())
                else:
                    # fallback: first string value
                    for v in item.values():
                        if isinstance(v, str) and v.strip():
                            results.append(v.strip())
                            break
            if results:
                return results

        # Case 3: single string (raw JSON string)
        if isinstance(parsed, str):
            return [parsed.strip()]

        # Case 4: single object
        if isinstance(parsed, dict):
            if "title_or_quick_description" in parsed:
                return [str(parsed["title_or_quick_description"]).strip()]
            if "text" in parsed:
                return [str(parsed["text"]).strip()]
            # fallback to first string value
            for v in parsed.values():
                if isinstance(v, str) and v.strip():
                    return [v.strip()]

        raise ValueError(f"Unexpected JSON structure from LLM: {type(parsed)} -- raw: {raw}")

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise ValueError(f"Invalid LLM response: {raw}") from e

# ========================
# Semantic Chunking (list detection improved)
# ========================
def semantic_chunk(text: str, max_chunk_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """
    Semantic chunking:
    - Paragraph = tách bằng dòng trống (Windows \r\n hoặc Unix \n\n).
    - Bên trong 1 paragraph, giữ nguyên các xuống dòng đơn (list items).
    - Nếu paragraph quá dài thì cắt theo câu.
    """
    import re

    # Chuẩn hóa: đổi tất cả \r\n và \r thành \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split đoạn: dựa vào ít nhất 1 dòng trống
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: List[Chunk] = []
    for para in paragraphs:
        # Giữ nguyên xuống dòng đơn trong paragraph (cho đẹp list)
        # Chỉ chuẩn hóa space thừa
        lines = [l.strip() for l in para.split("\n")]
        para_norm = "\n".join(lines)

        if len(para_norm) <= max_chunk_size:
            chunks.append(Chunk(text=para_norm))
            continue

        # Nếu dài quá thì cắt theo câu
        sentences = re.split(r'(?<=[\.!?])\s+', para_norm.replace("\n", " "))
        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                chunks.append(Chunk(text=current_chunk.strip()))
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
        if current_chunk:
            chunks.append(Chunk(text=current_chunk.strip()))

    return chunks



# ========================
# Pipeline & IO (unchanged except robust parsing)
# ========================
def semantic_chunker_pipeline(text: str) -> List[str]:
    chunks = semantic_chunk(text)
    outputs = []
    for chunk in chunks:
        outputs.extend(process_chunk(chunk))
    return outputs

def read_txt_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def save_outputs_to_json(outputs: List[str], filename: str = "chunk_outputs.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=2)

def semantic_chunker(filepath):
    input_text = read_txt_file(filepath)
    outputs = semantic_chunker_pipeline(input_text)
    save_outputs_to_json(outputs, "semantic_chunks.json")
