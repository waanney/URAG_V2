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
# TASK: Vietnamese Quick Description Generation with Hybrid Semantic Chunking

## ROLE
You are an expert in crafting concise, single-sentence descriptions that capture the main content of a text chunk, and in adjusting semantic chunking when necessary.

---------------------
## GOAL
Your task is:
1. Analyze the provided rewritten text chunk.
2. If the chunk already makes sense semantically, generate a **single-sentence title/quick description**.
3. If the chunk still contains multiple distinct ideas, you are allowed to **mentally split it into smaller sub-chunks** and generate **one description per sub-chunk**. Each description must be a separate JSON object inside the returned array.

---------------------
## GUIDELINES
1. **Core Focus:**
   * Identify the main idea(s) of the chunk or its sub-parts.
   * If splitting, ensure each description corresponds to one coherent semantic unit.
2. **Language:**
   * Input and output are in Vietnamese.
   * Do not mix in English words unless they appear in the original text.
3. **Conciseness:**
   * Each description must be exactly **one grammatically complete sentence**, ending with a period.
4. **Output Format:**
   * Return the final output in strict JSON format with a single key "title_or_quick_description".
   * Always return exactly one JSON array.  
   Example (for multiple sub-chunks):
[
  {{"title_or_quick_description": "..."}}
]

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

def process_chunk(chunk: Chunk) -> List[str]:
    if not chunk.text.strip():
        raise ValueError("Empty chunk passed to LLM")

    prompt = build_prompt(chunk)
    raw = call_llm(prompt)

    try:
        parsed = safe_json_parse(raw)

        # Nếu là list JSON
        if isinstance(parsed, list):
            validated = [LLMResponse(**item) for item in parsed]
            return [v.title_or_quick_description for v in validated]

        # Nếu là một object JSON
        elif isinstance(parsed, dict):
            validated = LLMResponse(**parsed)
            return [validated.title_or_quick_description]

        else:
            raise ValueError(f"Unexpected JSON type: {type(parsed)}")

    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid LLM response: {raw}") from e


# ========================
# Semantic Chunking
# ========================
def semantic_chunk(text: str, max_chunk_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """
    Hybrid semantic chunking:
    1. Split text into paragraphs (\n\n).
    2. If paragraph too long -> split by sentence.
    3. If still too long -> sliding window with overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        # Chuẩn hóa whitespace
        para = re.sub(r'\s+', ' ', para).strip()

        if len(para) <= max_chunk_size:
            chunks.append(Chunk(text=para))
            continue

        # Step 2: split into sentences
        sentences = re.split(r'(?<=[\.!?])\s+', para)
        current_chunk = ""

        for sentence in sentences:
            if not sentence.strip():
                continue

            if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                chunks.append(Chunk(text=current_chunk.strip()))
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(Chunk(text=current_chunk.strip()))

        # Step 3: if any chunk still too long, break into sliding windows
        final_chunks = []
        for c in chunks:
            if len(c.text) > max_chunk_size:
                text = c.text
                start = 0
                while start < len(text):
                    end = min(start + max_chunk_size, len(text))
                    window = text[start:end]
                    final_chunks.append(Chunk(text=window.strip()))
                    start += max_chunk_size - overlap
            else:
                final_chunks.append(c)
        chunks = final_chunks

    return chunks

def semantic_chunker_pipeline(text: str) -> List[str]:
    chunks = semantic_chunk(text)
    outputs = []
    for chunk in chunks:
        outputs.extend(process_chunk(chunk))
    return outputs


# ========================
# File Reader
# ========================
def read_txt_file(filepath: str) -> str:
    """
    Read the full content of a .txt file as a string.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


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
    # print(f"✅ Outputs saved to {filename}")

# ========================
# Example Usage
# ========================

def semantic_chunker(filepath):
    input_text = read_txt_file(filepath)
    
    outputs = semantic_chunker_pipeline(input_text)
    # print(outputs)

    save_outputs_to_json(outputs, "semantic_chunks.json")

# if __name__ == "__main__":
# #     input_text = """
# # Trường Đại học Bách khoa - ĐHQG-HCM  là một trường thành viên của hệ thống Đại học Quốc gia TP. Hồ Chí Minh. Tiền thân của Trường là Trung tâm Quốc gia Kỹ thuật được thành lập vào năm 1957. Hiện nay, Trường ĐH Bách Khoa là trung tâm đào tạo, nghiên cứu khoa học và chuyển giao công nghệ lớn nhất các tỉnh phía Nam và là trường đại học kỹ thuật quan trọng của cả nước.    """
    
#     input_text = read_txt_file(r"D:\Admin\Documents\CodeSavingMain\1_URA\Proj\DLTooClose\URAG_V2\example_Input.txt")
    
#     outputs = semantic_chunker_pipeline(input_text)
#     # print(outputs)

#     save_outputs_to_json(outputs, "semantic_chunks.json")