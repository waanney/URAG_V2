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
# TASK: Vietnamese Quick Description Generation 

## ROLE
You are an expert in crafting concise, single-sentence descriptions that capture the main content of a text chunk.
---------------------
## GOAL
Split the provided input text into multiple semantic chunks. Each chunk should represent a coherent and self-contained idea or topic. Preserve the original wording inside each chunk without rewriting or summarizing.
---------------------
## GUIDELINES
1. **Core Focus:**
   * Analyze the rewritten text chunk to identify its key message or theme.
   * Do not repeat the text verbatim; instead, paraphrase into a clearer summary.
2. **Language:**
   * Input and output are in Vietnamese.
   * Do not mix in English words unless they appear in the original text.
3. **Conciseness:**
   * The description must be exactly **one grammatically complete sentence**, ending with a period.
4. **Output Format:**
Always return exactly one JSON object, wrapped inside a JSON array, like this:
[
  {{"title_or_quick_description": "..."}},
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
def semantic_chunk(text: str) -> List[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return [Chunk(text=p) for p in paragraphs]

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