from __future__ import annotations
import os
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timezone

import ujson
import numpy as np
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.neighbors import NearestNeighbors

nltk.download("punkt", quiet=True)

# ---------------- Data Models ----------------

class Provenance(BaseModel):
    source_type: str  # "seed_faq" | "doc_qa" | "paraphrase"
    source_id: Optional[str] = None
    doc_path: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("source_type")
    def check_source_type_non_empty(cls, v):
        if not v.strip():
            raise ValueError("source_type cannot be empty")
        return v

class FAQItem(BaseModel):
    id: str
    question: str
    answer: str
    canonical_id: str
    provenance: Provenance

    @field_validator("question", "answer")
    def check_non_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v

# Rebuild Pydantic models to resolve postponed annotations
Provenance.model_rebuild()
FAQItem.model_rebuild()

# ---------------- Utilities ----------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(ujson.loads(line))
                except ujson.JSONDecodeError:
                    continue  # Skip invalid JSON lines
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(ujson.dumps(r, ensure_ascii=False) + "\n")

def simple_chunks(text: str, max_chars: int = 1500) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    sents = sent_tokenize(text)
    chunks, buf = [], ""
    for s in sents:
        if len(buf) + len(s) + 1 > max_chars:
            if buf.strip():
                chunks.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

# ---------------- Embedding Index (scikit-learn) ----------------

class FAQIndex:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.items: List[FAQItem] = []
        self.id_to_item: Dict[int, FAQItem] = {}
        self.nn = None
        self._matrix = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False
        )
        return vecs.astype("float32")

    def build(self, faq_items: List[FAQItem]):
        self.items = faq_items
        self.id_to_item = {i: faq_items[i] for i in range(len(faq_items))}
        self._matrix = None
        self.nn = None
        if faq_items:
            self._matrix = self._encode([x.question for x in faq_items])
            self.nn = NearestNeighbors(n_neighbors=min(50, len(faq_items)), metric="cosine")
            self.nn.fit(self._matrix)
        if not faq_items:
            raise ValueError("faq_items must not be empty")
        self.items = faq_items   

    def search(self, query: str, top_k: int = 5) -> List[Tuple[FAQItem, float]]:
        if self.nn is None or not self.items:
            raise ValueError("Index not built or empty. Call build() with items before search().")
        q_vec = self._encode([query])
        top_k = min(top_k, len(self.items))  # Ensure top_k <= number of items
        dists, idxs = self.nn.kneighbors(q_vec, n_neighbors=top_k)
        sims = 1.0 - dists[0]
        return [(self.id_to_item[i], float(sim)) for i, sim in zip(idxs[0], sims)]

    def save(self, base_dir: str):
        if self._matrix is None or not self.items:
            raise ValueError("FAQIndex._matrix is None or empty. Did you call build() with items?")
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(base_dir, "faq_matrix.npy"), self._matrix)
        write_jsonl(os.path.join(base_dir, "faq_items.jsonl"), [x.model_dump() for x in self.items])
        with open(os.path.join(base_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"embed_model_name": self.model_name, "dimension": self.dim}, f)

    @classmethod
    def load(cls, base_dir: str) -> "FAQIndex":
        with open(os.path.join(base_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        inst = cls(model_name=cfg["embed_model_name"])
        rows = read_jsonl(os.path.join(base_dir, "faq_items.jsonl"))
        inst.items = [FAQItem(**r) for r in rows]
        inst.id_to_item = {i: inst.items[i] for i in range(len(inst.items))}
        inst._matrix = np.load(os.path.join(base_dir, "faq_matrix.npy"))
        inst.nn = None
        if inst.items:
            inst.nn = NearestNeighbors(n_neighbors=min(50, len(inst.items)), metric="cosine")
            inst.nn.fit(inst._matrix)
        return inst

# ---------------- LLM Helpers ----------------

class LLM:
    def __init__(self, model: str = "gpt2", device: int = -1, max_new_tokens: int = 256):
        self.pipe = pipeline("text-generation", model=model, device=device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=50256
        )[0]["generated_text"]
        return out[len(prompt):].strip()

def parse_jsonl_block(s: str) -> List[Dict[str, str]]:
    rows = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = ujson.loads(line)
            if "question" in obj and "answer" in obj:
                rows.append({"question": obj["question"], "answer": obj["answer"]})
        except ujson.JSONDecodeError:
            continue
    return rows

def parse_bullets(s: str) -> List[str]:
    out = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "•", "*")):
            line = line[1:].strip()
        out.append(line)
    return out

# ---------------- Build & Enrich FAQ ----------------

QA_EXTRACT_PROMPT = """Trích xuất tối đa 5 cặp Hỏi-Đáp quan trọng từ đoạn văn sau.
Xuất ở dạng JSONL với khóa: question, answer.

Đoạn:
{chunk}

Kết quả:
"""

PARAPHRASE_PROMPT = """Viết {n} cách hỏi khác nhau (ngắn gọn, tự nhiên, giữ nguyên ý) cho câu hỏi sau:
Q: {q}
Trả lời dạng gạch đầu dòng:
"""

def build_enriched_faq(
    doc_dir: str,
    initial_faq_path: Optional[str],
    out_dir: str,
    paraphrase_n: int = 5,
    llm_model: str = "gpt2",
    embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> Dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    seed_items: List[FAQItem] = []
    if initial_faq_path and Path(initial_faq_path).exists():
        for r in read_jsonl(initial_faq_path):
            cid = r.get("id") or str(uuid.uuid4())
            try:
                seed_items.append(FAQItem(
                    id=str(uuid.uuid4()),
                    question=r["question"],
                    answer=r["answer"],
                    canonical_id=cid,
                    provenance=Provenance(source_type="seed_faq", source_id=cid)
                ))
            except ValueError:
                continue  # Skip invalid seed items

    llm = LLM(model=llm_model)
    doc_items: List[FAQItem] = []
    for p in Path(doc_dir).glob("**/*"):
        if not p.is_file() or p.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            for ch in simple_chunks(text):
                gen = llm.generate(QA_EXTRACT_PROMPT.format(chunk=ch))
                qas = parse_jsonl_block(gen)
                for qa in qas:
                    try:
                        cid = str(uuid.uuid4())
                        doc_items.append(FAQItem(
                            id=str(uuid.uuid4()),
                            question=qa["question"].strip(),
                            answer=qa["answer"].strip(),
                            canonical_id=cid,
                            provenance=Provenance(source_type="doc_qa", source_id=cid, doc_path=str(p))
                        ))
                    except ValueError:
                        continue
        except Exception:
            continue  # Skip file read errors

    canonical_map: Dict[str, Tuple[str, str, Provenance]] = {}
    for it in seed_items + doc_items:
        if it.canonical_id not in canonical_map:
            canonical_map[it.canonical_id] = (it.question, it.answer, it.provenance)

    enriched_items: List[FAQItem] = []
    for cid, (root_q, root_a, prov) in canonical_map.items():
        enriched_items.append(FAQItem(
            id=str(uuid.uuid4()),
            question=root_q,
            answer=root_a,
            canonical_id=cid,
            provenance=prov
        ))
        try:
            variants = parse_bullets(LLM(model=llm_model).generate(
                PARAPHRASE_PROMPT.format(q=root_q, n=paraphrase_n),
                temperature=0.9
            ))
            for vq in variants[:paraphrase_n]:
                if not vq or len(vq) < 3:
                    continue
                enriched_items.append(FAQItem(
                    id=str(uuid.uuid4()),
                    question=vq,
                    answer=root_a,
                    canonical_id=cid,
                    provenance=Provenance(source_type="paraphrase", source_id=cid)
                ))
        except Exception:
            continue

    idx = FAQIndex(model_name=embed_model_name)
    idx.build(enriched_items)
    idx.save(out_dir)

    write_jsonl(os.path.join(out_dir, "canonical.jsonl"), [
        {"canonical_id": cid, "question": q, "answer": a, "provenance": prov.model_dump()}
        for cid, (q, a, prov) in canonical_map.items()
    ])

    return {
        "count_seed": len(seed_items),
        "count_doc": len(doc_items),
        "count_total_enriched": len(enriched_items),
        "unique_canonicals": len(canonical_map),
        "index_dir": out_dir,
        "embed_model": embed_model_name,
    }

# ---------------- Tier‑1 Runtime Agent ----------------

class Tier1FAQAgent:
    def __init__(
        self,
        index_dir: str,
        threshold_faq: float = 0.90,
        top_k: int = 20,
        fallback: Optional[Callable[[str], str]] = None,
        disclaimer: str = "Vui lòng xác minh thông tin tại nguồn chính thức nếu đây là thông tin quan trọng."
    ):
        self.index = FAQIndex.load(index_dir)
        self.threshold_faq = float(threshold_faq)
        self.top_k = max(1, int(top_k))  # Ensure top_k >= 1
        self.fallback = fallback
        self.disclaimer = disclaimer

    def answer(self, query: str) -> Dict[str, Any]:
        try:
            hits = self.index.search(query, top_k=self.top_k)
            if hits and hits[0][1] >= self.threshold_faq:
                top_item, score = hits[0]
                return {
                    "answer": top_item.answer,
                    "score": score,
                    "matched_question": top_item.question,
                    "canonical_id": top_item.canonical_id,
                    "provenance": top_item.provenance.model_dump(),
                    "escalated": False
                }
        except ValueError:
            pass  # Handle empty index or invalid search
        if self.fallback:
            generated = self.fallback(query)
            return {
                "answer": f"{generated}\n\n{self.disclaimer}",
                "score": None,
                "matched_question": None,
                "canonical_id": None,
                "provenance": {"source_type": "tier2"},
                "escalated": True
            }
        return {
            "answer": f"Xin lỗi, tôi chưa có câu trả lời trong FAQ cho câu hỏi này.\n{self.disclaimer}",
            "score": None,
            "matched_question": None,
            "canonical_id": None,
            "provenance": {"source_type": "no_match"},
            "escalated": True
        }

# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(description="Generative & enriched FAQ agent (no-FAISS, sklearn backend)")
    sub = parser.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="Build/enrich FAQ index from docs and seed FAQ")
    b.add_argument("--docs", type=str, required=True, help="Directory with .txt/.md documents")
    b.add_argument("--seed", type=str, default=None, help="Seed FAQ JSONL path (question, answer[, id])")
    b.add_argument("--out", type=str, required=True, help="Output directory for index artifacts")
    b.add_argument("--paraphrase-n", type=int, default=12, help="Number of paraphrases per canonical")
    b.add_argument("--llm-model", type=str, default="gpt2", help="HF model name for generation")
    b.add_argument("--embed-model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model")

    c = sub.add_parser("chat", help="Run interactive Tier‑1 FAQ agent")
    c.add_argument("--index", type=str, required=True, help="Directory with built FAQ index")
    c.add_argument("--threshold", type=float, default=0.90, help="Similarity threshold for accepting FAQ answer")
    c.add_argument("--topk", type=int, default=20, help="Top‑k to retrieve")

    args = parser.parse_args()

    if args.cmd == "build":
        stats = build_enriched_faq(
            doc_dir=args.docs,
            initial_faq_path=args.seed,
            out_dir=args.out,
            paraphrase_n=args.paraphrase_n,
            llm_model=args.llm_model,
            embed_model_name=args.embed_model
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    elif args.cmd == "chat":
        agent = Tier1FAQAgent(
            index_dir=args.index,
            threshold_faq=args.threshold,
            top_k=args.topk,
            fallback=None
        )
        print("FAQ agent ready. Type your question (or 'exit').")
        while True:
            try:
                q = input("\nBạn hỏi: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            ans = agent.answer(q)
            print("\n--- Trả lời ---")
            print(ans["answer"])
            print("\n--- Thông tin khớp ---")
            print(f"Matched Q: {ans.get('matched_question')}")
            print(f"Score: {ans.get('score')}")
            print(f"Provenance: {ans.get('provenance')}")
            print(f"Escalated: {ans.get('escalated')}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()