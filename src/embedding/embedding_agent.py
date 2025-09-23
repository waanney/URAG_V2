# src/embedding/embedding_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

Metric = Literal["COSINE", "IP", "L2"]
Language = Literal["auto", "vi", "default"]  # "auto" dự phòng cho tương lai; hiện chọn cố định theo config

@dataclass
class EmbConfig:
    # Model mặc định (đa ngôn ngữ, dim=384)
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Model tiếng Việt chuyên dụng
    vi_model_name: str = "dangvantuan/vietnamese-embedding"

    # Chọn ngôn ngữ: "vi" -> dùng vi_model_name, "default" -> dùng model_name
    # (Không auto-switch trong runtime để tránh lệch dimension của Milvus collection)
    language: Language = "default"

    batch_size: int = 64
    device: Optional[str] = None  # 'cpu' | 'cuda' | None
    normalize_for_cosine: bool = True
    metric: Metric = "COSINE"

class EmbedderAgent:
    def __init__(self, cfg: EmbConfig):
        self.cfg = cfg

        # Chốt model theo language ngay từ đầu để đảm bảo dimension nhất quán
        chosen_model = (
            self.cfg.vi_model_name
            if self.cfg.language == "vi"
            else self.cfg.model_name
        )

        if self.cfg.device is None:
            self.model = SentenceTransformer(chosen_model)
        else:
            self.model = SentenceTransformer(chosen_model, device=self.cfg.device)

        # cache dim
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._encode(["_probe_"]).shape[1])
        return self._dim

    # ---------- core ----------
    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # tự normalize nếu cần
            show_progress_bar=False,
        ).astype("float32")
        if self.cfg.normalize_for_cosine and self.cfg.metric == "COSINE":
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vecs /= norms
        return vecs

    # ---------- public APIs ----------
    def embed_docs(
        self,
        docs: List[Dict[str, Any] | str],
        default_source: str = "doc_src",
        ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Input:
          - mỗi item có thể là str (chỉ text) hoặc dict {"text":..., ...meta}
        Output:
          - list[dict] không có id, gồm: {"type":"doc","text","vector","source","metadata","ts"}
        """
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []

        for it in docs:
            if isinstance(it, str):
                text = it
                meta: Dict[str, Any] = {}
            else:
                text = str(it.get("text", "")).strip()
                meta = {k: v for k, v in it.items() if k != "text"}

            if not text:
                continue

            texts.append(text)
            metas.append(meta)

        if not texts:
            return []

        vecs = self._encode(texts)
        out: List[Dict[str, Any]] = []
        for i, text in enumerate(texts):
            meta = metas[i]
            out.append({
                "type": "doc",
                "text": text,
                "vector": vecs[i].tolist(),
                "source": meta.get("source", default_source),
                "metadata": meta.get("metadata", {}),
                "ts": int(meta.get("ts", ts or 0)),
            })
        return out

    def embed_faqs(
        self,
        faqs: List[Dict[str, Any]],
        default_source: str = "faq_src",
        ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Input:
          - list[dict] có "question" và "answer"
        Output:
          - list[dict] không có id, gồm: {"type":"faq","question","answer","vector","source","metadata","ts"}
        """
        questions: List[str] = []
        metas: List[Tuple[str, Dict[str, Any]]] = []  # (answer, meta)

        for it in faqs:
            if not isinstance(it, dict):
                continue
            q = str(it.get("question", "")).strip()
            a = str(it.get("answer", "")).strip()
            if not q or not a:
                continue
            meta = {k: v for k, v in it.items() if k not in ("question", "answer")}
            questions.append(q)
            metas.append((a, meta))

        if not questions:
            return []

        vecs = self._encode(questions)
        out: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            a, meta = metas[i]
            out.append({
                "type": "faq",
                "question": q,
                "answer": a,
                "vector": vecs[i].tolist(),
                "source": meta.get("source", default_source),
                "metadata": meta.get("metadata", {}),
                "ts": int(meta.get("ts", ts or 0)),
            })
        return out
