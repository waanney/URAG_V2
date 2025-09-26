# -*- coding: utf-8 -*-
"""
DocumentLoaderAgent — Pydantic wrapper cho DocLoaderLC (không dùng LLM)
- Giữ nguyên logic đọc file trong DocLoaderLC, chỉ thêm:
  + Config/Request/Response bằng Pydantic
  + Validate input, gom log/meta
  + API run() trả schema rõ ràng
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import os
# Tái dùng loader hiện có của bạn
try:
    from src.llm.URag_D.document_loader import DocLoaderLC, DocLoaderConfig
except Exception:
    from llm.URag_D.document_loader import DocLoaderLC, DocLoaderConfig

# ---------- Pydantic IO Schemas ----------

class DLConfig(BaseModel):
    pdf_glob: str = "**/*.pdf"
    docx_glob: str = "**/*.docx"
    txt_glob:  str = "**/*.txt"
    md_glob:   str = "**/*.md"
    csv_glob:  str = "**/*.csv"

    csv_source_column: Optional[str] = None
    csv_delimiter: str = ","

    use_unstructured: bool = False
    other_glob: Optional[str] = None

    autodetect_encoding: bool = True

    def to_dc(self) -> DocLoaderConfig:
        return DocLoaderConfig(
            pdf_glob=self.pdf_glob,
            docx_glob=self.docx_glob,
            txt_glob=self.txt_glob,
            md_glob=self.md_glob,
            csv_glob=self.csv_glob,
            csv_source_column=self.csv_source_column,
            csv_delimiter=self.csv_delimiter,
            use_unstructured=self.use_unstructured,
            other_glob=self.other_glob,
            autodetect_encoding=self.autodetect_encoding,
        )

class DocRecord(BaseModel):
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FaqItem(BaseModel):
    question: str
    answer: str
    canonical_id: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    ts: Optional[int] = None

class DLRequest(BaseModel):
    mode: Literal["normal", "faq", "both", "csv"] = "normal"
    root_dir: Optional[str] = Field(default=None, description="Thư mục gốc chứa tài liệu thường")
    faq_path: Optional[str] = Field(default=None, description="Đường dẫn file JSON/JSONL FAQ")
    default_source: str = "doc_src"
    limit_docs: Optional[int] = None  # cắt bớt để test nhanh

    @field_validator("root_dir")
    @classmethod
    def _check_root(cls, v, info):
        mode = info.data.get("mode", "normal")
        if mode in ("normal", "both"):
            if not v:
                raise ValueError("root_dir is required when mode in ['normal', 'both'].")
            if not Path(v).exists():
                raise ValueError(f"root_dir not found: {v}")
        return v

    @field_validator("faq_path")
    @classmethod
    def _check_faq(cls, v, info):
        mode = info.data.get("mode", "normal")
        if mode in ("faq", "both"):
            if not v:
                raise ValueError("faq_path is required when mode in ['faq', 'both'].")
            if not Path(v).exists():
                raise ValueError(f"faq_path not found: {v}")
        return v

class DLResponse(BaseModel):
    documents: List[DocRecord] = Field(default_factory=list)
    faqs: List[FaqItem] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    # tiện cho các chỗ cũ cần list[dict]
    def docs_legacy(self) -> List[Dict[str, Any]]:
        return [d.model_dump() for d in self.documents]

# ---------- Agent Wrapper (no LLM involved) ----------

class DocumentLoaderAgent:
    def __init__(self, cfg: Optional[DLConfig] = None):
        self.cfg = cfg or DLConfig()
        self._loader = DocLoaderLC(self.cfg.to_dc())

    def run(self, req: DLRequest) -> DLResponse:
        documents: List[DocRecord] = []
        faqs: List[FaqItem] = []
        meta: Dict[str, Any] = {"mode": req.mode}

        if req.mode == "csv":
            raise ValueError("mode='csv' không dùng ở run(); hãy gọi load_context_csv_as_docs(...)")

        if req.mode in ("normal", "both"):
            docs_lc = self._loader.load_normal_docs(req.root_dir or ".")
            aug = self._loader.to_augmented_inputs(docs_lc, default_source=req.default_source)
            
            if req.limit_docs is not None and req.limit_docs >= 0:
                aug = aug[: req.limit_docs]
            documents = [DocRecord(**x) for x in aug]
            meta["num_documents"] = len(documents)

        if req.mode in ("faq", "both"):
            rows = self._loader.load_faq_json(req.faq_path or "")
            faqs = [FaqItem(**r) for r in rows]
            meta["num_faqs"] = len(faqs)

        return DLResponse(documents=documents, faqs=faqs, meta=meta)
        
    def load_context_csv_as_docs(
        self,
        csv_path: str,
        context_col: str = "context",
        id_col: Optional[str] = None,
        min_len: int = 5,
        default_source: str = "csv_src",
    ) -> DLResponse:
        """
        Đọc 1 CSV, lấy đúng cột `context` làm nội dung tài liệu.
        Mỗi dòng -> 1 DocRecord(text=context, doc_id sinh từ id_col hoặc hash).
        """
        docs_lc = self._loader.load_csv_contexts(
            csv_path=csv_path,
            context_col=context_col,
            id_col=id_col,
            source_col=None,    
            min_len=min_len,
        )
        aug = self._loader.to_augmented_inputs(docs_lc, default_source=default_source)
        for x in aug:
            x["doc_id"] = x.get("metadata", {}).get("doc_id", x["doc_id"])
        documents = [DocRecord(**x) for x in aug]
        meta = {
            "mode": "csv-context",
            "csv_path": os.fspath(csv_path),
            "num_documents": len(documents),
            "context_col": context_col,
        }
        return DLResponse(documents=documents, faqs=[], meta=meta)

    # --- NEW (tuỳ chọn): chỉ lấy list[DocRecord] cho tiện tái dùng ---
    def load_context_csv_records(
        self,
        csv_path: str,
        context_col: str = "context",
        id_col: Optional[str] = None,
        min_len: int = 5,
        default_source: str = "csv_src",
    ) -> List[DocRecord]:
        docs_lc = self._loader.load_csv_contexts(
            csv_path=csv_path,
            context_col=context_col,
            id_col=id_col,
            source_col=None,
            min_len=min_len,
        )
        aug = self._loader.to_augmented_inputs(docs_lc, default_source=default_source)
        for x in aug:
            x["doc_id"] = x.get("metadata", {}).get("doc_id", x["doc_id"])
        return [DocRecord(**x) for x in aug]