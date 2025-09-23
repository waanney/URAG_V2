# src/llm/URag_D/document_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable
from pathlib import Path
import json
import pandas as pd

# LangChain Document type (compatible across versions)
try:
    from langchain_core.documents import Document  # LC >= 0.2
except Exception:  # pragma: no cover
    from langchain.schema import Document  # older LC

# Individual loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader, # MỚI THÊM
)

# Optional Unstructured
try:  # pragma: no cover
    from langchain_community.document_loaders import UnstructuredFileLoader  # type: ignore
    HAS_UNSTRUCTURED = True
except Exception:  # pragma: no cover
    UnstructuredFileLoader = None  # type: ignore[assignment]
    HAS_UNSTRUCTURED = False


@dataclass
class DocLoaderConfig:
    pdf_glob: str = "**/*.pdf"
    docx_glob: str = "**/*.docx"
    txt_glob:  str = "**/*.txt"
    md_glob:   str = "**/*.md"
    csv_glob: str = "**/*.csv"  # MỚI THÊM

    # Cấu hình cho CSV Loader
    csv_source_column: Optional[str] = None # MỚI THÊM: Tên cột làm metadata 'source'

    # Optional: load everything else via Unstructured (if installed)
    use_unstructured: bool = False
    other_glob: Optional[str] = None  # e.g. "**/*.html"

    # TextLoader options
    autodetect_encoding: bool = True


class DocLoaderLC:
    """Document reader dùng loader riêng cho từng định dạng (tránh lỗi type của DirectoryLoader)."""

    def __init__(self, cfg: Optional[DocLoaderConfig] = None):
        self.cfg = cfg or DocLoaderConfig()

    def load_normal_docs(self, root_dir: str) -> List[Document]:
        """
        Đọc tất cả tài liệu trong thư mục theo glob đã cấu hình.
        Trả về list[Document] (mỗi doc có page_content + metadata).
        """
        root = Path(root_dir)
        if not root.exists():
            return []

        docs: List[Document] = []
        docs += self._load_by_glob(root, self.cfg.pdf_glob, loader="pdf")
        docs += self._load_by_glob(root, self.cfg.docx_glob, loader="docx")
        docs += self._load_by_glob(root, self.cfg.txt_glob, loader="txt")
        docs += self._load_by_glob(root, self.cfg.md_glob, loader="md")
        docs += self._load_by_glob(root, self.cfg.csv_glob, loader="csv") # MỚI THÊM

        if self.cfg.use_unstructured and self.cfg.other_glob and HAS_UNSTRUCTURED:
            docs += self._load_by_glob(root, self.cfg.other_glob, loader="unstructured")

        # Chuẩn hoá metadata
        for d in docs:
            d.metadata = d.metadata or {}
            if "source" not in d.metadata:
                d.metadata["source"] = d.metadata.get("file_path") or d.metadata.get("path")
            if "path" not in d.metadata:
                d.metadata["path"] = d.metadata.get("file_path") or d.metadata.get("source")
        return docs

    def _load_by_glob(self, root: Path, pattern: str, loader: str) -> List[Document]:
        out: List[Document] = []
        for p in root.rglob(pattern):
            if not p.is_file():
                continue
            try:
                if loader == "pdf":
                    ld = PyPDFLoader(str(p))
                elif loader == "docx":
                    ld = Docx2txtLoader(str(p))
                elif loader in ("txt", "md"):
                    ld = TextLoader(str(p), autodetect_encoding=self.cfg.autodetect_encoding)
                elif loader == "csv": # MỚI THÊM
                    ld = CSVLoader(
                        file_path=str(p),
                        source_column=self.cfg.csv_source_column,
                        encoding="utf-8"
                    )
                elif loader == "unstructured":
                    if not HAS_UNSTRUCTURED or UnstructuredFileLoader is None:
                        continue
                    ld = UnstructuredFileLoader(str(p))  # type: ignore[call-arg]
                else:
                    continue

                loaded = ld.load()
                # Gắn file_path để chuẩn hoá về sau
                for d in loaded:
                    d.metadata = d.metadata or {}
                    d.metadata.setdefault("file_path", str(p))
                out.extend(loaded)
            except Exception:
                # skip lỗi đọc từng file
                continue
        return out

    # -------- FAQ docs (JSON/JSONL) --------
    def load_faq_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Đọc FAQ từ .json hoặc .jsonl.
        Mỗi dòng/record kỳ vọng: {"question": "...", "answer": "...", ...optional meta...}
        """
        p = Path(file_path)
        if not p.exists():
            return []

        out: List[Dict[str, Any]] = []
        if p.suffix.lower() in {".jsonl", ".jsonl.txt"}:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                            out.append(obj)
                    except Exception:
                        continue
        else:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                            out.append(obj)
                elif isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                    for obj in data["items"]:
                        if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                            out.append(obj)
            except Exception:
                pass
        return out

    # -------- Helpers: chuẩn hoá về “augmented input” --------
    def to_augmented_inputs(
        self,
        documents: List[Document],
        default_source: str = "doc_src"
    ) -> List[Dict[str, Any]]:
        """
        Chuẩn hoá list[Document] -> list[dict] {doc_id, text, metadata} để đẩy qua semantic chunker.
        """
        out: List[Dict[str, Any]] = []
        for d in documents:
            text = (d.page_content or "").strip()
            if not text:
                continue
            meta = dict(d.metadata or {})
            src = meta.get("source") or meta.get("path") or default_source
            doc_id = str(src)
            out.append({
                "doc_id": doc_id,
                "text": text,
                "metadata": meta
            })
        return out