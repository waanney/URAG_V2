# src/llm/URag_D/document_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv
import hashlib

from langchain_core.documents import Document  # LC >= 0.2

# Individual loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

# CSV loader (optional across versions)
_HAS_CSV = True
try:
    from langchain_community.document_loaders import CSVLoader
except Exception:  # pragma: no cover
    CSVLoader = None  # type: ignore[assignment]
    _HAS_CSV = False

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
    csv_glob:  str = "**/*.csv"

    # CSV options (for CSVLoader)
    csv_source_column: Optional[str] = None
    csv_delimiter: str = ","  # used via csv_args for better compatibility

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
        docs += self._load_by_glob(root, self.cfg.csv_glob, loader="csv")

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
                elif loader == "csv":
                    if _HAS_CSV and CSVLoader is not None:
                        # dùng csv_args để tương thích nhiều version
                        ld = CSVLoader(
                            file_path=str(p),
                            csv_args={"delimiter": self.cfg.csv_delimiter},
                            source_column=self.cfg.csv_source_column,
                        )
                    else:
                        # Fallback: đọc nhị phân & coi như text
                        ld = TextLoader(str(p), autodetect_encoding=self.cfg.autodetect_encoding)
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
                    except Exception:
                        continue
                    item = self._coerce_faq_row(obj)
                    if item:
                        out.append(item)
        else:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for obj in data:
                        item = self._coerce_faq_row(obj)
                        if item:
                            out.append(item)
                elif isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                    for obj in data["items"]:
                        item = self._coerce_faq_row(obj)
                        if item:
                            out.append(item)
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

    # -------- private: normalize FAQ row --------
    @staticmethod
    def _coerce_faq_row(row: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(row, dict):
            return None
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if not q or not a:
            return None
        out: Dict[str, Any] = {"question": q, "answer": a}
        if row.get("canonical_id"):
            out["canonical_id"] = str(row["canonical_id"])
        if row.get("source"):
            out["source"] = str(row["source"])
        meta = row.get("metadata")
        if isinstance(meta, dict):
            out["metadata"] = meta
        if row.get("ts") is not None:
            try:
                out["ts"] = int(row["ts"])
            except Exception:
                pass
        return out
    
    def load_csv_contexts(
        self,
        csv_path: str,
        context_col: Optional[str] = None,
        id_col: Optional[str] = None,
        source_col: Optional[str] = None,
        min_len: int = 5,
    ) -> List[Document]:
        """
        Đọc 1 CSV, lấy mỗi dòng một 'context' -> Document-like:
        - context_col: tên cột chứa ngữ cảnh; nếu None, auto-detect: ['context','content','text','body','passage']
        - id_col: nếu có, dùng làm doc_id (ghi vào metadata['doc_id'])
        - source_col: nếu có, ghi vào metadata['source']
        """
        p = Path(csv_path)
        if not p.exists():
            return []

        # Ưu tiên delimiter cấu hình; nếu không, sniff
        delim = self.cfg.csv_delimiter or ","
        try:
            with p.open("r", encoding="utf-8", newline="") as fh:
                sample = fh.read(4096)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delim = dialect.delimiter or delim
        except Exception:
            pass

        out: List[Document] = []
        with p.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            headers = reader.fieldnames or []

            candidates = ["context", "content", "text", "body", "passage"]
            def _pick_col(want: Optional[str], cands: List[str]) -> Optional[str]:
                if want and want in headers:
                    return want
                low = {h.lower(): h for h in headers}
                for c in cands:
                    if c.lower() in low:
                        return low[c.lower()]
                return None

            ctx_col = _pick_col(context_col, candidates)
            if not ctx_col:
                # fallback: nếu không có cột hợp lệ, trả rỗng
                return []

            for idx, row in enumerate(reader, start=1):
                ctx = (row.get(ctx_col) or "").strip()
                if len(ctx) < min_len:
                    continue

                doc_id = None
                if id_col and id_col in row and row[id_col]:
                    doc_id = str(row[id_col]).strip()
                else:
                    h = hashlib.md5()
                    h.update(f"{csv_path}|{idx}|{ctx[:64]}".encode("utf-8", errors="ignore"))
                    doc_id = f"csv_{h.hexdigest()[:16]}"

                meta: Dict[str, Any] = {
                    "file_path": str(p),
                    "path": str(p),
                    "row_index": idx,
                    "context_col": ctx_col,
                    "delimiter": delim,
                    "doc_id": doc_id,
                }
                if source_col and source_col in row:
                    meta["source"] = (row.get(source_col) or "").strip()

                # Khởi tạo Document thật (nếu LangChain có), vẫn hợp type DocLike
                doc = Document(page_content=ctx, metadata=meta)  # type: ignore[call-arg]
                out.append(doc)
        return out

    