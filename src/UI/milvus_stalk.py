# -*- coding: utf-8 -*-
# Streamlit Milvus Inspector — READ-ONLY dashboard
#
# Tính năng:
# 1) Kết nối Milvus (URI/TOKEN)
# 2) Liệt kê collections
# 3) Xem chi tiết 1 collection: schema, index, num_entities
# 4) Duyệt mẫu (query limit N) an toàn theo schema (DOC/FAQ)
# 5) Search nhanh bằng EmbedderAgent (nhập text -> embed -> search)
#
# Yêu cầu:
#   pip/uv install streamlit pymilvus sentence-transformers torch (nếu dùng GPU)
#   và project của bạn trong PYTHONPATH (để import EmbedderAgent/IndexingAgent)
#
# Chạy:
#   uv run streamlit run src/UI/milvus_stalk.py
#
import sys
from pathlib import Path

# .../URAG_V2/src/UI/milvus_stalk.py -> project root = .../URAG_V2
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import math
import streamlit as st
from typing import Any, Dict, List, Optional, cast

from pymilvus import connections, utility, Collection
# Một số phiên bản pymilvus dùng SearchFuture cho search async
try:
    # path tùy version; khối try/except giúp Pylance hiểu được type
    from pymilvus.orm.future import SearchFuture  # type: ignore
except Exception:
    class SearchFuture:  # fallback để type-checking không báo lỗi
        pass

# ---- import từ codebase của bạn ----
from src.embedding.embedding_agent import EmbedderAgent, EmbConfig, Language
from src.indexing.indexing_agent import AgentConfig as IdxCfg  # chỉ để lấy default kết nối




# ----------------------------- helpers -----------------------------
def safe_connect(uri: str, token: Optional[str], alias: str = "default") -> bool:
    try:
        kwargs: Dict[str, Any] = {"alias": alias, "uri": uri}
        if token:
            kwargs["token"] = token
        connections.connect(**kwargs)
        return True
    except Exception as e:
        st.error(f"Không kết nối được Milvus: {e}")
        return False


def get_schema_dict(col: Collection) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in col.schema.fields:
        pd = getattr(f, "params", {}) or {}
        rows.append({
            "name": f.name,
            "dtype": f.dtype,
            "is_primary": getattr(f, "is_primary", False),
            "auto_id": getattr(f, "auto_id", False),
            "max_length": getattr(f, "max_length", None),
            "params": pd
        })
    return rows


def get_index_info(col: Collection) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        for idx in (col.indexes or []):
            out.append({
                "field": idx.field_name,
                "index_type": idx.params.get("index_type"),
                "metric_type": idx.params.get("metric_type"),
                "params": {k: v for k, v in idx.params.items() if k not in ("index_type", "metric_type")}
            })
    except Exception as e:
        st.warning(f"Không đọc được index info: {e}")
    return out


def get_vector_dim(col: Collection) -> int:
    for f in col.schema.fields:
        if f.name == "vector":
            return int(f.params["dim"])
    raise RuntimeError("Không tìm thấy trường vector")


def l2_normalize(v: List[float]) -> List[float]:
    s = sum(x * x for x in v)
    if s <= 0:
        return v
    inv = 1.0 / math.sqrt(s)
    return [x * inv for x in v]


def guess_is_faq(col: Collection) -> bool:
    names = {f.name for f in col.schema.fields}
    return "question" in names and "answer" in names


def sample_rows(col: Collection, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Lấy vài bản ghi để xem. Ta cố gắng dùng expr an toàn:
    - Nếu có 'ts' -> expr = "ts >= 0"
    - Nếu không, nếu có 'id' -> expr = "id like '%'"
    """
    names = {f.name for f in col.schema.fields}
    if "ts" in names:
        expr = "ts >= 0"
    elif "id" in names:
        expr = "id like '%'"
    else:
        raise RuntimeError("Không có field phù hợp để query")

    # chọn output_fields phù hợp
    if guess_is_faq(col):
        out_fields = [f for f in ["id", "question", "answer", "source", "metadata", "ts"] if f in names]
    else:
        out_fields = [f for f in ["id", "text", "source", "metadata", "ts"] if f in names]

    try:
        rows = col.query(expr=expr, output_fields=out_fields, limit=limit)
        # parse metadata JSON nếu cần
        for r in rows:
            if isinstance(r.get("metadata"), str):
                try:
                    r["metadata"] = json.loads(r["metadata"])
                except Exception:
                    pass
        return rows
    except Exception as e:
        st.warning(f"Query mẫu lỗi: {e}")
        return []


def do_search(col: Collection, query_vec: List[float], top_k: int = 5, metric: str = "COSINE",
              output_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    dim = get_vector_dim(col)
    if len(query_vec) != dim:
        return {"error": f"Dim không khớp: collection={dim}, query={len(query_vec)}"}

    # đoán search params theo index
    sp: Dict[str, Any] = {}
    idx_type: Optional[str] = None
    try:
        if col.indexes:
            idx_type = col.indexes[0].params.get("index_type")
        if idx_type == "HNSW":
            sp = {"ef": 64}
        elif idx_type and idx_type.startswith("IVF"):
            sp = {"nprobe": 16}
    except Exception:
        pass

    # normalize nếu cosine
    qv = query_vec[:]
    if metric.upper() == "COSINE":
        qv = l2_normalize(qv)

    names = {f.name for f in col.schema.fields}
    if not output_fields:
        if guess_is_faq(col):
            output_fields = [x for x in ["id", "question", "answer", "source", "metadata"] if x in names]
        else:
            output_fields = [x for x in ["id", "text", "source", "metadata"] if x in names]

    try:
        results: Any = col.search(
            data=[qv], anns_field="vector", param=sp, limit=top_k,
            output_fields=output_fields, consistency_level="Strong"
        )

        # Nếu đối tượng có .result() (async/future), gọi để lấy kết quả thực
        if hasattr(results, "result"):
            try:
                results = results.result()
            except Exception:
                # Nếu .result() không khả dụng thực sự, cứ để nguyên (một số bản trả list luôn)
                pass

        hits_list = results[0]
        out: List[Dict[str, Any]] = []
        for h in hits_list:
            row: Dict[str, Any] = {"id": h.entity.get("id"), "distance": float(h.distance)}
            for f in (output_fields or []):
                try:
                    row[f] = h.entity.get(f)
                except Exception:
                    pass
            if isinstance(row.get("metadata"), str):
                try:
                    row["metadata"] = json.loads(row["metadata"])
                except Exception:
                    pass
            out.append(row)
        return {"index_type": idx_type, "search_params": sp, "results": out}
    except Exception as e:
        return {"error": str(e)}


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Milvus Inspector (URAG-V2)", layout="wide")

st.title("🔎 Milvus Inspector — URAG-V2 (read-only)")
st.caption("Xem nhanh collections, schema, index, đếm entities, duyệt mẫu & search thử.")

with st.sidebar:
    st.header("Kết nối")
    default_uri = IdxCfg().uri  # lấy default từ AgentConfig của bạn
    uri = st.text_input(
        "MILVUS URI",
        value=default_uri,
        help="VD: http://127.0.0.1:19530 hoặc milvus+https://... nếu dùng Milvus Cloud"
    )
    token_env = os.getenv("MILVUS_TOKEN")
    token = st.text_input("Token (nếu có)", type="password", value=(token_env or ""))

    if st.button("🔌 Connect"):
        ok = safe_connect(uri, (token or None))
        st.session_state["connected"] = ok

    connected = st.session_state.get("connected", False)
    st.markdown("---")
    st.header("Embedder (Search)")
    model_name = st.text_input("Embedding model", value="BAAI/bge-m3")
    language_str = st.selectbox("Language", ["default", "vi", "auto"], index=0)
    top_k = st.number_input("Top-K", min_value=1, max_value=50, value=5, step=1)

if not st.session_state.get("connected"):
    st.info("Hãy kết nối Milvus ở thanh bên trái để bắt đầu.")
    st.stop()

# Health
ok = connections.has_connection("default")
st.success("Đã kết nối Milvus.") if ok else st.error("Kết nối không ổn.")

# List collections
cols = utility.list_collections()
st.subheader(f"📚 Collections ({len(cols)})")
st.write(cols if cols else "Không có collection nào.")

if not cols:
    st.stop()

# luôn chọn index=0 để tránh None
sel = st.selectbox("Chọn collection để xem chi tiết", options=cols, index=0)
if not sel:
    st.stop()

col = Collection(cast(str, sel))
col.load()

# Summary row
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Entities", value=f"{col.num_entities:,}")
with c2:
    try:
        st.metric("Dim", value=get_vector_dim(col))
    except Exception:
        st.metric("Dim", value="—")
with c3:
    st.metric("Type", value=("FAQ" if guess_is_faq(col) else "DOC"))

# Schema & Index
st.markdown("### 🧱 Schema")
st.dataframe(get_schema_dict(col), use_container_width=True)

st.markdown("### 🧮 Index")
idx_info = get_index_info(col)
if idx_info:
    st.json(idx_info)
else:
    st.write("Chưa có index (Milvus sẽ search theo flat).")

# Browse sample
st.markdown("### 👀 Duyệt mẫu")
limit = st.slider("Số bản ghi", 1, 100, 20)
rows = sample_rows(col, limit=limit)
st.write(rows if rows else "Không lấy được bản ghi nào (thử tăng limit hoặc kiểm tra expr).")

# Search quick test
st.markdown("### 🔍 Search nhanh (text → embed → search)")
query = st.text_area("Nhập câu để tìm", placeholder="Ví dụ: Điểm sàn tuyển sinh là bao nhiêu?")

if st.button("Search"):
    try:
        # Ép kiểu Literal cho Pylance (Language = Literal["auto","vi","default"])
        lang_val: Language = cast(Language, language_str)

        embedder = EmbedderAgent(EmbConfig(model_name=model_name, language=lang_val))
        dim_embed = embedder.info()["dim"]
        vec = embedder.encode([query])[0]

        # Nếu dim lệch với collection, cảnh báo rõ ràng
        dim_col = get_vector_dim(col)
        if dim_embed != dim_col:
            st.error(
                f"Dim mismatch: embedder={dim_embed}, collection={dim_col}. "
                f"Đổi model hoặc chọn collection khác cho khớp."
            )
        else:
            out = do_search(col, vec, top_k=top_k, metric="COSINE")
            if "error" in out:
                st.error(out["error"])
            else:
                st.write("Search params:", {"index_type": out["index_type"], **out["search_params"]})
                st.json(out["results"])
    except Exception as e:
        st.error(f"Lỗi search: {e}")
