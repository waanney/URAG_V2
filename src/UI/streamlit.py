
# streamlit_rag_control_panel.py
# -------------------------------------------------------------
# A Streamlit control panel to configure and drive an Agentic RAG system.
# - Customize provider, LLM model, temperature, streaming, etc.
# - Toggle embeddings, pick embedding model & vector DB, tune chunking.
# - Upload/inspect documents, simulate ingestion, start/stop system.
# - Export full config as JSON.
# -------------------------------------------------------------
# Usage:
#   1) pip install streamlit pandas requests
#   2) streamlit run streamlit_rag_control_panel.py
# -------------------------------------------------------------

import json
import time
from typing import Dict, List
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agentic RAG Control Panel", layout="wide")

# ----------------------------- Constants -----------------------------

PROVIDERS: Dict[str, List[str]] = {
    "Google": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "OpenAI": ["gpt-4o", "gpt-4.1-mini", "o4-mini", "gpt-4o-realtime"],
    "Anthropic": ["claude-3.5-sonnet", "claude-3.5-haiku"],
    "Ollama": ["llama3.1", "qwen2.5", "mistral-nemo", "phi4"],
}

EMBEDDING_MODELS = [
    ("text-embedding-3-large", "OpenAI text-embedding-3-large (3072d)"),
    ("bge-m3", "BAAI bge-m3 (1024d)"),
    ("e5-large-v2", "e5-large-v2 (1024d)"),
    ("gte-large", "gte-large (1024d)"),
    ("intfloat/multilingual-e5-large", "multilingual-e5-large"),
]

VDBS = [
    ("milvus", "Milvus"),
    ("pgvector", "Postgres (pgvector)"),
    ("qdrant", "Qdrant"),
]

# ------------------------- Session Initialization ---------------------

def ensure_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "catalog" not in st.session_state:
        st.session_state.catalog = []  # list of dicts: name, size, type, status
    if "provider" not in st.session_state:
        st.session_state.provider = "Google"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = PROVIDERS["Google"][0]
    if "use_embedding" not in st.session_state:
        st.session_state.use_embedding = True
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = EMBEDDING_MODELS[0][0]
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = VDBS[0][0]
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.4
    if "concurrency" not in st.session_state:
        st.session_state.concurrency = 4
    if "enable_streaming" not in st.session_state:
        st.session_state.enable_streaming = True
    if "enable_agentic" not in st.session_state:
        st.session_state.enable_agentic = True
    if "enable_autofaq" not in st.session_state:
        st.session_state.enable_autofaq = False
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1200
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 150
    if "detect_doc_type" not in st.session_state:
        st.session_state.detect_doc_type = True
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"
    if "provider_key" not in st.session_state:
        st.session_state.provider_key = ""
    if "ollama_url" not in st.session_state:
        st.session_state.ollama_url = "http://localhost:11434"
    if "test_query" not in st.session_state:
        st.session_state.test_query = ""

ensure_state()

def log(msg: str):
    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ------------------------------ Sidebar -------------------------------

with st.sidebar:
    st.markdown("### LLM Provider & Model")
    provider = st.selectbox("Provider", list(PROVIDERS.keys()), index=list(PROVIDERS.keys()).index(st.session_state.provider))
    if provider != st.session_state.provider:
        st.session_state.provider = provider
        st.session_state.llm_model = PROVIDERS[provider][0]

    llm_model = st.selectbox("LLM Model", PROVIDERS[st.session_state.provider], index=PROVIDERS[st.session_state.provider].index(st.session_state.llm_model))
    st.session_state.llm_model = llm_model

    st.slider("Temperature", 0.0, 1.0, key="temperature", step=0.05)
    st.slider("Concurrency", 1, 16, key="concurrency", step=1)
    st.checkbox("Enable streaming", key="enable_streaming")

    st.markdown("---")
    st.markdown("### Embeddings & Vector DB")
    st.checkbox("Use embedding index (recommended for RAG)", key="use_embedding")
    if st.session_state.use_embedding:
        labels = [label for _, label in EMBEDDING_MODELS]
        ids = [id for id, _ in EMBEDDING_MODELS]
        idx = ids.index(st.session_state.embedding_model) if st.session_state.embedding_model in ids else 0
        label = st.selectbox("Embedding Model", labels, index=idx)
        st.session_state.embedding_model = ids[labels.index(label)]

        v_labels = [label for _, label in VDBS]
        v_ids = [id for id, _ in VDBS]
        vidx = v_ids.index(st.session_state.vector_db) if st.session_state.vector_db in v_ids else 0
        vlabel = st.selectbox("Vector Database", v_labels, index=vidx)
        st.session_state.vector_db = v_ids[v_labels.index(vlabel)]

        st.number_input("Chunk size (tokens)", min_value=200, max_value=4000, key="chunk_size", step=50)
        st.number_input("Chunk overlap", min_value=0, max_value=1000, key="chunk_overlap", step=10)
        st.checkbox("Auto-detect document type (PDF/Word/Slides/Video…)", key="detect_doc_type")

    st.markdown("---")
    st.markdown("### Connectivity (Demo)")
    st.text_input("Backend Base URL", key="api_base_url")
    st.text_input(f"{st.session_state.provider} API Key (demo only)", key="provider_key", type="password", help="⚠️ Demo UI. In production, never expose keys on client; route via backend.")
    if st.session_state.provider == "Ollama":
        st.text_input("Ollama Base URL", key="ollama_url")

    st.checkbox("Enable Agentic Orchestrator (planner/tool use)", key="enable_agentic")
    st.checkbox("Auto-FAQ from documents", key="enable_autofaq")

# ------------------------------- Header --------------------------------

left, right = st.columns([0.65, 0.35])
with left:
    st.title("Agentic RAG Control Panel")
    st.caption("Streamlit UI — cấu hình model, ingest tài liệu, khởi động hệ thống, test truy vấn.")
with right:
    c1, c2 = st.columns(2)
    if not st.session_state.running:
        if c1.button("Start System", use_container_width=True):
            # TODO: requests.post(f"{api}/startup", json=payload)
            st.session_state.running = True
            log("Bootstrapping Agentic RAG… (demo)")
            time.sleep(0.3)
            log("System is up. You can send test queries.")
    else:
        if c1.button("Stop System", use_container_width=True, type="secondary"):
            # TODO: requests.post(f"{api}/shutdown")
            st.session_state.running = False
            log("Shut down agentic workers.")

    # Export config JSON
    config = {
        "provider": st.session_state.provider,
        "llmModel": st.session_state.llm_model,
        "useEmbedding": st.session_state.use_embedding,
        "embeddingModel": st.session_state.embedding_model if st.session_state.use_embedding else None,
        "vectorDb": st.session_state.vector_db if st.session_state.use_embedding else None,
        "agentic": {
            "enableAgentic": st.session_state.enable_agentic,
            "enableAutoFAQ": st.session_state.enable_autofaq,
            "enableStreaming": st.session_state.enable_streaming,
            "temperature": st.session_state.temperature,
            "concurrency": st.session_state.concurrency,
        },
        "chunking": {
            "chunkSize": st.session_state.chunk_size,
            "chunkOverlap": st.session_state.chunk_overlap,
            "detectDocType": st.session_state.detect_doc_type,
        },
        "connectivity": {
            "apiBaseUrl": st.session_state.api_base_url,
            "providerKey": "*** (present)" if st.session_state.provider_key else "",
            "ollamaUrl": st.session_state.ollama_url,
        },
    }
    c2.download_button(
        "Save Config JSON",
        data=json.dumps(config, indent=2),
        file_name="rag_agentic_config.json",
        mime="application/json",
        use_container_width=True,
    )

# ------------------------------ Ingestion -------------------------------

ing_col, stat_col = st.columns([0.55, 0.45])

with ing_col:
    st.subheader("Kho tài liệu (Upload/nhận diện)")
    files = st.file_uploader(
        "Kéo-thả hoặc chọn nhiều file (PDF, DOCX, PPTX, TXT, CSV, ảnh, video…)",
        accept_multiple_files=True,
    )
    add_clicked = st.button("Thêm vào kho (tạm)")
    if add_clicked and files:
        added = 0
        for f in files:
            st.session_state.catalog.append({
                "name": f.name,
                "sizeKB": round(f.size / 1024, 1),
                "type": f.type or "(unknown)",
                "status": "new",
            })
            added += 1
        log(f"Added {added} file(s) to catalog.")

    st.markdown("#### Danh sách tài liệu")
    if len(st.session_state.catalog) == 0:
        st.info("Chưa có tài liệu.")
    else:
        df = pd.DataFrame(st.session_state.catalog)
        st.dataframe(df, use_container_width=True, height=280)

    c_ing1, c_ing2 = st.columns([0.5, 0.5])
    if c_ing1.button("Ingest / Index (demo)"):
        if len(st.session_state.catalog) == 0:
            st.warning("No files to ingest.")
        else:
            log("Starting ingestion… (demo only)")
            for row in st.session_state.catalog:
                row["status"] = "ingesting"
            time.sleep(0.6)
            for row in st.session_state.catalog:
                row["status"] = "indexed"
            log(f"Ingestion finished. Indexed {len(st.session_state.catalog)} file(s).")

with stat_col:
    st.subheader("System Status")
    st.write(f"**Provider:** {st.session_state.provider}")
    st.write(f"**LLM:** {st.session_state.llm_model}")
    st.write(f"**Embeddings:** {st.session_state.embedding_model if st.session_state.use_embedding else '(disabled)'}")
    st.write(f"**Vector DB:** {st.session_state.vector_db if st.session_state.use_embedding else '—'}")
    st.write(f"**Agentic:** {'ON' if st.session_state.enable_agentic else 'OFF'} • **FAQ:** {'ON' if st.session_state.enable_autofaq else 'OFF'}")
    st.write(f"**Streaming:** {'ON' if st.session_state.enable_streaming else 'OFF'}")
    st.write(f"**Chunking:** {st.session_state.chunk_size}/{st.session_state.chunk_overlap} • Detect: {'Yes' if st.session_state.detect_doc_type else 'No'}")
    st.write(f"**System:** {'RUNNING' if st.session_state.running else 'STOPPED'}")

# ------------------------------ Test Query ------------------------------

st.subheader("Test Query (sau khi khởi động)")
q_col1, q_col2 = st.columns([0.8, 0.2])
q = q_col1.text_input("Nhập câu hỏi…", key="test_query", placeholder="Ví dụ: Quy chế tuyển sinh năm nay?")
send = q_col2.button("Gửi")
if send:
    if not st.session_state.running:
        log("System is not running.")
        st.warning("System is not running.")
    elif not st.session_state.test_query.strip():
        st.info("Hãy nhập câu hỏi.")
    else:
        # TODO: requests.post(f"{api}/query", json={"text": st.session_state.test_query})
        log(f"User: {st.session_state.test_query}")
        time.sleep(0.3)
        fake = "(demo) Đây là câu trả lời mẫu. Khi nối backend, thay bằng output thật từ LLM/RAG."
        st.success(fake)
        log(f"Assistant: {fake}")
        st.session_state.test_query = ""

# -------------------------------- Logs ---------------------------------

st.subheader("Logs")
if len(st.session_state.logs) == 0:
    st.code("(empty)")
else:
    st.code("\\n".join(st.session_state.logs), language="bash")

st.caption("Replace demo sections with real API calls: /ingest, /startup, /query, /stats")
