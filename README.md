# URAG V2
Unified Retrieval-Augmented Generation v2 — a modular, two-tier Agentic RAG architecture with a Hybrid Doc-FAQ pipeline:
- Document pipeline: Load → Semantic Chunking → (LLM) Text Augmentation → Embedding → Indexing (__doc)
- FAQ pipeline: FAQ Generation → Paraphrasing → Embedding → Indexing (__faq)
Comes with 2 managers for 2 tasks, with a meta manager controls the workflow with multiple AI Agent.

# Key Features:
- 🧭 Two-tier Milvus indexing: separate __doc and __faq collections.
- 🧠 Semantic chunking synchronized with the embedder — minimizes embedding mismatch.
- 📝 FAQ paraphrasing with answer preservation — boosts recall without corrupting answers.
- 🔌 LLM Kernel with runtime selection of Google Gemini, OpenAI, or Ollama.
- 📂 Supports multiple data sources: document folders, CSV, or FAQ JSON/JSONL.
- 🧰 Uses uv for fast, reliable Python environment management.

# Requirements:
- Python 3.10+
- uv (package manager)
- Docker + Docker Compose (for Milvus)
- GPU (optional, for faster embedding and LLM inference)

# Setting up the environment:
## Setup with uv:
    uv venv
    uv sync
    (To activate the venv, use "source .venv/bin/activate" for macOS/Linux or .venv\Scripts\Activate.ps1 for Windows)

## Setup Milvus database for Windows:
    (Install Docker Desktop)
    (Mở docker desktop)
    docker compose up -d (để chạy file docker)
    
## Create .env file, with the following properties:
    # Choose one of the following:
    GOOGLE_API_KEY=...
    OPENAI_BASE_URL="https://api.openai.com/v1"
    OPENAI_API_KEY=...
    OLLAMA_BASE_URL=http://localhost:11434

## Run - Embed to DB:
Turn on MilvusDB, using "docker compose up -d"
Run the examples/experiment/embed_to_db.py
    uv run python -m examples.experiment.embed_to_db

## Run - Inference:
To run inference, run the examples/experiment/inference.py
    uv run python -m examples.experiment.inference

Code flow: Under the src/
## The Embedding process
- Indexing Agent: Indexing data and send into MilvusDB (src/indexing/indexing_agent.py)
- Embedding HAK (Hybrid Agent-Kernel): Acts as a embedding agent, also a kernel for embedding services (src/embedding/embedding_agen.py)
- LLM Kernel: Provide all LLM services (src/llm/llm_kernel.py)
- Meta Manager: Controls the Embed-To-DB flow, includes URAG-D manager and URAG-F manager (src/managers/meta_manager.py)
- URAG-D: The flow of process the documents into chunks -> augmented chunks, which then be used for embedding and later in URAG-F for faq-generating.
    + Document Loader
    + Chunker Agent
    + Text Generating Agent (used for augmented chunks generation)
    + D Manager: Controls the flow, serve as connecting
- URAG-F: The flow of generating FAQ, then enrich FAQ with multiple ways of asking.
    + FAQ Agent
    + F Manager: Controls the flow, serve as connecting
## The Inference process
- Searching Agent: Search the DB, is related to inference step.
- Notes: The searching agent may lack of data-loading types, can be fix by adding more functions into src/search/search_agent.py

## Config explaination:
- 