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

# Config explaination:
## Data Format for embedding process: 
- Documents folder to run with embed_to_db_doc: plain .txt, .pdf, .docx, .md, etc. Drop into data/docs/.
- CSV mode (special): csv with context or data field. For example:
    id,context
    1,"This is a paragraph about the policy..."
    2,"Another paragraph..."
## Advanced: Data format behind the scene:
- FAQ:
    {"question":"How to apply?", "answer":"Submit the form at ...", "canonical_id":"faq-001", "metadata":{"source":"policy.pdf"}}

# Code APIs:
## Meta Manager:
- What it does: Runs full pipelines, wires URagDManager (Docs) and FManager (FAQ), supports multiple inputs.
- Common methods
+ run(input_path, input_type) — input_type ∈ { "docs", "faqs" }
+ run_from_documents(documents: List[{"doc_id","text","metadata?"}])
+ run_from_faqs(roots: List[{"question","answer",...}])
+ run_from_csv_contexts(csv_path, context_col="context", id_col=None, min_len=5, default_source="csv_src")
- Example: docs folder → full pipeline
    from src.managers.meta_manager import MetaManager, MetaManagerConfig

    mm = MetaManager(MetaManagerConfig(
        collection_base="ura_rag_demo",
        language="default"
    ))
    res = mm.run(input_path="data/docs", input_type="docs")
    print(res)
- Example: FAQ JSON → enrich + index
    mm = MetaManager(MetaManagerConfig(collection_base="ura_rag_faq"))
    res = mm.run(input_path="data/faqs.json", input_type="faqs")
    print(res)
- Example: CSV contexts
    res = mm.run_from_csv_contexts(
        csv_path="datasets/sample.csv",
        context_col="context",
        id_col="id",
        min_len=5,
        default_source="csv_src"
    )
## URAG-D Manager:
- Key config (DManagerConfig):
+ root_dir, default_source, limit_docs
+ lang (chunking), buffer_size, min_chunk_size, number_of_chunks
+ emb_language, emb_model_name, emb_vi_model_name, emb_device, metric
+ milvus_collection_base, milvus_uri, milvus_token, shards_num
- Key methods:
+ run_pipeline(root_dir=None, collection_base=None) → index docs to __doc
+ run_pipeline_from_records(documents, collection_base=None) → in-memory
+ get_augmented_chunks() → pass to FManager later
- Example: custom config + augmentation
    from src.managers.urag_d_manager import DManagerConfig, URagDManager, ExistingTextGenerateAugmenter

    cfg = DManagerConfig(
        root_dir="data/docs",
        emb_model_name="BAAI/bge-m3",
        emb_language="default",     # or "vi"
        lang="default",
        milvus_collection_base="ura_doc_demo",
        metric="COSINE"
    )

    dm = URagDManager(cfg, augmenter=ExistingTextGenerateAugmenter())
    resp = dm.run_pipeline()
    print(resp)

    aug = dm.get_augmented_chunks()  # → feed FManager
## URAG-F Manager:
- Key config (FManagerConfig):
+ embed_field = "question" (default) or "answer"
+ default_source = "faq_src"
+ l2_normalize = True
- Common entrypoints
+ run_from_augmented(augmented, collection_base, paraphrase_n=5, metric="COSINE")
+ run_from_roots(roots, collection_base, paraphrase_n=5, metric="COSINE")
- Example: build FAQ from augmented chunks:
    from src.managers.urag_f_manager import FManager, FManagerConfig
    from src.managers.meta_manager import MetaManagerConfig, FAQGeneratorAdapter
    from src.embedding.embedding_agent import EmbedderAgent, EmbConfig
    embedder = EmbedderAgent(EmbConfig(model_name="BAAI/bge-m3", metric="COSINE"))
    fman = FManager(FAQGeneratorAdapter(), embedder, FManagerConfig())
    res = fman.run_from_augmented(
        augmented=aug,
        collection_base="ura_rag_demo",
        paraphrase_n=3,
        metric="COSINE"
    )
    print(res)
## LLM KERNEL:
- What it does: Central place to pick LLM provider without changing agent code.
- Typical usage (pseudocode – depends on your llm_kernel.py):
    from src.llm.llm_kernel import KERNEL, GoogleConfig, OpenAIConfig, OllamaConfig
    KERNEL.set_active_config(GoogleConfig(api_key=os.getenv("GOOGLE_API_KEY")))
## Embedder HAK:
    from src.embedding.embedding_agent import EmbedderAgent, EmbConfig

    emb = EmbedderAgent(EmbConfig(
        model_name="BAAI/bge-m3",
        language="default",
        device=None,               # "cuda" to force GPU
        normalize_for_cosine=True,
        metric="COSINE"
    ))
    vectors = emb.encode_texts(["hello world", "xin chào"])
    print(len(vectors), len(vectors[0]))
## Indexing Agent:
    from src.indexing.indexing_agent import IndexingAgent, AgentConfig, CreateCollectionReq, UpsertIndexReq, Item

    ia = IndexingAgent(AgentConfig(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN")))
    # ensure collection (__doc/__faq is handled internally if you pass dual=true in your implementation)
    ia.process(CreateCollectionReq(collection="my_base", dim=1024, metric_type="COSINE", shards_num=2))

    # upsert
    payload = UpsertIndexReq(
    op="upsert",
    collection="my_base",
    metric_type="COSINE",
    items=[
        Item(id="doc1__c1", type="doc", vector=[...], text="...", source="doc1", metadata={"chunk_id":"c1"}, ts=1720000000)
    ],
    shards_num=2,
    build_index=True
    )
    res = ia.process(payload.model_dump())
    print(res)
