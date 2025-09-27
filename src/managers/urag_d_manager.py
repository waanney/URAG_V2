from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol, Literal, Union, cast, TypedDict, Set
import os, json, time

# ===== Doc Loader Agent (wrapper Pydantic) =====
from src.llm.URag_D.doc_loader_agent import (  # phiên bản bạn đã dùng
    DocumentLoaderAgent, DLConfig, DLRequest, DocRecord
)

# ===== Semantic Chunker Agent =====
from src.llm.URag_D.chunker_agent import (
    SemanticChunkerAgent, SemanticChunkerAgentConfig, SemanticChunkerRequest
)


# ===== Embedding Agent =====
try:
    from src.embedding.embedding_agent import EmbConfig, EmbedderAgent
except Exception:
    from embedding.embedding_agent import EmbConfig, EmbedderAgent

# ===== Indexing Agent (Milvus) =====
from src.indexing.indexing_agent import (
    IndexingAgent, AgentConfig as IndexCfg,
    CreateCollectionReq, UpsertIndexReq, Item
)

# ===================== Config & Models =====================

@dataclass
class DManagerConfig:
    # Data loading
    root_dir: str = "data/docs"
    default_source: str = "doc_src"
    limit_docs: Optional[int] = None

    # Chunker
    lang: Literal["default", "vi", "en"] = "default"
    buffer_size: int = 1
    min_chunk_size: Optional[int] = None
    number_of_chunks: Optional[int] = None

    # Embedding
    emb_language: Literal["default", "vi", "en"] = "default"
    emb_model_name: str = "BAAI/bge-m3"
    emb_vi_model_name: str = "dangvantuan/vietnamese-embedding"
    emb_device: Optional[str] = None
    metric: Literal["COSINE", "IP", "L2"] = "COSINE"

    # Indexing
    milvus_collection_base: str = "ura_rag_demo"
    milvus_uri: str = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    milvus_token: Optional[str] = os.getenv("MILVUS_TOKEN") or None
    shards_num: int = 2

class AugmentedItem(TypedDict):
    doc_id: str
    chunk_id: str
    original: str
    transformed: str
# ===================== Augmenter interface & adapters =====================

class IAugmenter(Protocol):
    def augment(self, doc_id: str, chunk_tuples: List[Tuple[str, str]]) -> List[AugmentedItem]:
        """
        chunk_tuples: list of (chunk_id, text) preserving order.
        Returns: list of dicts {doc_id, chunk_id, original, transformed} in the same order.
        """
        ...


class ExistingTextGenerateAugmenter(IAugmenter):
    """
    Adapter dùng chính agents trong textGenerate.py của bạn (pydantic-ai).
    Kỳ vọng trong file có:
      - worker1, worker2: Agent(...)
      - MyDeps: dataclass có trường listChunks: List[str]
      - checking_output(raw: str) -> str: strip code-fences
    """
    def __init__(self) -> None:
        try:
            from src.llm.URag_D.textGenerate import worker1, worker2, MyDeps, checking_output
        except Exception:
            from llm.URag_D.textGenerate import worker1, worker2, MyDeps, checking_output

        self.worker1 = worker1
        self.worker2 = worker2
        self.MyDeps = MyDeps
        self.clean = checking_output

    def _run_worker(self, chunks: List[str], which: int = 1) -> List[Dict[str, str]]:
        deps = self.MyDeps(listChunks=chunks)
        # dùng worker1 mặc định; nếu which=2 sẽ gọi worker2 (nếu cần)
        agent = self.worker2 if (which == 2 and self.worker2 is not None) else self.worker1
        res = agent.run_sync(deps=deps)
        raw = self.clean(res.output)
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Augmenter output must be a JSON list.")
        # mỗi phần tử phải có 'original','transformed'
        out: List[Dict[str, str]] = []
        for x in data:
            o = (x.get("original") or "").strip()
            t = (x.get("transformed") or "").strip()
            out.append({"original": o, "transformed": t})
        return out

    def augment(self, doc_id: str, chunk_tuples: List[Tuple[str, str]]) -> List[AugmentedItem]:
        if not chunk_tuples:
            return []
        texts: List[str] = [t for (_, t) in chunk_tuples]

        # logic tương tự file mẫu của bạn: chia làm 2 phần gọi 2 worker
        mid = len(texts) // 2
        outs: List[Dict[str, str]] = []
        if mid > 0:
            outs.extend(self._run_worker(texts[:mid], which=1))
            outs.extend(self._run_worker(texts[mid:], which=2))
        else:
            outs.extend(self._run_worker(texts, which=1))

        # map theo thứ tự (zip) về augmented item
        augmented: List[AugmentedItem] = []
        for (chunk_id, original), pair in zip([(cid, txt) for (cid, txt) in chunk_tuples], outs):
            transformed = pair.get("transformed") or ""
            transformed = transformed.strip() or original  # fallback giữ nguyên
            augmented.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "original": original,
                "transformed": transformed
            })
        return augmented


class NoOpAugmenter(IAugmenter):
    """Không dùng LLM — giữ y nguyên chunk."""
    def augment(self, doc_id: str, chunk_tuples: List[Tuple[str, str]]) -> List[AugmentedItem]:
        return [
            {"doc_id": doc_id, "chunk_id": cid, "original": txt, "transformed": txt}
            for (cid, txt) in chunk_tuples
        ]


# ===================== D-Manager =====================

class URagDManager:
    """
    Điều phối D-pipeline: load -> chunk -> augment -> embed -> index
    """

    def __init__(self, cfg: Optional[DManagerConfig] = None, augmenter: Optional[IAugmenter] = None):
        self.cfg = cfg or DManagerConfig()

        # 1) Doc loader
        self.doc_loader = DocumentLoaderAgent(DLConfig(autodetect_encoding=True))

        # 2) Chunker
        self.chunker = SemanticChunkerAgent(
            SemanticChunkerAgentConfig(
                language=self.cfg.lang,
                buffer_size=self.cfg.buffer_size,
                min_chunk_size=self.cfg.min_chunk_size,
                number_of_chunks=self.cfg.number_of_chunks,
                use_agent_embedder=True
            )
        )

        # 3) Augmenter — dùng lại textGenerate.py (hoặc NoOp)
        self.augmenter: IAugmenter = augmenter or ExistingTextGenerateAugmenter()

        # 4) Embedder
        self.embedder = EmbedderAgent(EmbConfig(
            model_name=self.cfg.emb_model_name,
            vi_model_name=self.cfg.emb_vi_model_name,
            language=("vi" if self.cfg.emb_language == "vi" else "default"),
            device=self.cfg.emb_device,
            normalize_for_cosine=True,
            metric=self.cfg.metric,
        ))

        # 5) Indexer
        self.indexer = IndexingAgent(IndexCfg(
            uri=self.cfg.milvus_uri,
            token=self.cfg.milvus_token,
            dual_collections=True,          # __doc / __faq
            normalize_l2_for_cosine=True
        ))

        # runtime storage
        self._augmented: List[AugmentedItem] = []
        self._last_dim: Optional[int] = None

    # ---------- Public getters ----------
    def get_augmented_chunks(self) -> List[AugmentedItem]:
        """Trả augmentedChunks (đưa sang F-Manager)."""
        return list(self._augmented)

    # ---------- Steps ----------
    def load_documents(self) -> List[DocRecord]:
        req = DLRequest(
            mode="normal",
            root_dir=self.cfg.root_dir,
            default_source=self.cfg.default_source,
            limit_docs=self.cfg.limit_docs
        )
        resp = self.doc_loader.run(req)
        return resp.documents  # List[DocRecord]

    def chunk_documents(self, docs: List[DocRecord]) -> List[Tuple[str, str, str]]:
        """
        Return: list of (doc_id, chunk_id, chunk_text)
        """
        out: List[Tuple[str, str, str]] = []
        for d in docs:
            resp = self.chunker.chunk(SemanticChunkerRequest(text=d.text, doc_id=d.doc_id))
            for c in resp.chunks:
                out.append((c.doc_id, c.chunk_id, c.text))
        return out

    def augment_chunks(self, triples: List[Tuple[str, str, str]]) -> List[AugmentedItem]:
        """
        triples: (doc_id, chunk_id, text)
        """
        if not triples:
            self._augmented = []
            return []

        # nhóm theo doc_id (để quản lý prompt/giới hạn)
        by_doc: Dict[str, List[Tuple[str, str]]] = {}
        order_doc: List[str] = []
        for doc_id, chunk_id, text in triples:
            if doc_id not in by_doc:
                order_doc.append(doc_id)
                by_doc[doc_id] = []
            by_doc[doc_id].append((chunk_id, text))

        augmented: List[AugmentedItem] = []
        for doc_id in order_doc:
            chunk_tuples = by_doc[doc_id]
            # Adapter sẽ tự chia batch theo logic có sẵn (worker1/worker2)
            augmented.extend(self.augmenter.augment(doc_id, chunk_tuples))

        self._augmented = augmented
        return augmented

    def embed_and_index(self, collection_base: Optional[str] = None) -> Dict[str, Any]:
        """
        Embed transformed text & upsert vào Milvus __doc
        """
        if not self._augmented:
            return {"status": "empty", "message": "No augmented chunks to index."}

        base = collection_base or self.cfg.milvus_collection_base
        now_ts = int(time.time())

        # 4) Embed
        emb_inputs: List[Dict[str, Any]] = []
        for a in self._augmented:
            headline = a.get("headline", "")
            emb_inputs.append({
                "text": a["transformed"],
                "source": a["doc_id"],
                "metadata": {"chunk_id": a["chunk_id"],
                              "original": a["original"],
                              "headline": headline},
                "ts": now_ts
            })

        embedded_docs = self.embedder.embed_docs(
            cast(List[Union[Dict[str, Any], str]], emb_inputs),  # cast để Pylance im lặng
            default_source="doc_src",
            ts=now_ts
        )
        dim = self.embedder.dim
        self._last_dim = dim

        # 5) Ensure collections (__doc/__faq)
        _ = self.indexer.process(CreateCollectionReq(
            collection=base, dim=dim, metric_type="COSINE", shards_num=self.cfg.shards_num
        ))

        # Upsert vào __doc
        items: List[Item] = []
        for i, d in enumerate(embedded_docs):
            chunk_id = emb_inputs[i]["metadata"]["chunk_id"]
            id_join = f"{d['source']}__{chunk_id}"
            items.append(Item(
                id=id_join,
                type="doc",
                vector=d["vector"],
                text=d["text"],
                source=d["source"],
                metadata=d["metadata"],
                ts=d["ts"]
            ))

        res = self.indexer.process(UpsertIndexReq(
            op="upsert",
            collection=base,
            metric_type="COSINE",
            items=items,
            shards_num=self.cfg.shards_num,
            build_index=True
        ))
        return res

    def run_pipeline(self, root_dir: Optional[str] = None, collection_base: Optional[str] = None) -> Dict[str, Any]:
        """One-shot: load -> chunk -> augment -> embed -> index"""
        if root_dir:
            self.cfg.root_dir = root_dir
        docs = self.load_documents()
        triples = self.chunk_documents(docs)
        self.augment_chunks(triples)
        return self.embed_and_index(collection_base or self.cfg.milvus_collection_base)

    def run_pipeline_from_records(
        self,
        documents: List[Dict[str, Any]],
        collection_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        One-shot (records): [{doc_id, text, metadata}] -> chunk -> augment -> embed -> index.
        Trả về phản hồi indexing của __doc.
        """
        if not documents:
            return {"status": "empty", "message": "No documents provided."}

        # 1) Chuẩn hoá đầu vào (khoan tạo DocRecord; chunker chỉ cần text, doc_id)
        norm_docs: List[Tuple[str, str, Dict[str, Any]]] = []  # (doc_id, text, metadata)
        for i, d in enumerate(documents):
            # d có thể là pydantic DocRecord (.doc_id/.text) hoặc dict
            doc_id = (getattr(d, "doc_id", None) or d.get("doc_id") or f"rec_{i}")
            text   = (getattr(d, "text",   None) or d.get("text") or "").strip()
            meta   = (getattr(d, "metadata", None) or d.get("metadata") or {})
            if not text:
                continue
            norm_docs.append((str(doc_id), text, dict(meta)))

        if not norm_docs:
            return {"status": "empty", "message": "All documents were empty."}

        # (tuỳ chọn) cảnh báo trùng doc_id
        seen: Set[str] = set()
        dedup_docs: List[Tuple[str, str, Dict[str, Any]]] = []
        for i, (doc_id, text, meta) in enumerate(norm_docs):
            if doc_id in seen:
                # đảm bảo uniqueness để id_join=doc_id__chunk_id không đè nhau
                doc_id = f"{doc_id}__dup{i}"
                meta = {**meta, "dedup": True}
            seen.add(doc_id)
            dedup_docs.append((doc_id, text, meta))
        norm_docs = dedup_docs

        # 2) Chunk
        triples: List[Tuple[str, str, str]] = []  # (doc_id, chunk_id, text)
        for doc_id, full_text, _meta in norm_docs:
            resp = self.chunker.chunk(SemanticChunkerRequest(text=full_text, doc_id=doc_id))
            for c in resp.chunks:
                triples.append((c.doc_id, c.chunk_id, c.text))

        # 3) Augment
        self.augment_chunks(triples)

        # 4) Embed + Index vào __doc
        return self.embed_and_index(collection_base or self.cfg.milvus_collection_base)
# ===================== Demo =====================
if __name__ == "__main__":
    cfg = DManagerConfig(
        root_dir="data/docs",
        milvus_collection_base=f"ura_dmgr_demo_{int(time.time())}",
        emb_language="vi",
        lang="vi"
    )
    dmgr = URagDManager(cfg, augmenter=ExistingTextGenerateAugmenter())
    out = dmgr.run_pipeline()
    print(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n[augmentedChunks sample]")
    print(json.dumps(dmgr.get_augmented_chunks()[:5], ensure_ascii=False, indent=2))
