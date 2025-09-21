# ====== deps ======
# pip install pymilvus sentence-transformers pydantic-ai numpy

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- LLM agents (pydantic-ai) ----------
from pydantic_ai import Agent, models
llm = models.OpenAIChat(id="gpt-4o-mini")  # đổi theo backend bạn dùng

chunker = Agent(model=llm, system_prompt=(
    "Split long documents into cohesive chunks (~120–200 words). "
    "Return JSON list [{chunk_id, text}]. Strict JSON."
))
augmenter = Agent(model=llm, system_prompt=(
    "Rewrite the chunk for clarity while preserving facts. "
    "Return JSON {generated_text}."
))
faq_gen = Agent(model=llm, system_prompt=(
    "Generate concise, grounded FAQ Q/A from the corpus. "
    "Return JSON list [{q, a}]."
))
faq_enricher = Agent(model=llm, system_prompt=(
    "For each Q/A, produce K diverse paraphrased questions keeping the same answer. "
    "Return JSON list [{q}]."
))

# ---------- Domain models ----------
class OriginalDocument(BaseModel):
    id: str
    text: str
    extra: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str

class AugmentedDoc(BaseModel):
    doc_id: str
    chunk_id: str
    generated_text: str

class QAPair(BaseModel):
    doc_id: str
    q: str
    a: str
    root_id: Optional[str] = None

# ---------- Embedding ----------
class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._dim: Optional[int] = None

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.encode(["_probe_"]).shape[1]
        return self._dim

# ---------- Milvus vector DB adapters ----------
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

class MilvusVectorDB:
    """
    Generic Milvus wrapper with simple upsert/search.
    Create separate instances for 'document_index' and 'faq_index'
    with different extra_fields.
    """
    def __init__(
        self,
        collection: str,
        dim: int,
        uri: str = "http://localhost:19530",
        token: Optional[str] = None,
        metric_type: str = "COSINE",     # or "IP", "L2"
        index_type: str = "HNSW",        # or "IVF_FLAT", "IVF_SQ8", "DISKANN"
        extra_fields: List[str] = None,  # list of VarChar fields to store metadata
        text_field: str = "text",
        answer_field: Optional[str] = None,  # only for FAQ index
        max_text_len: int = 8192
    ):
        self.collection_name = collection
        self.dim = dim
        self.metric_type = metric_type.upper()
        self.index_type = index_type.upper()
        self.extra_fields = extra_fields or []
        self.text_field = text_field
        self.answer_field = answer_field

        # connect
        connections.connect("default", uri=uri, token=token)

        # ensure collection
        self.col = self._ensure_collection(max_text_len=max_text_len)

        # ensure index
        self._ensure_index()

        # load to memory for search
        self.col.load()

    def _ensure_collection(self, max_text_len: int) -> Collection:
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=max_text_len),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            for f in self.extra_fields:
                fields.append(FieldSchema(name=f, dtype=DataType.VARCHAR, max_length=1024))
            if self.answer_field:
                fields.append(FieldSchema(name=self.answer_field, dtype=DataType.VARCHAR, max_length=max_text_len))

            schema = CollectionSchema(fields=fields, description=f"{self.collection_name} schema")
            col = Collection(name=self.collection_name, schema=schema)
        else:
            col = Collection(name=self.collection_name)
        return col

    def _ensure_index(self):
        if not self.col.indexes:
            if self.index_type == "HNSW":
                index_params = {"index_type": "HNSW", "metric_type": self.metric_type,
                                "params": {"M": 32, "efConstruction": 200}}
                self.search_params = {"params": {"ef": 128}, "metric_type": self.metric_type}
            elif self.index_type == "IVF_FLAT":
                index_params = {"index_type": "IVF_FLAT", "metric_type": self.metric_type,
                                "params": {"nlist": 1024}}
                self.search_params = {"params": {"nprobe": 16}, "metric_type": self.metric_type}
            else:
                # fallback
                index_params = {"index_type": self.index_type, "metric_type": self.metric_type,
                                "params": {}}
                self.search_params = {"params": {}, "metric_type": self.metric_type}

            self.col.create_index(field_name="embedding", index_params=index_params)

    def upsert(self, texts: List[str], embs: np.ndarray, metas: List[Dict[str, Any]]):
        """
        Insert rows. metas must contain keys in self.extra_fields (+ answer_field if set).
        """
        assert len(texts) == len(embs) == len(metas)
        # Build column data in schema order (excluding auto id)
        cols = [
            texts,  # text field
            embs.tolist(),  # embeddings
        ]
        for f in self.extra_fields:
            cols.append([str(m.get(f, "")) for m in metas])
        if self.answer_field:
            cols.append([str(m.get(self.answer_field, "")) for m in metas])

        # Reorder columns to match schema order (id auto -> skip)
        ordered = []
        for fs in self.col.schema.fields:
            if fs.name == "id":
                continue
            if fs.name == self.text_field:
                ordered.append(cols[0])
            elif fs.name == "embedding":
                ordered.append(cols[1])
            elif fs.name in self.extra_fields:
                idx = 2 + self.extra_fields.index(fs.name)
                ordered.append(cols[idx])
            elif self.answer_field and fs.name == self.answer_field:
                ordered.append(cols[-1])

        self.col.insert(ordered)
        self.col.flush()

    def search(self, query_emb: np.ndarray, k: int = 5, filter_expr: Optional[str] = None):
        if query_emb.ndim == 1:
            query_emb = query_emb[np.newaxis, :]
        res = self.col.search(
            data=query_emb.tolist(),
            anns_field="embedding",
            param=self.search_params,
            limit=k,
            expr=filter_expr,
            output_fields=[self.text_field] + self.extra_fields + ([self.answer_field] if self.answer_field else [])
        )
        hits = []
        for hit in res[0]:
            rec = {f: hit.entity.get(f) for f in hit.entity.fields if f != "embedding"}
            # Milvus returns distance; convert to similarity if using COSINE
            score = 1.0 - float(hit.distance) if self.metric_type == "COSINE" else float(hit.distance)
            rec["score"] = score
            hits.append(rec)
        return hits

# ---------- Orchestrator (URAG-D + URAG-F) ----------
class URAGPipeline:
    def __init__(self, embedder: Embedder, milvus_uri: str, milvus_token: Optional[str] = None):
        self.embedder = embedder
        # document index: store doc_id, chunk_id
        self.doc_index = MilvusVectorDB(
            collection="document_index",
            dim=self.embedder.dim,
            uri=milvus_uri,
            token=milvus_token,
            metric_type="COSINE",
            index_type="HNSW",
            extra_fields=["doc_id", "chunk_id"]
        )
        # faq index: store doc_id, root_id, and the answer text
        self.faq_index = MilvusVectorDB(
            collection="faq_index",
            dim=self.embedder.dim,
            uri=milvus_uri,
            token=milvus_token,
            metric_type="COSINE",
            index_type="HNSW",
            extra_fields=["doc_id", "root_id"],
            answer_field="answer"  # store A for convenience
        )

    async def urag_d(self, docs: List[OriginalDocument]):
        # 1) Chunk
        chunks: List[Chunk] = []
        for d in docs:
            r = await chunker.run(f"Document ID: {d.id}\nText:\n{d.text}\nReturn JSON.")
            for i, it in enumerate(r.data):
                chunks.append(Chunk(doc_id=d.id, chunk_id=f"{d.id}_{i+1}", text=it["text"]))

        # 2) Augment
        augmented: List[AugmentedDoc] = []
        for ch in chunks:
            r = await augmenter.run(f"Chunk (doc_id={ch.doc_id}, chunk_id={ch.chunk_id}):\n{ch.text}\nReturn JSON.")
            augmented.append(AugmentedDoc(doc_id=ch.doc_id, chunk_id=ch.chunk_id,
                                          generated_text=r.data["generated_text"]))

        # 3) Embed + upsert to Milvus
        texts = [a.generated_text for a in augmented]
        embs = self.embedder.encode(texts)
        metas = [{"doc_id": a.doc_id, "chunk_id": a.chunk_id} for a in augmented]
        self.doc_index.upsert(texts, embs, metas)

    async def urag_f(self, docs: List[OriginalDocument], paraphrase_k: int = 3):
        # 1) Generate base FAQ from corpus
        merged = "\n\n".join([f"[{d.id}]\n{d.text}" for d in docs])
        r = await faq_gen.run(f"Corpus:\n{merged}\nReturn JSON.")
        base = [QAPair(doc_id="*corpus*", q=qa["q"], a=qa["a"], root_id=f"root_{i+1}") for i, qa in enumerate(r.data)]

        # 2) Enrich/paraphrase
        enriched: List[QAPair] = []
        for p in base:
            rr = await faq_enricher.run(
                f"Q: {p.q}\nA: {p.a}\nParaphrase the question into {paraphrase_k} variants. Return JSON list of {{q}}."
            )
            enriched.append(p)  # keep root
            for v in rr.data:
                enriched.append(QAPair(doc_id=p.doc_id, q=v["q"], a=p.a, root_id=p.root_id))

        # 3) Only embed QUESTIONS -> upsert to Milvus FAQ index
        questions = [x.q for x in enriched]
        embs = self.embedder.encode(questions)
        metas = [{"doc_id": x.doc_id, "root_id": x.root_id, "answer": x.a} for x in enriched]
        self.faq_index.upsert(questions, embs, metas)

    # --- Search APIs ---
    def search_docs(self, query: str, k: int = 5):
        emb = self.embedder.encode([query])[0]
        return self.doc_index.search(emb, k=k)

    def search_faq(self, question: str, k: int = 5):
        emb = self.embedder.encode([question])[0]
        return self.faq_index.search(emb, k=k)

# ---------- Usage example ----------
"""
import asyncio, os

async def main():
    docs = [
        OriginalDocument(id="D1", text="(Your original document text 1...)"),
        OriginalDocument(id="D2", text="(Your original document text 2...)"),
    ]

    pipe = URAGPipeline(
        embedder=Embedder("all-MiniLM-L6-v2"),          # hoặc SBERT tiếng Việt
        milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        milvus_token=os.getenv("MILVUS_TOKEN")          # Zilliz Cloud: 'db_admin:xxxx'
    )

    await pipe.urag_d(docs)                 # Red flow → document_index
    await pipe.urag_f(docs, paraphrase_k=3) # Blue flow → faq_index

    print("DOC SEARCH:", pipe.search_docs("regulation about student card", k=3))
    print("FAQ SEARCH:", pipe.search_faq("How to get a student card?", k=3))

asyncio.run(main())
"""
