# examples/f_manager_test.py
import time, json, random, string
from typing import Any, Dict, List, Optional
from pymilvus import utility

from src.managers.urag_f_manager import FManager, FManagerConfig, AugmentedChunk, IFaqGenerator, IEmbedder
from src.indexing.indexing_agent import IndexingAgent
from src.embedding.embedding_agent import EmbedderAgent, EmbConfig  

# ===== Adapters =====
class EmbedderAdapter(IEmbedder):
    """Adapter khớp Protocol IEmbedder, dùng EmbedderAgent của bạn."""
    def __init__(self, impl: EmbedderAgent):
        self.impl = impl

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        # Dùng hàm embed_faqs/doc hay _encode đều được. Ở đây dùng _encode để trả đúng kiểu List[List[float]].
        # (Nếu không thích gọi private, bạn có thể: [x["vector"] for x in impl.embed_faqs([{"question":t,"answer":"-"}])]
        return self.impl._encode(texts).tolist()  # type: ignore[attr-defined]

    def info(self) -> Dict[str, Any]:
        return {"model": self.impl.cfg.model_name, "dim": self.impl.dim}

class DummyFaqGenerator(IFaqGenerator):
    def generate_roots(self, augmented_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        roots: List[Dict[str, Any]] = []
        for i, ch in enumerate(augmented_chunks, start=1):
            t = ch["text"]
            if "cơ sở" in t:
                roots.append({
                    "question": "Trường có bao nhiêu cơ sở?",
                    "answer": "Có 2 cơ sở (Q10 và Thủ Đức).",
                    "canonical_id": f"can_{i}_campus",
                    "source": "auto_root",
                })
            if "học phí" in t:
                roots.append({
                    "question": "Hạn chót đóng học phí khi nào?",
                    "answer": "Trước ngày 15 hàng tháng.",
                    "canonical_id": f"can_{i}_tuition",
                    "source": "auto_root",
                })
        return roots

    def enrich_from_roots(self, roots: List[Dict[str, Any]], paraphrase_n: int = 3) -> List[Dict[str, Any]]:
        all_items: List[Dict[str, Any]] = []
        for r in roots:
            all_items.append({**r, "is_paraphrase": False})
            qs = (
                ["Bao nhiêu cơ sở vậy?", "Trường có mấy cơ sở?", "Cơ sở của trường ở đâu?"]
                if "cơ sở" in r["question"]
                else ["Đóng học phí hạn cuối lúc nào?", "Khi nào phải nộp học phí?", "Deadline đóng học phí là ngày nào?"]
            )
            for q in qs[:paraphrase_n]:
                all_items.append({
                    "question": q,
                    "answer": r["answer"],
                    "canonical_id": r.get("canonical_id"),
                    "source": "auto_para",
                    "is_paraphrase": True,
                })
        return all_items

def rand_suffix(n=6) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

# ===== Test flow =====
# 1) Embedder + FManager
emb = EmbedderAgent(EmbConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))
embedder = EmbedderAdapter(emb)
faq_gen = DummyFaqGenerator()
mgr = FManager(faq_generator=faq_gen, embedder=embedder, cfg=FManagerConfig(embed_field="question"))

# 2) Augmented chunks mẫu
chunks = [
    AugmentedChunk(doc_id="D1", chunk_id="1", text="Trường có 2 cơ sở, cơ sở 1 ở quận 10 và cơ sở 2 ở Thủ Đức."),
    AugmentedChunk(doc_id="D2", chunk_id="1", text="Thời hạn đóng học phí là trước ngày 15 hàng tháng."),
]

# 3) Generate + Embed
gen_out = mgr.generate(chunks, paraphrase_n=2)
faqs_with_vec = mgr.embed(gen_out["faqs"])
if not faqs_with_vec:
    raise SystemExit("Không tạo được FAQ nào để test.")

dim = len(faqs_with_vec[0].vector)

# 4) Tạo collections + upsert
agent = IndexingAgent()
base = f"ura_faq_test_{int(time.time())}_{rand_suffix()}"
print("== CREATE TWO COLLECTIONS ==")
agent.process_and_print({
    "op": "create_collection",
    "collection": base,
    "dim": dim,
    "metric_type": "COSINE"
})

payload = mgr.make_index_payload(collection_base=base, items_with_vec=faqs_with_vec)
print("\n== UPSERT FAQ ==")
agent.process_and_print(payload)

# 5) Search thử
faq_coll = f"{base}__faq"
print("\n== SEARCH IN FAQ COLLECTION ==")
print(agent.process_to_json({
    "op": "search",
    "collection": faq_coll,
    "search_vector": faqs_with_vec[0].vector,
    "top_k": 3,
    "metric_type": "COSINE",
    "output_fields": ["id","question","answer","source","metadata"]
}, pretty=True))

# 6) Stats
print("\n== STATS FAQ ==")
agent.process_and_print({"op":"stats", "collection":faq_coll})

# 7) Cleanup
doc_coll = f"{base}__doc"
if utility.has_collection(doc_coll):
    utility.drop_collection(doc_coll)
    print(f"[cleanup] dropped collection {doc_coll}")
if utility.has_collection(faq_coll):
    utility.drop_collection(faq_coll)
    print(f"[cleanup] dropped collection {faq_coll}")
