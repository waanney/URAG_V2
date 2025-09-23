# examples/embedding_test/embedding_test_1.py
from __future__ import annotations
import os, json, time, random, string
from pathlib import Path

# 👉 chỉnh lại import này theo nơi bạn đặt file embedder_agent.py
# ví dụ: from src.embedding.embedder_agent import EmbedderAgent, EmbConfig
from src.llm.embedding_agent import EmbedderAgent, EmbConfig

def rand_suffix(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

# === cấu hình embedder (COSINE + normalize để hợp với Milvus COSINE) ===
cfg = EmbConfig(
    model_name=os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    metric="COSINE",
    normalize_for_cosine=True,
    batch_size=64,
)
agent = EmbedderAgent(cfg)

print("== EMBEDDER AGENT INFO ==")
print(f"model: {cfg.model_name}")
print(f"dim:   {agent.dim}")
print()

# --------- dữ liệu mẫu ----------
docs_input = [
    # có thể truyền list[str]
    "Trường ĐH Bách Khoa có 2 cơ sở. Cơ sở 1 ở quận 10, cơ sở 2 ở Thủ Đức.",
    # hoặc list[dict] có 'text' + metadata tuỳ ý (KHÔNG có id)
    {
        "text": "Thời hạn đóng học phí là trước ngày 15 hàng tháng.",
        "source": "handbook.md",
        "metadata": {"sensitivity": "critical"}
    },
]

faqs_input = [
    {"question": "BK có bao nhiêu cơ sở?", "answer": "HCMUT có 2 cơ sở (Q10 và Thủ Đức)."},
    {"question": "Hạn chót đóng học phí khi nào?", "answer": "Trước ngày 15 hàng tháng."},
]

# --------- embed ---------
doc_with_vec = agent.embed_docs(docs_input, default_source="doc_src")
faq_with_vec = agent.embed_faqs(faqs_input, default_source="faq_src")

# --------- in tóm tắt kết quả (ẩn bớt vector cho gọn) ---------
def preview(item, k=6):
    v = item.get("vector", [])[:k]
    show = {**{k: v for k, v in item.items() if k != "vector"}}
    show["vector_preview"] = v
    show["vector_dim"] = len(item.get("vector", []))
    return show

print("== DOC EMBEDS ==")
for i, it in enumerate(doc_with_vec, 1):
    print(json.dumps(preview(it), ensure_ascii=False, indent=2))

print("\n== FAQ EMBEDS ==")
for i, it in enumerate(faq_with_vec, 1):
    print(json.dumps(preview(it), ensure_ascii=False, indent=2))

# --------- lưu JSON để feed cho IndexingAgent (không có id) ---------
out_dir = Path("examples/embedding_test/tmp_embeds")
out_dir.mkdir(parents=True, exist_ok=True)
suffix = f"{int(time.time())}_{rand_suffix()}"
doc_out = out_dir / f"docs_with_vec_{suffix}.json"
faq_out = out_dir / f"faqs_with_vec_{suffix}.json"

with open(doc_out, "w", encoding="utf-8") as f:
    json.dump(doc_with_vec, f, ensure_ascii=False, indent=2)
with open(faq_out, "w", encoding="utf-8") as f:
    json.dump(faq_with_vec, f, ensure_ascii=False, indent=2)

print(f"\n== WROTE FILES ==")
print(f"- {doc_out}")
print(f"- {faq_out}")

# Gợi ý: ở bước tiếp theo, bạn có thể:
# 1) map mỗi item -> thêm 'id' (ví dụ uuid4), 
# 2) gửi từng nhóm sang IndexingAgent:
#    - DOC:  collection = <base_name>__doc, items: [{id, type='doc', text, vector, source, metadata, ts}]
#    - FAQ:  collection = <base_name>__faq, items: [{id, type='faq', question, answer, vector, source, metadata, ts}]
# 3) giữ nguyên metric COSINE để khớp normalize ở đây.
