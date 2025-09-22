# examples/indexing_test_dual_collections.py
"""
Run directly (no main()) to test IndexingAgent with TWO separate Milvus collections:
  - <base>__doc  (stores docs)
  - <base>__faq  (stores FAQs)
Requires a running Milvus (docker compose up -d) and pymilvus installed.
"""

import time, random, string
from pymilvus import utility
from src.indexing.indexing_agent import IndexingAgent, CreateCollectionReq

def rand_suffix(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

# --- Setup ---
agent = IndexingAgent()  # dual_collections should be enabled inside the agent
base = f"ura_rag_dual_{int(time.time())}_{rand_suffix()}"
coll_doc = f"{base}__doc"
coll_faq = f"{base}__faq"

try:
    # 1) Create base -> agent should create <base>__doc and <base>__faq
    print("== CREATE TWO COLLECTIONS ==")
    agent.process_and_print(CreateCollectionReq(
        collection=base, dim=4, metric_type="COSINE"
    ))

    assert utility.has_collection(coll_doc), f"missing {coll_doc}"
    assert utility.has_collection(coll_faq), f"missing {coll_faq}"

    # 2) Upsert mixed items (agent routes by type)
    print("\n== UPSERT MIXED (DOC + FAQ) ==")
    agent.process_and_print({
        "op": "index",
        "collection": base,              # pass BASE; agent routes to __doc / __faq
        "metric_type": "COSINE",
        "items": [
            # DOCs
            {"id":"doc#A","type":"doc","vector":[0.3,0.1,0.2,0.4],"text":"đoạn tài liệu A","source":"doc_src"},
            {"id":"doc#B","type":"doc","vector":[0.4,0.2,0.1,0.3],"text":"đoạn tài liệu B","source":"doc_src"},
            # FAQs (embed QUESTION)
            {"id":"faq#1","type":"faq","vector":[0.29,0.11,0.19,0.41],
             "question":"BK có bao nhiêu cơ sở?","answer":"HCMUT có 2 cơ sở.","source":"faq_src"},
            {"id":"faq#2","type":"faq","vector":[0.42,0.19,0.11,0.28],
             "question":"Học phí khoảng bao nhiêu?","answer":"~30–40 triệu/năm tuỳ chương trình.","source":"faq_src"},
        ]
    })

    # 3) Search DOC collection
    print("\n== SEARCH DOC COLLECTION ==")
    print(agent.process_to_json({
        "op": "search",
        "collection": coll_doc,          # search directly on __doc
        "search_vector": [0.3,0.1,0.2,0.4],
        "top_k": 2,
        "metric_type": "COSINE",
        "output_fields": ["id","type","text","source","metadata"]
    }, pretty=True))

    # 4) Search FAQ collection
    print("\n== SEARCH FAQ COLLECTION ==")
    print(agent.process_to_json({
        "op": "search",
        "collection": coll_faq,          # search directly on __faq
        "search_vector": [0.29,0.11,0.19,0.41],
        "top_k": 2,
        "metric_type": "COSINE",
        "output_fields": ["id","type","question","answer","source","metadata"]
    }, pretty=True))

    # 5) Stats
    print("\n== STATS DOC ==")
    agent.process_and_print({"op":"stats", "collection": coll_doc})

    print("\n== STATS FAQ ==")
    agent.process_and_print({"op":"stats", "collection": coll_faq})

finally:
    # Cleanup
    for name in (coll_doc, coll_faq):
        if utility.has_collection(name):
            utility.drop_collection(name)
            print(f"[cleanup] dropped collection {name}")
