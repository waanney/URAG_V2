# examples/indexing_test_3.py
import time, random, string
from pymilvus import utility
from src.indexing.indexing_agent import IndexingAgent

def r(n=6): return ''.join(random.choices(string.ascii_lowercase+string.digits, k=n))

agent = IndexingAgent()
coll = f"ura_rag_test3_{int(time.time())}_{r()}"

try:
    # 1) create
    agent.process_and_print({
        "op":"create_collection", "collection":coll, "dim":4, "metric_type":"COSINE"
    })

    # 2) index: 2 doc + 2 faq
    agent.process_and_print({
        "op":"index", "collection":coll, "metric_type":"COSINE",
        "items":[
            # DOC
            {"id":"doc#A","type":"doc","vector":[0.1,0.2,0.3,0.4], "text":"tài liệu về Milvus compose"},
            {"id":"doc#B","type":"doc","vector":[0.4,0.2,0.1,0.3], "text":"hướng dẫn cài đặt và cấu hình"},
            # FAQ (vector nên là embed của question)
            {"id":"faq#1","type":"faq","vector":[0.2,0.1,0.4,0.3],
             "question":"Milvus chạy sao?", "answer":"Dùng Docker compose standalone."},
            {"id":"faq#2","type":"faq","vector":[0.15,0.25,0.35,0.25],
             "question":"Làm sao xóa dữ liệu trong Milvus?",
             "answer":"Dùng col.delete(expr) rồi compaction tự động."},
        ]
    })

    # 3) search trong FAQ
    print("\n=== SEARCH FAQ ===")
    agent.process_and_print({
        "op":"search", "collection":coll,
        "search_vector":[0.2,0.1,0.4,0.3],      # giống faq#1
        "top_k":3, "metric_type":"COSINE",
        "filter": 'type == "faq"',
        "output_fields":["id","type","question","answer"]
    })

    # 4) search trong DOC
    print("\n=== SEARCH DOC ===")
    agent.process_and_print({
        "op":"search", "collection":coll,
        "search_vector":[0.1,0.2,0.3,0.4],      # giống doc#A
        "top_k":3, "metric_type":"COSINE",
        "filter": 'type == "doc"',
        "output_fields":["id","type","text"]
    })

    # 5) (tuỳ chọn) xoá 1 FAQ rồi search lại
    print("\n=== DELETE faq#1 & SEARCH FAQ AGAIN ===")
    agent.process_and_print({"op":"delete", "collection":coll, "ids":["faq#1"]})
    agent.process_and_print({
        "op":"search", "collection":coll,
        "search_vector":[0.2,0.1,0.4,0.3],
        "top_k":5, "metric_type":"COSINE",
        "filter": 'type == "faq"',
        "output_fields":["id","type","question"]
    })

    # 6) stats
    print("\n=== STATS ===")
    agent.process_and_print({"op":"stats", "collection":coll})

finally:
    if utility.has_collection(coll):
        utility.drop_collection(coll)
        print(f"[cleanup] dropped collection {coll}")
