# examples/indexing_test_2.py
import time, random, string
from pymilvus import utility
from src.indexing.indexing_agent import IndexingAgent

def rand_suffix(n=6):
    import random, string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

agent = IndexingAgent()
coll = f"ura_rag_test2_{int(time.time())}_{rand_suffix()}"

try:
    # tạo collection
    agent.process_and_print({
        "op": "create_collection",
        "collection": coll,
        "dim": 4,
        "metric_type": "COSINE"
    })

    # index 2 doc
    agent.process_and_print({
        "op": "index",
        "collection": coll,
        "metric_type": "COSINE",
        "items": [
            {"id":"doc#A","type":"doc","vector":[0.3,0.1,0.2,0.4],"text":"đoạn tài liệu A"},
            {"id":"doc#B","type":"doc","vector":[0.4,0.2,0.1,0.3],"text":"đoạn tài liệu B"},
        ]
    })

    # search trong DOC
    print(agent.process_to_json({
        "op":"search",
        "collection":coll,
        "search_vector":[0.3,0.1,0.2,0.4],
        "top_k":2,
        "metric_type":"COSINE",
        "filter": 'type == "doc"',
        "output_fields":["id","type","text"]
    }))

    # stats
    agent.process_and_print({"op":"stats", "collection":coll})

finally:
    if utility.has_collection(coll):
        utility.drop_collection(coll)
        print(f"[cleanup] dropped collection {coll}")
