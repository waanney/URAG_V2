# tests/test_agent_basic.py
import time, random, string
import pytest
from pymilvus import utility
from src.indexing.indexing_agent import IndexingAgent, CreateCollectionReq, UpsertIndexReq, SearchReq, Item

def _rand(n=6): return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

@pytest.fixture(scope="function")
def coll():
    return f"ura_rag_pytest_{int(time.time())}_{_rand()}"

@pytest.fixture(scope="function")
def agent():
    return IndexingAgent()

def test_create_index_search(agent, coll):
    try:
        r = agent.process(CreateCollectionReq(collection=coll, dim=4, metric_type="COSINE"))
        assert r["status"] == "ok"

        items = [
            Item(id="doc#1", type="doc", vector=[0.1,0.2,0.3,0.4], text="xin chào"),
            Item(id="faq#1", type="faq", vector=[0.2,0.1,0.4,0.3],
                 question="Milvus chạy sao?", answer="Dùng Docker compose standalone.")
        ]
        r = agent.process(UpsertIndexReq(op="index", collection=coll, items=items, metric_type="COSINE"))
        assert r["status"] == "ok"
        assert r["data"]["inserted"] == 2

        r = agent.process(SearchReq(collection=coll, search_vector=[0.2,0.1,0.4,0.3], top_k=1,
                                    metric_type="COSINE", filter='type == "faq"',
                                    output_fields=["id","type","question","answer"]))
        assert r["status"] == "ok"
        assert r["data"]["results"][0]["id"] == "faq#1"
    finally:
        if utility.has_collection(coll):
            utility.drop_collection(coll)
