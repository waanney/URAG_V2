# -*- coding: utf-8 -*-
from pymilvus import connections, utility, Collection
import os, json

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
# base phải trùng với collection_base khi bạn index (ví dụ từ meta_manager_test)
BASE = os.getenv("COL_BASE", "rag_meta_test_1758648190")  # sửa hoặc export COL_BASE=...

def main():
    print("[i] Connect:", MILVUS_URI)
    connections.connect(uri=MILVUS_URI, alias="default")

    for suffix in ("__doc", "__faq"):
        name = f"{BASE}{suffix}"
        print(f"\n=== {name} ===")
        if not utility.has_collection(name):
            print("  (missing)")
            continue

        col = Collection(name)
        print("  -> num_entities:", col.num_entities)
        print("  -> schema fields:", [f.name + ":" + f.dtype.name for f in col.schema.fields])
        print("  -> indexes:", [idx.params for idx in col.indexes])

        # load vào memory để query/search
        col.load()

        # 1) Lấy vài record (Milvus v2 hỗ trợ query theo expr; nếu có auto id/id field):
        try:
            # nếu bạn có trường 'id' dạng string, có thể query expr, còn không thì dùng pk mặc định
            res = col.query(expr="", output_fields=["id","type","source","text","question","answer","metadata"], limit=5)
            print("  sample rows:")
            for r in res:
                print("   ", json.dumps(r, ensure_ascii=False))
        except Exception as e:
            print("  query sample failed:", e)

        # 2) Lấy raw vectors một phần (tuỳ tên field — thường là 'vector' hoặc 'embedding'):
        try:
            res = col.query(expr="", output_fields=["id","vector"], limit=2)
            for r in res:
                v = r.get("vector")
                print(f"  vector sample id={r.get('id')} dim={len(v) if v else 0} first5={v[:5] if v else None}")
        except Exception as e:
            print("  vector fetch failed:", e)

if __name__ == "__main__":
    main()
