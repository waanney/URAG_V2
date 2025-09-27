# examples/run_search.py
from src.search.search_agent import SearchAgent, SearchConfig
from src.embedding.embedding_agent import EmbConfig
from src.indexing.indexing_agent import AgentConfig
from src.llm.llm_kernel import KERNEL, GoogleConfig  # hoặc OllamaConfig/OpenAIConfig
from pymilvus import utility, connections, Collection

def _drop_one_collection(col_name: str) -> None:
    """Drop một collection (nếu có), kèm thả index và release."""
    if not utility.has_collection(col_name):
        print(f"[Clean] Collection '{col_name}' không tồn tại -> bỏ qua.")
        return
    try:
        col = Collection(col_name)
        # Thả index nếu có (tránh warning khi drop)
        try:
            for idx in getattr(col, "indexes", []):
                try:
                    col.drop_index(idx.index_name)
                    print(f"[Clean]  - Dropped index: {idx.index_name}")
                except Exception as e:
                    print(f"[Clean]  - Drop index lỗi ({idx.index_name}): {e}")
        except Exception as e:
            print(f"[Clean]  - Liệt kê index lỗi: {e}")
        # Release khỏi memory
        try:
            col.release()
        except Exception:
            pass
        # Drop collection
        utility.drop_collection(col_name)
        print(f"[Clean] Dropped collection: {col_name}")
    except Exception as e:
        print(f"[Clean] Lỗi khi drop '{col_name}': {e}")

def clean_collection_base(
    collection_base: str,
    i_cfg: "AgentConfig",
    also_single_base: bool = True
) -> None:
    """
    Xoá sạch dữ liệu cho một collection base:
      - Nếu dùng dual_collections: {base}__doc, {base}__faq
      - Tuỳ chọn xoá cả collection đơn lẻ trùng tên base (also_single_base=True)
    """
    # 1) Kết nối Milvus theo AgentConfig
    connections.connect(alias=i_cfg.alias, uri=i_cfg.uri)
    print(f"[Clean] Connected alias='{i_cfg.alias}' -> {i_cfg.uri}")

    # 2) Tạo danh sách collection cần xoá
    targets = set()
    if getattr(i_cfg, "dual_collections", True):
        targets.add(f"{collection_base}{i_cfg.doc_suffix}")
        targets.add(f"{collection_base}{i_cfg.faq_suffix}")
    else:
        targets.add(collection_base)

    if also_single_base:
        targets.add(collection_base)  # đề phòng có collection trùng base

    # 3) Drop từng collection
    for name in targets:
        _drop_one_collection(name)

    # 4) Kiểm tra lại
    for name in targets:
        exists = utility.has_collection(name)
        print(f"[Clean] Verify '{name}': {'EXISTS' if exists else 'OK (deleted)'}")

def delete(collection='viquad_demo_1'):
    i_cfg = AgentConfig(alias="default")
    clean_collection_base(collection, i_cfg=i_cfg, also_single_base=True)

def main():
    # 1) Chọn LLM qua Kernel (UI/ENV của bạn có thể set chỗ khác)
    KERNEL.set_active_config(GoogleConfig())  # ví dụ: dùng Gemini qua SDK pydantic-ai
    # 2) Cấu hình Search
    s_cfg = SearchConfig(
        collection_base="viquad_test_2", 
        faq_top_k=5, doc_top_k=5,
        tFAQ=0.70, tDOC=0.60,
        metric="COSINE",
        max_ctx_docs=4,
        disclaimer="Lưu ý: Câu trả lời được tổng hợp từ tài liệu hệ thống.",
        llm_system_instruction="Bạn là trợ lý chỉ trả lời dựa trên CONTEXT. Nếu thiếu dữ liệu, hãy nói 'không có thông tin'.",
        # (tuỳ index) ví dụ HNSW cho Milvus:
        faq_search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        doc_search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )

    # 3) (Optional) cấu hình Embedder/Indexer nếu bạn muốn override mặc định
    e_cfg = EmbConfig(language="default")            # model embed mặc định của bạn
    i_cfg = AgentConfig(alias="default")  # Milvus conn nếu cần

    agent = SearchAgent(s_cfg, e_cfg=e_cfg, i_cfg=i_cfg)

    # 4) Hỏi thử
    q = "Vấn đề về lĩnh vực y tế nào mà Hà Nội đang vấp phải tương tự như thành phố Hồ Chí Minh?"
    resp = agent.answer(q)

    
    print("[Trace]\n", resp.trace.model_dump())
    print("[Tier]", resp.tier)
    print("[Answer]\n", resp.final_answer)
if __name__ == "__main__":
    main()
