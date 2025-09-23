# examples/semantic_chunker_test.py
from __future__ import annotations
import json
import time
from pathlib import Path

from src.llm.URag_D.semantic_chunker import SemanticChunkerLC, LCChunkerConfig

# ====== input demo ======
DOC_ID = f"doc_demo_{int(time.time())}"
TEXT = """
Trường Đại học Bách Khoa (HCMUT) có 2 cơ sở chính. Cơ sở 1 nằm ở Quận 10,
còn Cơ sở 2 ở Thành phố Thủ Đức. Sinh viên có thể đăng ký học phần ở cả hai cơ sở
tuỳ theo kế hoạch đào tạo.

Học phí được thu theo học kỳ. Thời hạn đóng học phí là trước ngày 15 mỗi tháng
khi có thông báo. Nếu chậm trễ, sinh viên có thể bị tính phí phạt theo quy định
của nhà trường.

Thư viện mở cửa từ 7:30 đến 20:30 các ngày trong tuần. Vào kỳ thi, thời gian
mở cửa có thể kéo dài để hỗ trợ sinh viên ôn tập.
""".strip()

# ====== init chunker ======
# language="vi" -> dùng model VN (dangvantuan/vietnamese-embedding) qua EmbedderAgent adapter
cfg = LCChunkerConfig(
    language="vi",                # "vi" hoặc "default"
    use_agent_embedder=True,      # dùng EmbedderAgent có sẵn (tránh cần langchain_community)
    buffer_size=1,
    breakpoint_threshold_type="percentile",  # 'percentile' | 'standard_deviation' | 'interquartile' | 'gradient'
    breakpoint_threshold_amount=95,          # ngưỡng gộp câu (percentile)
    min_chunk_size=200,                      # tối thiểu ~200 ký tự mỗi chunk để đỡ vụn (tùy bạn)
)

chunker = SemanticChunkerLC(cfg)

# ====== run chunking ======
chunks = chunker.chunk(TEXT, doc_id=DOC_ID)

# ====== show / save ======
print("== CHUNKS (JSON) ==")
print(json.dumps(chunks, ensure_ascii=False, indent=2))

# ghi ra file để xem lại
out_dir = Path("tmp_chunks")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{DOC_ID}.json"
out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[WROTE] {out_path}")
