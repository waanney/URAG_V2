# -*- coding: utf-8 -*-
"""
faq.py — URAG-F: Generate & Enrich FAQs from a document

- generate(document): sinh các cặp (question, answer) từ tài liệu
- enrich(faqs): chỉ tạo biến thể câu hỏi (paraphrase); câu trả lời GIỮ NGUYÊN
- generate_and_enrich(document): quy trình kết hợp

Yêu cầu môi trường:
  pip install pydantic-ai google-genai
  export GEMINI_API_KEY=...

Lưu ý:
- Enrich KHÔNG cho LLM tạo answer; code tự map answer từ seed để đảm bảo nhất quán.
"""

from __future__ import annotations
from typing import List, Tuple, Iterable, Set, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from src.llm.llm_kernel import KERNEL


# =========================
# Schemas & Prompts
# =========================

class FAQPair(BaseModel):
    question: str = Field(..., description="Câu hỏi (FAQ)")
    answer: str = Field(..., description="Câu trả lời tương ứng (từ document hoặc giữ nguyên khi enrich)")

class FAQList(BaseModel):
    faqs: List[FAQPair]

class _GenOut(BaseModel):
    """Dạng đầu ra khi generate: list các tuple (q, a)."""
    output: List[FAQPair]

class QuestionVariantsOut(BaseModel):
    """Dạng đầu ra khi enrich: chỉ danh sách câu hỏi mới (không có answer)."""
    questions: List[str]


PROMPT_GENERATE = """
# ROLE
Bạn là một chuyên gia tạo bộ câu hỏi thường gặp (FAQ) từ tài liệu được cung cấp. Mục tiêu của bạn là tạo ra các cặp câu hỏi và câu trả lời súc tích, chính xác và hữu ích.

# INPUT
<document>
{document}
</document>

# INSTRUCTIONS
1.  **Nghiêm túc phân tích** nội dung trong thẻ `<document>`.
2.  Tạo ra **ít nhất {min_pairs} cặp câu hỏi và câu trả lời** chất lượng cao.
3.  **YÊU CẦU VỀ CÂU HỎI:**
    * Ngắn gọn, rõ ràng, và đi thẳng vào vấn đề mà người dùng có thể quan tâm.
    * Ưu tiên các câu hỏi về: định nghĩa ("...là gì?"), chức năng ("làm thế nào để...?"), quy trình ("các bước thực hiện..."), so sánh ("sự khác biệt giữa...").
4.  **YÊU CẦU VỀ CÂU TRẢ LỜI:**
    * **CHỈ ĐƯỢC PHÉP** trích xuất hoặc tóm tắt thông tin trực tiếp từ `<document>`.
    * **TUYỆT ĐỐI KHÔNG** suy diễn, bình luận, hoặc thêm thông tin không có trong văn bản.
5.  **RÀNG BUỘC PHỦ ĐỊNH (NHỮNG ĐIỀU CẦN TRÁNH):**
    * Không tạo các câu hỏi trùng lặp về ý nghĩa.
    * Không hỏi những câu yêu cầu ý kiến cá nhân ("bạn nghĩ sao về...?").
    * Nếu tài liệu quá ngắn hoặc không đủ thông tin để tạo `{min_pairs}` câu hỏi chất lượng, hãy tạo ít hơn thay vì bịa đặt.

# INSTRUCTIONS
- Tạo ra chính xác {min_pairs} cặp (question, answer).
- Câu hỏi ngắn gọn, bám sát tài liệu; tránh trùng lặp.
- Câu trả lời CHỈ dựa trên DOCUMENT; không suy diễn, không bịa.

# OUTPUT (JSON STRICT)
{{
  "output": [
    {{"question": "<question-1>", "answer": "<answer-1>"}},
    {{"question": "<question-2>", "answer": "<answer-2>"}}
  ]
}}
""".strip()

PROMPT_ENRICH_QUESTIONS = """
# ROLE
Bạn là một chuyên gia sáng tạo nội dung, có nhiệm vụ tạo ra các biến thể câu hỏi (paraphrasing) một cách tự nhiên và đa dạng.

# INPUT
<seed_question>
{seed_q}
</seed_question>

<seed_answer_context>
{seed_a}
</seed_answer_context>

# INSTRUCTIONS
1.  **Phân tích:** Đọc kỹ `<seed_question>` và `<seed_answer_context>` để hiểu rõ ý nghĩa cốt lõi và phạm vi của câu hỏi gốc.
2.  **Sáng tạo:** Dựa trên ý nghĩa cốt lõi đó, sinh ra **ít nhất {n_variants} biến thể câu hỏi** mới.
3.  **YÊU CẦU VỀ BIẾN THỂ:**
    * **GIỮ NGUYÊN Ý NGHĨA:** Các câu hỏi mới phải hỏi về cùng một vấn đề như câu hỏi gốc.
    * CÂU HỎI MỚI CHỈ CÓ TÁC DỤNG BIỂU DIỄN MỘT CÁCH DIỄN ĐẠT KHÁC
    * CÂU HỎI MỚI CÓ CÂU TRẢ LỜI GIỐNG VỚI CÂU HỎI GỐC, CÂU TRẢ LỜI GỐC PHẢI TRẢ LỜI ĐƯỢC CÂU HỎI MỚI.
    * **ĐA DẠNG HÓA:** Sử dụng nhiều cách diễn đạt khác nhau: câu hỏi có/không, câu hỏi Wh- (what, how, why), dùng từ đồng nghĩa, đảo cấu trúc, dạng rút gọn...
    * **TỰ NHIÊN:** Nghe giống như cách một người thật sẽ hỏi.
4.  **RÀNG BUỘC PHỦ ĐỊNH:**
    * **KHÔNG** thay đổi, mở rộng hay thu hẹp ý nghĩa của câu hỏi gốc.
    * **KHÔNG** tạo lại câu trả lời.
    * Loại bỏ các biến thể gần như giống hệt nhau.


# OUTPUT (JSON STRICT)
{{
  "questions": [
    "<question-variant-1>",
    "<question-variant-2>"
  ]
}}
""".strip()


# =========================
# FAQ Agent
# =========================

class FAQAgent:
    """
    Tạo và enrich FAQ từ document bằng 2 agent (pydantic-ai) chạy trên Gemini.
      - generate(): sinh (question, answer) gốc từ document
      - enrich():   chỉ paraphrase QUESTION, ANSWER giữ NGUYÊN văn seed
      - generate_and_enrich(): pipeline kết hợp
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = 'gemini-2.0-flash',
        min_pairs:int = 2,
        enrich_pairs_per_seed: int = 2,
    ):
        self.model = KERNEL.get_active_model(model_name=model_name)
        self.min_pairs = int(min_pairs)
        self.enriched_pairs_per_seed = int(enrich_pairs_per_seed)

        self.gen_agent = Agent(
            self.model,
            system_prompt="Bạn nhận prompt JSON/text và chỉ trả về JSON đúng schema yêu cầu.",
            output_type=_GenOut,
        )
        self.enrich_q_agent = Agent(
            self.model,
            system_prompt="Bạn nhận prompt JSON/text và chỉ trả về JSON đúng schema yêu cầu.",
            output_type=QuestionVariantsOut,
        )

    # ---------- Helpers ----------

    @staticmethod
    def _pairs_to_models(pairs) -> List[FAQPair]:
        out = []
        for it in pairs:
            if isinstance(it, FAQPair):
                q, a = it.question.strip(), it.answer.strip()
            else:
                q, a = (it[0] or "").strip(), (it[1] or "").strip()
            if q and a:
                out.append(FAQPair(question=q, answer=a))
        return out


    @staticmethod
    def _dedup(items: Iterable[FAQPair]) -> List[FAQPair]:
        """Chống trùng theo lowercase(question)."""
        seen: Set[str] = set()
        uniq: List[FAQPair] = []
        for it in items:
            k = it.question.strip().lower()
            if k and k not in seen:
                seen.add(k)
                uniq.append(it)
        return uniq

    # ---------- Sync API ----------

    def generate(self, document: str) -> List[FAQPair]:
        prompt = PROMPT_GENERATE.format(document=document, min_pairs=self.min_pairs)
        res = self.gen_agent.run_sync(prompt)
        base: List[FAQPair] = res.output.output or []
        return self._dedup(base)

    def enrich(self, faqs: List[FAQPair]) -> List[FAQPair]:
        """
        ENRICH: chỉ tạo biến thể câu hỏi; câu trả lời GIỮ NGUYÊN theo seed.
        Trả về các FAQPair mới (question mới, answer = seed.answer).
        """
        if not faqs:
            return []
        variants: List[FAQPair] = []
        for f in faqs:
            prompt = PROMPT_ENRICH_QUESTIONS.format(
                seed_q=f.question.replace('"', '\\"'),
                seed_a=f.answer.replace('"', '\\"'),
                n_variants=self.enriched_pairs_per_seed,
            )
            out = self.enrich_q_agent.run_sync(prompt).output
            for q in (out.questions or []):
                q = (q or "").strip()
                if q:
                    variants.append(FAQPair(question=q, answer=f.answer))
        return self._dedup(variants)

    def generate_and_enrich(self, document: str, do_enrich: bool = True) -> FAQList:
        base = self.generate(document)
        if not do_enrich:
            return FAQList(faqs=base)
        more = self.enrich(base)
        merged = self._dedup([*base, *more])
        return FAQList(faqs=merged)

    # ---------- Async API ----------

    async def agenerate(self, document: str) -> List[FAQPair]:
        prompt = PROMPT_GENERATE.format(document=document, min_pairs=self.min_pairs)
        res = await self.gen_agent.run(prompt)
        base = self._pairs_to_models(res.output.output or [])
        return self._dedup(base)

    async def aenrich(self, faqs: List[FAQPair]) -> List[FAQPair]:
        if not faqs:
            return []
        variants: List[FAQPair] = []
        for f in faqs:
            prompt = PROMPT_ENRICH_QUESTIONS.format(
                seed_q=f.question.replace('"', '\\"'),
                seed_a=f.answer.replace('"', '\\"'),
                n_variants=self.enriched_pairs_per_seed,
            )
            out = (await self.enrich_q_agent.run(prompt)).output
            for q in (out.questions or []):
                q = (q or "").strip()
                if q:
                    variants.append(FAQPair(question=q, answer=f.answer))
        return self._dedup(variants)

    async def agenerate_and_enrich(self, document: str, do_enrich: bool = True) -> FAQList:
        base = await self.agenerate(document)
        if not do_enrich:
            return FAQList(faqs=base)
        more = await self.aenrich(base)
        merged = self._dedup([*base, *more])
        return FAQList(faqs=merged)