# llm_kernel.py (phiên bản nâng cấp hỗ trợ local model)
# -*- coding: utf-8 -*-
"""
Module LLM Kernel - Lõi LLM linh hoạt cho dự án.

Hỗ trợ chuyển đổi giữa các nhà cung cấp LLM (đám mây hoặc local)
thông qua cấu hình mà không cần thay đổi code ứng dụng.
"""
from __future__ import annotations
import os
from typing import Literal, Union

from pydantic import BaseModel, Field

# Import các thành phần cần thiết từ pydantic-ai
# LLM là một protocol (giao diện) chung mà tất cả các model phải tuân theo
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# ==============================================================================
# Phần 1: Các Lớp Cấu Hình (Config Schemas)
# ==============================================================================

class GoogleConfig(BaseModel):
    """Cấu hình cho nhà cung cấp Google Gemini."""
    provider: Literal["google"] = "google"
    api_key: str | None = Field(default=None, description="API Key của Google. Nếu là None, sẽ đọc từ biến môi trường.")

class OllamaConfig(BaseModel):
    """Cấu hình cho nhà cung cấp Ollama (chạy model local)."""
    provider: Literal["ollama"] = "ollama"
    base_url: str = Field(default="http://localhost:11434", description="URL của server Ollama.")
    timeout: int = Field(default=120, description="Thời gian chờ (giây) cho mỗi yêu cầu.")

# Sử dụng Union để Pydantic tự động nhận diện và xác thực cấu hình phù hợp
LLMConfig = Union[GoogleConfig, OllamaConfig]

# ==============================================================================
# Phần 2: Kernel linh hoạt
# ==============================================================================

class LLMKernel:
    """
    Kernel quản lý việc khởi tạo LLM từ một cấu hình cho trước.
    Hoạt động như một nhà máy (factory) tạo ra các model tương thích pydantic-ai.
    """
    def __init__(self, config: LLMConfig):
        """
        Khởi tạo Kernel dựa trên đối tượng cấu hình.
        
        Args:
            config: Một đối tượng cấu hình (GoogleConfig hoặc OllamaConfig).
        """
        self.config = config
        self._provider = self._create_provider()

    def _create_provider(self) -> Union[GoogleProvider, OllamaProvider]:
        """Phương thức nội bộ để tạo provider dựa trên cấu hình."""
        
        # Trường hợp 1: Cấu hình là Google
        if isinstance(self.config, GoogleConfig):
            api_key = self.config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Cấu hình là 'google' nhưng không tìm thấy API key.")
            return GoogleProvider(api_key=api_key)

        # Trường hợp 2: Cấu hình là Ollama
        if isinstance(self.config, OllamaConfig):
            # Thêm một thông báo để người dùng biết họ đang dùng model local
            print(f"INFO: Khởi tạo provider cho Ollama tại {self.config.base_url}")
            return OllamaProvider(base_url=self.config.base_url)
        
        # Báo lỗi nếu cấu hình không được hỗ trợ
        raise NotImplementedError(f"Provider '{self.config.provider}' chưa được hỗ trợ.")

    def get_model(self, model_name: str):
        """
        Lấy về một đối tượng model tương thích với `pydantic-ai.Agent`.

        Args:
            model_name: Tên của model (ví dụ: 'gemini-1.5-flash' cho Google,
                        hoặc 'llama3' cho Ollama).

        Returns:
            Một đối tượng model (GoogleModel, OllamaModel, v.v.).
        """
        if isinstance(self.config, GoogleConfig):
            return GoogleModel(model_name, provider=self._provider)
        
        if isinstance(self.config, OllamaConfig):
            return OpenAIChatModel(model_name, provider=self._provider)

        raise NotImplementedError(f"Không thể tạo model cho provider '{self.config.provider}'.")