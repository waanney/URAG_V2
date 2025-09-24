# -*- coding: utf-8 -*-
"""
llm_kernel.py — Kernel trung lập, chọn model từ UI (deferred-binding + global selection)
======================================================================================

Mục tiêu
- Không buộc provider/model khi khởi tạo; **UI** có thể chọn/đổi bất kỳ lúc nào.
- Code ứng dụng chỉ gọi **KERNEL.get_active_model(...)** để lấy model pydantic‑ai.
- Hỗ trợ Google (Gemini) & Ollama (OpenAI‑compatible). Dễ mở rộng provider khác.
- Tuỳ chọn: lưu cấu hình đang chọn ra file JSON, để app khởi động lại vẫn giữ.

Cách dùng nhanh
---------------
```python
from llm_kernel import KERNEL, GoogleConfig, OllamaConfig

# 1) UI (hoặc đoạn code bất kỳ) đặt lựa chọn đang dùng
KERNEL.set_active_config(GoogleConfig())
# hoặc local
# KERNEL.set_active_config(OllamaConfig(base_url="http://localhost:11434"))

# 2) Ở mọi nơi cần model cho Agent:
model = KERNEL.get_active_model(model_name="gemini-1.5-flash")
# ... Agent(model=model)
```

Nếu UI muốn lưu/đọc từ file JSON:
```python
KERNEL.save_active_config()   # ghi ra LLM_CONFIG_PATH (mặc định ./.llm_config.json)
KERNEL.load_active_config()   # đọc lại nếu có
```
"""
from __future__ import annotations
import os
import json
import logging
from threading import RLock
from typing import Callable, Dict, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, ValidationError

# pydantic-ai models/providers
from pydantic_ai.models import Model as PAModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider

log = logging.getLogger(__name__)

# =====================================
# Config schemas (nhẹ, expandable)
# =====================================
ProviderName = Literal["google", "ollama"]

class BaseKernelConfig(BaseModel):
    provider: ProviderName | None = Field(default=None, description="Tên provider. Nếu None sẽ quyết định sau.")
    model: str | None = Field(default=None, description="Model mặc định.")

class GoogleConfig(BaseKernelConfig):
    provider: ProviderName | None = Field(default="google")
    api_key: str | None = Field(default=None, description="ENV: GEMINI_API_KEY/GOOGLE_API_KEY")

class OllamaConfig(BaseKernelConfig):
    provider: ProviderName | None = Field(default="ollama")
    base_url: str | None = Field(default=None, description="Mặc định http://localhost:11434")

LLMConfig = Union[GoogleConfig, OllamaConfig]

# =====================================
# Kernel (deferred-binding + active selection)
# =====================================
class LLMKernel:
    """Kernel trung lập có **active_config** để UI có thể thay đổi runtime.

    API chính:
      - configure(config): cấu hình mặc định (fallback)
      - set_provider(name): chỉ định provider nhanh (tạo config rỗng)
      - set_active_config(config): chọn cấu hình đang dùng từ UI
      - get_model(...): build model theo config chỉ định (tức thời)
      - get_active_model(...): build theo **active_config** (nếu có),
        fallback về config mặc định hoặc ENV.
      - save_active_config()/load_active_config(): lưu/đọc JSON.
    """

    def __init__(self, default_config: Optional[LLMConfig] = None) -> None:
        self._default_config: Optional[LLMConfig] = default_config
        self._active_config: Optional[LLMConfig] = None
        self._lock = RLock()
        # provider -> factory
        self._registry: Dict[str, Callable[[str, Dict[str, object]], PAModel]] = {
            "google": self._factory_google,
            "ollama": self._factory_ollama,
        }

    # ---------- configuration APIs ----------
    def configure(self, config: LLMConfig) -> None:
        """Đặt cấu hình **mặc định** (fallback), không phải active."""
        with self._lock:
            self._default_config = config

    def set_provider(self, provider: ProviderName) -> None:
        """Chỉ định provider mặc định (khởi tạo config rỗng)."""
        if provider not in self._registry:
            raise ValueError(f"Provider '{provider}' chưa được hỗ trợ.")
        cfg = GoogleConfig() if provider == "google" else OllamaConfig()
        self.configure(cfg)

    def set_active_config(self, config: LLMConfig) -> None:
        """UI gọi hàm này để **chọn** cấu hình đang dùng."""
        with self._lock:
            self._active_config = config

    # ---------- public model builders ----------
    def get_model(self, model_name: Optional[str] = None, **overrides) -> PAModel:
        provider, model_name, params = self._resolve_provider_and_params(self._active_config, model_name, overrides)
        return self._build(provider, model_name, params)

    def get_active_model(self, model_name: Optional[str] = None, **overrides) -> PAModel:
        """Alias thuận tay cho code ứng dụng: luôn ưu tiên active_config nếu có."""
        return self.get_model(model_name=model_name, **overrides)

    # ---------- persistence (optional) ----------
    @staticmethod
    def _config_path() -> str:
        return os.getenv("LLM_CONFIG_PATH", os.path.abspath(".llm_config.json"))

    def save_active_config(self) -> str:
        """Ghi active_config ra JSON (để UI lưu lại lựa chọn)."""
        with self._lock:
            if self._active_config is None:
                raise RuntimeError("Chưa có active_config để lưu.")
            path = self._config_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._active_config.model_dump(), f, ensure_ascii=False, indent=2)
            log.info("Saved active LLM config to %s", path)
            return path

    def load_active_config(self) -> Optional[LLMConfig]:
        """Đọc JSON nếu có, set active_config. Trả về config hoặc None nếu không có file."""
        path = self._config_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prov = data.get("provider")
        try:
            if prov == "google":
                cfg = GoogleConfig.model_validate(data)
            elif prov == "ollama":
                cfg = OllamaConfig.model_validate(data)
            else:
                raise ValidationError(f"provider không hợp lệ: {prov}")
        except Exception as e:
            raise ValueError(f"Không parse được file cấu hình {path}: {e}")
        self.set_active_config(cfg)
        log.info("Loaded active LLM config from %s", path)
        return cfg

    # ---------- internals ----------
    def _build(self, provider: str, model_name: str, params: Dict[str, object]) -> PAModel:
        if provider not in self._registry:
            raise ValueError(f"Provider '{provider}' chưa được đăng ký trong kernel.")
        return self._registry[provider](model_name, params)

    def _resolve_provider_and_params(
        self,
        prefer_config: Optional[LLMConfig],
        model_name: Optional[str],
        overrides: Dict[str, object],
    ) -> Tuple[str, str, Dict[str, object]]:
        """Quyết định provider, tên model, và tham số provider.
        Ưu tiên: active_config → default_config → ENV heuristic.
        """
        with self._lock:
            cfg = prefer_config or self._active_config or self._default_config

        provider: Optional[str] = getattr(cfg, "provider", None) if cfg else None

        # Heuristic nếu chưa có provider: ưu tiên ENV
        if not provider:
            if os.getenv("USE_OLLAMA", "0") == "1":
                provider = "ollama"
            elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                provider = "google"
            else:
                provider = "ollama"  # local-friendly default
                log.info("Provider chưa đặt; mặc định 'ollama'. Dùng USE_OLLAMA=0 và set GEMINI_API_KEY để ưu tiên Google.")

        # Model name
        final_model = model_name or (getattr(cfg, "model", None) if cfg else None)
        if not final_model:
            raise ValueError("Thiếu tên model. Truyền get_*model(model_name=...) hoặc đặt trong config.model")

        # Params từ config + overrides
        base_params: Dict[str, object] = {}
        if isinstance(cfg, GoogleConfig):
            if cfg.api_key is not None:
                base_params["api_key"] = cfg.api_key
        elif isinstance(cfg, OllamaConfig):
            if cfg.base_url is not None:
                base_params["base_url"] = cfg.base_url

        base_params.update(overrides)  # ưu tiên overrides
        return provider, final_model, base_params

    # ---------- factories (provider‑specific) ----------
    @staticmethod
    def _factory_google(model_name: str, params: Dict[str, object]) -> PAModel:
        api_key = (
            params.get("api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("Google provider: thiếu API key (GEMINI_API_KEY/GOOGLE_API_KEY hoặc truyền api_key=...).")
        prov = GoogleProvider(api_key=str(api_key))
        return GoogleModel(model_name, provider=prov)

    @staticmethod
    def _factory_ollama(model_name: str, params: Dict[str, object]) -> PAModel:
        # Lazy import để tránh cần 'openai' khi không dùng Ollama
        from pydantic_ai.models.openai import OpenAIChatModel  # noqa: WPS433

        base_url = str(params.get("base_url") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434")
        prov = OllamaProvider(base_url=base_url)
        return OpenAIChatModel(model_name, provider=prov)


# =====================================
# Global kernel (để mọi nơi import dùng chung)
# =====================================
KERNEL = LLMKernel()


# =====================================
# Demo
# =====================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Ví dụ UI chọn Google
    KERNEL.set_active_config(GoogleConfig())
    try:
        gm = KERNEL.get_active_model("gemini-1.5-flash")
        print("Created google model:", type(gm))
    except Exception as e:
        print("Google skipped:", e)

    # Ví dụ UI chọn Ollama
    KERNEL.set_active_config(OllamaConfig(base_url="http://localhost:11434"))
    om = KERNEL.get_active_model("llama3")
    print("Created ollama model:", type(om))
