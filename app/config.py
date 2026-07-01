"""Centralised configuration.

All tunables are read from environment variables (with sensible defaults) so the
same code runs locally (via a .env file) and on Render (via dashboard env vars).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if present. In production (Render) the vars are already in the
# environment, so this is a harmless no-op.
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # ---- Paths ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    raw_catalog_path: Path = PROJECT_ROOT / "data" / "shl_product_catalog.json"

    # ---- LLM / embeddings ----
    llm_provider: str = _get("LLM_PROVIDER", "openai")
    chat_model: str = _get("CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = _get("EMBED_MODEL", "text-embedding-3-small")

    # API keys: dedicated overrides fall back to the single OPENAI_API_KEY.
    openai_api_key: str = _get("OPENAI_API_KEY", "")
    llm_api_key: str = _get("LLM_API_KEY", "") or _get("OPENAI_API_KEY", "")
    embed_api_key: str = _get("EMBED_API_KEY", "") or _get("OPENAI_API_KEY", "")

    # Base URLs: None => official OpenAI endpoint. Set for Groq/OpenRouter/Gemini.
    llm_base_url: str | None = os.getenv("LLM_BASE_URL") or None
    embed_base_url: str | None = os.getenv("EMBED_BASE_URL") or None

    # ---- Behaviour / budgets ----
    request_timeout_s: float = float(_get("REQUEST_TIMEOUT_S", "20"))
    max_retries: int = int(_get("LLM_MAX_RETRIES", "2"))
    retrieval_top_k: int = int(_get("RETRIEVAL_TOP_K", "50"))
    max_recommendations: int = int(_get("MAX_RECOMMENDATIONS", "10"))
    embed_dim: int = int(_get("EMBED_DIM", "1536"))  # text-embedding-3-small

    @property
    def catalog_path(self) -> Path:
        return self.processed_dir / "catalog.json"

    @property
    def embeddings_path(self) -> Path:
        return self.processed_dir / "embeddings.npy"

    @property
    def has_api_key(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()
