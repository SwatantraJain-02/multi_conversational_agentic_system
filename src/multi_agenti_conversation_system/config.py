from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables and the project-root .env file."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OLLAMA_BASE_URL: str
    CHAT_MODEL: str
    EMBED_MODEL: str
    TEMPERATURE: float
    MODEL_REASONING : bool
    OPENROUTER_API_KEY : str


settings = Settings()
