from pathlib import Path

from psycopg.conninfo import make_conninfo
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration, loaded from the project-root .env file."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------
    # Database configuration
    # ---------------------------
    POSTGRES_ADMIN_PASSWORD: str
    AGENT_APP_PASSWORD: str
    AGENT_READONLY_PASSWORD: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    NORTHWIND_DB: str = "northwind"
    MAX_ROWS: int = 100

    # ---------------------------
    # Vector database configuration
    # ---------------------------
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 5430

    # ---------------------------
    # LLM / model configuration
    # ---------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENROUTER_API_KEY: str
    CHAT_MODEL: str
    EMBED_MODEL: str = "nomic-embed-text"
    TEMPERATURE: float = 0.0
    MODEL_REASONING: bool = False

    # ---------------------------
    # MCP configuration
    # ---------------------------
    DB_MCP_HOST: str = "localhost"
    DB_MCP_PORT: int = 5555
    DB_MCP_TOKEN: str

    @property
    def northwind_conninfo(self) -> str:
        return make_conninfo(
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            dbname=self.NORTHWIND_DB,
            user="agent_readonly",
            password=self.AGENT_READONLY_PASSWORD,
        )


settings = Settings()
