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
    # Database configuration (Aiven Postgres: one service, two databases)
    # ---------------------------
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_SSLMODE: str = "require"
    POSTGRES_ADMIN_USER: str = "avnadmin"
    POSTGRES_ADMIN_PASSWORD: str
    POSTGRES_ADMIN_DB: str = "defaultdb"
    AGENT_APP_PASSWORD: str
    AGENT_READONLY_PASSWORD: str
    NORTHWIND_DB: str = "northwind"
    MEMORY_DB: str = "agent_memory"
    MAX_ROWS: int = 100

    # ---------------------------
    # Vector database configuration (Pinecone serverless)
    # ---------------------------
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "capstone"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # ---------------------------
    # LLM / model configuration (OpenRouter)
    # ---------------------------
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    CHAT_MODEL: str
    JUDGE_MODEL: str = ""
    EMBED_MODEL: str = "nvidia/nemotron-3.5-embed-8b:free"
    # Nemotron embed models are trained with these prefixes (check against the model card).
    EMBED_QUERY_PREFIX: str = "query: "
    EMBED_DOC_PREFIX: str = "passage: "
    # The free Nemotron endpoint fails ("service inference failed") on batches of real
    # documents, so send one text per request by default (G18).
    EMBED_BATCH_SIZE: int = 1
    TEMPERATURE: float = 0.0
    MODEL_REASONING: bool = False

    # ---------------------------
    # Langfuse (prompts, traces, evals)
    # ---------------------------
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    PROMPT_LABEL: str = "production"

    # ---------------------------
    # MCP configuration
    # ---------------------------
    DB_MCP_HOST: str = "localhost"
    DB_MCP_PORT: int = 5555
    DB_MCP_TOKEN: str

    # ---------------------------
    # Web search
    # ---------------------------
    TAVILY_API_KEY: str = ""

    def _conninfo(self, dbname: str, user: str, password: str) -> str:
        return make_conninfo(
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            dbname=dbname,
            user=user,
            password=password,
            sslmode=self.POSTGRES_SSLMODE,
        )

    @property
    def northwind_conninfo(self) -> str:
        return self._conninfo(
            self.NORTHWIND_DB, "agent_readonly", self.AGENT_READONLY_PASSWORD
        )

    @property
    def memory_conninfo(self) -> str:
        return self._conninfo(self.MEMORY_DB, "agent_app", self.AGENT_APP_PASSWORD)

    def admin_conninfo(self, dbname: str | None = None) -> str:
        return self._conninfo(
            dbname or self.POSTGRES_ADMIN_DB,
            self.POSTGRES_ADMIN_USER,
            self.POSTGRES_ADMIN_PASSWORD,
        )


settings = Settings()
