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
    # How long a fetched prompt is reused before Langfuse is asked again: a prompt promoted
    # in the UI reaches the running agents within this many seconds.
    PROMPT_CACHE_TTL_S: int = 60
    # Keep these small: a bad key or an outage must not stall an agent for long.
    PROMPT_FETCH_TIMEOUT_S: int = 3
    # Off in unit tests and offline runs. Tracing is also off when the keys are empty.
    TRACING_ENABLED: bool = True

    # ---------------------------
    # MCP configuration
    # ---------------------------
    DB_MCP_HOST: str = "localhost"
    DB_MCP_PORT: int = 5555
    DB_MCP_TOKEN: str

    # ---------------------------
    # Web search (Web MCP server)
    # ---------------------------
    TAVILY_API_KEY: str = ""
    WEB_MCP_HOST: str = "localhost"
    WEB_MCP_PORT: int = 5556
    WEB_MCP_TOKEN: str = ""
    # Token buckets per tool: sustained rate per minute and burst size.
    WEB_SEARCH_PER_MIN: int = 10
    WEB_SEARCH_BURST: int = 5
    WEB_FETCH_PER_MIN: int = 30
    WEB_FETCH_BURST: int = 10

    # ---------------------------
    # A2A agents (each its own service; the supervisor is the A2A client)
    # ---------------------------
    A2A_HOST: str = "localhost"
    # One static bearer token shared by every agent (same model as the MCP servers).
    A2A_TOKEN: str = ""
    # Default per-call deadline; a TaskRequest's constraints.timeout_s overrides it.
    A2A_TIMEOUT_S: float = 60.0
    SQL_AGENT_PORT: int = 7001
    WEB_AGENT_PORT: int = 7002
    SYNTHESIS_AGENT_PORT: int = 7003
    ECHO_AGENT_PORT: int = 7009  # plumbing test agent (Phase C)

    # Discovery: the registry fetches each URL's agent card. JSON list in .env.
    AGENT_URLS: list[str] = [
        "http://localhost:7001",
        "http://localhost:7002",
        "http://localhost:7003",
    ]

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
