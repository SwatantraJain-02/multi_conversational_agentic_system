"""Typed configuration, read from environment variables and the project-root .env file.

Settings are split by concern, so each process needs only the variables it uses: the DB MCP
server never needs OPENROUTER_API_KEY, and the unit tests need nothing at all. Nothing is read
at import time. Composition roots (server main/create_app, script main functions) call the
get_*() getters and pass the values on; everything else receives its settings as arguments.

Set ENV_FILE to load a different file (CI and tests point it at a file that does not exist).
"""

import os
from functools import cache
from pathlib import Path
from typing import Literal

from psycopg.conninfo import make_conninfo
from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Source-tree root. Only a default: DATA_DIR and ENV_FILE override it for installed copies.
PROJECT_DIR = Path(__file__).resolve().parents[2]


class _Base(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")


class PathSettings(_Base):
    DATA_DIR: Path = PROJECT_DIR / "data"


class NorthwindDbSettings(_Base):
    """The read-only connection to the Northwind business database."""

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = Field(5432, gt=0)
    NORTHWIND_DB: str = "northwind"
    NORTHWIND_READONLY_USER: str = "agent_readonly"
    AGENT_READONLY_PASSWORD: SecretStr
    MAX_ROWS: int = Field(100, ge=1)
    DB_POOL_MIN_SIZE: int = Field(1, ge=1)
    DB_POOL_MAX_SIZE: int = Field(5, ge=1)
    DB_POOL_OPEN_TIMEOUT_S: float = Field(10, gt=0)
    DB_POOL_CHECKOUT_TIMEOUT_S: float = Field(30, gt=0)
    DB_CONNECT_TIMEOUT_S: int = Field(5, gt=0)  # libpq wants whole seconds

    # Field validators (not model validators) so an error message never echoes the whole
    # input, which includes the password.
    @field_validator("DB_POOL_MAX_SIZE")
    @classmethod
    def _check_pool_bounds(cls, value: int, info: ValidationInfo) -> int:
        if value < info.data.get("DB_POOL_MIN_SIZE", 1):
            raise ValueError("DB_POOL_MAX_SIZE must be >= DB_POOL_MIN_SIZE")
        return value

    @property
    def conninfo(self) -> str:
        return make_conninfo(
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            dbname=self.NORTHWIND_DB,
            user=self.NORTHWIND_READONLY_USER,
            password=self.AGENT_READONLY_PASSWORD.get_secret_value(),
            connect_timeout=self.DB_CONNECT_TIMEOUT_S,
        )


class ChromaSettings(_Base):
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = Field(5430, gt=0)
    CHROMA_TIMEOUT_S: float = Field(10, gt=0)


class EmbeddingSettings(_Base):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBED_MODEL: str = "nomic-embed-text"
    # nomic-embed-text is trained with these task prefixes; set both to "" for other models.
    EMBED_DOC_PREFIX: str = "search_document: "
    EMBED_QUERY_PREFIX: str = "search_query: "
    OLLAMA_TIMEOUT_S: float = Field(30, gt=0)
    EMBED_MAX_RETRIES: int = Field(2, ge=0)


class ChatModelSettings(_Base):
    CHAT_PROVIDER: Literal["openrouter", "ollama"] = "openrouter"
    CHAT_MODEL: str
    OPENROUTER_API_KEY: SecretStr | None = Field(None, validate_default=True)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    TEMPERATURE: float = Field(0.0, ge=0)
    CHAT_MAX_RETRIES: int = Field(4, ge=0)

    @field_validator("OPENROUTER_API_KEY")
    @classmethod
    def _check_provider_key(
        cls, value: SecretStr | None, info: ValidationInfo
    ) -> SecretStr | None:
        if info.data.get("CHAT_PROVIDER") == "openrouter" and not value:
            raise ValueError("required when CHAT_PROVIDER=openrouter")
        return value


class DbMcpSettings(_Base):
    """Where the DB MCP server listens, and the bearer token clients must send."""

    DB_MCP_HOST: str = "localhost"
    DB_MCP_PORT: int = Field(5555, gt=0)
    # Empty means no auth; the server only allows that on a loopback address.
    DB_MCP_TOKEN: SecretStr = SecretStr("")

    @property
    def base_url(self) -> str:
        return f"http://{self.DB_MCP_HOST}:{self.DB_MCP_PORT}"

    @property
    def url(self) -> str:
        return f"{self.base_url}/mcp"


class DbToolLimits(_Base):
    """Input limits the DB MCP tools enforce on what the agent sends."""

    MCP_MAX_SQL_CHARS: int = Field(5000, ge=1)
    MCP_MAX_QUESTION_CHARS: int = Field(500, ge=1)
    MCP_MAX_K: int = Field(10, ge=1)


class ObservabilitySettings(_Base):
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    OTEL_SERVICE_NAME: str = "northwind-db-mcp"
    # Traces go to this OTLP/HTTP endpoint (e.g. http://localhost:4318) when set.
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_CONSOLE_EXPORT: bool = False


def _load[T: _Base](cls: type[T]) -> T:
    return cls(_env_file=os.environ.get("ENV_FILE", PROJECT_DIR / ".env"))  # type: ignore[call-arg]


@cache
def get_path_settings() -> PathSettings:
    return _load(PathSettings)


@cache
def get_db_settings() -> NorthwindDbSettings:
    return _load(NorthwindDbSettings)


@cache
def get_chroma_settings() -> ChromaSettings:
    return _load(ChromaSettings)


@cache
def get_embedding_settings() -> EmbeddingSettings:
    return _load(EmbeddingSettings)


@cache
def get_chat_settings() -> ChatModelSettings:
    return _load(ChatModelSettings)


@cache
def get_db_mcp_settings() -> DbMcpSettings:
    return _load(DbMcpSettings)


@cache
def get_db_tool_limits() -> DbToolLimits:
    return _load(DbToolLimits)


@cache
def get_observability_settings() -> ObservabilitySettings:
    return _load(ObservabilitySettings)
