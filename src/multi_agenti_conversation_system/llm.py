"""Chat and embedding model adapters.

LangChain's BaseChatModel and Embeddings are the interfaces the rest of the code depends on;
these builders are the only place that knows which provider sits behind them.
"""

import logging
import random
import time
from collections.abc import Callable
from functools import cache

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openrouter import ChatOpenRouter

from multi_agenti_conversation_system.config import (
    ChatModelSettings,
    EmbeddingSettings,
    get_chat_settings,
    get_embedding_settings,
)

logger = logging.getLogger(__name__)

# Worth retrying: the server was briefly unreachable or slow. Anything else (unknown model,
# bad request) fails the same way on every attempt.
TRANSIENT_ERRORS = (ConnectionError, TimeoutError, httpx.TransportError)


def build_chat_model(cfg: ChatModelSettings) -> BaseChatModel:
    match cfg.CHAT_PROVIDER:
        case "openrouter":
            return ChatOpenRouter(
                api_key=cfg.OPENROUTER_API_KEY,
                model=cfg.CHAT_MODEL,
                temperature=cfg.TEMPERATURE,
                max_retries=cfg.CHAT_MAX_RETRIES,
            )
        case "ollama":
            return ChatOllama(
                base_url=cfg.OLLAMA_BASE_URL,
                model=cfg.CHAT_MODEL,
                temperature=cfg.TEMPERATURE,
            )


def _ollama_embeddings(cfg: EmbeddingSettings) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=cfg.OLLAMA_BASE_URL,
        model=cfg.EMBED_MODEL,
        client_kwargs={"timeout": cfg.OLLAMA_TIMEOUT_S},
    )


def with_retries[T](call: Callable[[], T], retries: int, what: str) -> T:
    """Run call(), retrying transient failures with exponential backoff and jitter."""
    for attempt in range(retries + 1):
        try:
            return call()
        except TRANSIENT_ERRORS as exc:
            if attempt == retries:
                raise
            delay = 0.5 * 2**attempt * random.uniform(0.8, 1.2)
            logger.warning(
                "retrying %s after %s (attempt %d/%d, sleeping %.2fs)",
                what,
                type(exc).__name__,
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


class PrefixedEmbeddings(Embeddings):
    """Adds the model's task prefixes and retries transient failures.

    Some embedding models (nomic-embed-text) retrieve noticeably better when documents and
    queries carry task prefixes. The prefixes belong to the model, so they live in the
    embedding config instead of in the code that stores or searches vectors.
    """

    def __init__(
        self, inner: Embeddings, doc_prefix: str, query_prefix: str, retries: int
    ):
        self.inner = inner
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.retries = retries

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.doc_prefix + t for t in texts]
        return with_retries(
            lambda: self.inner.embed_documents(prefixed),
            self.retries,
            "embed_documents",
        )

    def embed_query(self, text: str) -> list[float]:
        return with_retries(
            lambda: self.inner.embed_query(self.query_prefix + text),
            self.retries,
            "embed_query",
        )


def build_embeddings(cfg: EmbeddingSettings) -> PrefixedEmbeddings:
    """The embedder for storing and searching vectors (the DB MCP server, the index build)."""
    return PrefixedEmbeddings(
        _ollama_embeddings(cfg),
        doc_prefix=cfg.EMBED_DOC_PREFIX,
        query_prefix=cfg.EMBED_QUERY_PREFIX,
        retries=cfg.EMBED_MAX_RETRIES,
    )


def ping_ollama(cfg: EmbeddingSettings, timeout_s: float) -> bool:
    try:
        httpx.get(
            f"{cfg.OLLAMA_BASE_URL}/api/version", timeout=timeout_s
        ).raise_for_status()
    except httpx.HTTPError:
        return False
    return True


# Conveniences for scripts only. Application code receives its models from a composition root.
@cache
def get_chat_model() -> BaseChatModel:
    return build_chat_model(get_chat_settings())


@cache
def get_embedding_model() -> OllamaEmbeddings:
    """The raw embedder, without task prefixes (the smoke checks compare plain vectors)."""
    return _ollama_embeddings(get_embedding_settings())
