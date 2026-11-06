"""Embeddings through OpenRouter's OpenAI-compatible `/embeddings` endpoint.

`langchain_openai.OpenAIEmbeddings` pointed at OpenRouter, with three additions:
- the query/document prefixes the embedding model was trained with (the API takes no
  `input_type` parameter, so the distinction is made in the text);
- an LRU cache of query embeddings, because the `:free` tier has tight per-minute and
  per-day caps (G18) and agents re-embed the same question several times per turn;
- a counter of HTTP 429 responses, so smoke tests can show when the rate limit bites.

Retries with backoff on 429/5xx come from the OpenAI client (`max_retries`).
"""

from collections import OrderedDict

import httpx
from langchain_openai import OpenAIEmbeddings
from pydantic import PrivateAttr

from multi_agenti_conversation_system.config import settings

APP_TITLE = "multi-agent-conversational-system"


class PrefixedOpenAIEmbeddings(OpenAIEmbeddings):
    query_prefix: str = ""
    doc_prefix: str = ""
    query_cache_size: int = 512

    _query_cache: OrderedDict[str, list[float]] = PrivateAttr(
        default_factory=OrderedDict
    )

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs
    ) -> list[list[float]]:
        return super().embed_documents(
            [self.doc_prefix + t for t in texts], chunk_size, **kwargs
        )

    async def aembed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs
    ) -> list[list[float]]:
        return await super().aembed_documents(
            [self.doc_prefix + t for t in texts], chunk_size, **kwargs
        )

    def embed_query(self, text: str, **kwargs) -> list[float]:
        if (cached := self._cached(text)) is not None:
            return cached
        # The parent's embed_query calls embed_documents, so call that directly
        # with the query prefix instead of the document prefix.
        vector = super().embed_documents([self.query_prefix + text], **kwargs)[0]
        return self._remember(text, vector)

    async def aembed_query(self, text: str, **kwargs) -> list[float]:
        if (cached := self._cached(text)) is not None:
            return cached
        vectors = await super().aembed_documents([self.query_prefix + text], **kwargs)
        return self._remember(text, vectors[0])

    def dimension(self) -> int:
        """Output size of the model, measured with one (cached) request."""
        return len(self.embed_query("dimension probe"))

    def _cached(self, text: str) -> list[float] | None:
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        return None

    def _remember(self, text: str, vector: list[float]) -> list[float]:
        self._query_cache[text] = vector
        if len(self._query_cache) > self.query_cache_size:
            self._query_cache.popitem(last=False)
        return vector


class RateLimitCounter:
    """httpx response hook that counts HTTP 429s (each one is retried by the client)."""

    def __init__(self) -> None:
        self.responses = 0
        self.rate_limited = 0

    def __call__(self, response: httpx.Response) -> None:
        self.responses += 1
        self.rate_limited += response.status_code == 429


rate_limits = RateLimitCounter()


def build_embeddings() -> PrefixedOpenAIEmbeddings:
    return PrefixedOpenAIEmbeddings(
        model=settings.EMBED_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        default_headers={"X-OpenRouter-Title": APP_TITLE},
        # OpenRouter takes raw text: don't tokenize with tiktoken and send token ids.
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
        chunk_size=settings.EMBED_BATCH_SIZE,
        max_retries=4,
        http_client=httpx.Client(event_hooks={"response": [rate_limits]}),
        query_prefix=settings.EMBED_QUERY_PREFIX,
        doc_prefix=settings.EMBED_DOC_PREFIX,
    )
