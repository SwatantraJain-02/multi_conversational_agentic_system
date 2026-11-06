from functools import cache

from langchain_openrouter import ChatOpenRouter

from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.embeddings import (
    PrefixedOpenAIEmbeddings,
    build_embeddings,
)


def get_chat_model():
    llm = ChatOpenRouter(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.CHAT_MODEL,
        temperature=settings.TEMPERATURE,
        max_retries=4,
    )

    return llm


@cache
def get_embedding_model() -> PrefixedOpenAIEmbeddings:
    # One instance per process, so the HTTP connection and the query cache are shared.
    return build_embeddings()
