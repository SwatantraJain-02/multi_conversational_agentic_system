from langchain_ollama import OllamaEmbeddings
from langchain_openrouter import ChatOpenRouter

from multi_agenti_conversation_system.config import (
    get_chat_settings,
    get_embedding_settings,
)


def get_chat_model():
    cfg = get_chat_settings()
    return ChatOpenRouter(
        api_key=cfg.OPENROUTER_API_KEY,
        model=cfg.CHAT_MODEL,
        temperature=cfg.TEMPERATURE,
        max_retries=cfg.CHAT_MAX_RETRIES,
    )


def get_embedding_model():
    cfg = get_embedding_settings()
    return OllamaEmbeddings(base_url=cfg.OLLAMA_BASE_URL, model=cfg.EMBED_MODEL)
