from langchain_ollama import OllamaEmbeddings
from langchain_openrouter import ChatOpenRouter

from multi_agenti_conversation_system.config import settings


def get_chat_model():
    llm = ChatOpenRouter(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.CHAT_MODEL,
        temperature=settings.TEMPERATURE,
        max_retries=4,
    )

    return llm


def get_embedding_model():
    embedder = OllamaEmbeddings(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.EMBED_MODEL,
    )

    return embedder
