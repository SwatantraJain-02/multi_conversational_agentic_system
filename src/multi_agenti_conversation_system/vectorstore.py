import chromadb
from chromadb import Collection

from multi_agenti_conversation_system.config import get_chroma_settings


def get_chroma_client() -> chromadb.ClientAPI:
    cfg = get_chroma_settings()
    return chromadb.HttpClient(host=cfg.CHROMADB_HOST, port=cfg.CHROMADB_PORT)


def get_collection(name: str) -> Collection:
    """Get or create a collection that measures cosine distance (Chroma defaults to L2).

    Always pass `embeddings=` / `query_embeddings=` yourself. These collections have a
    default embedding function that would otherwise download a different model.
    """
    return get_chroma_client().get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
    )
