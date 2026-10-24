import chromadb
from chromadb import Collection

from multi_agenti_conversation_system.config import settings


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.HttpClient(host=settings.CHROMADB_HOST, port=settings.CHROMADB_PORT)


def get_collection(name: str) -> Collection:
    """Get or create a collection that measures cosine distance (Chroma defaults to L2).

    Always pass `embeddings=` / `query_embeddings=` yourself. These collections have a
    default embedding function that would otherwise download a different model.
    """
    return get_chroma_client().get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
    )
