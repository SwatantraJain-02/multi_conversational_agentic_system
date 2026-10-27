"""Chroma adapter: one HTTP client per process, collections that measure cosine distance."""

import logging

import chromadb
import httpx
from chromadb import Collection

from multi_agenti_conversation_system.config import ChromaSettings

logger = logging.getLogger(__name__)


def build_chroma_client(cfg: ChromaSettings) -> chromadb.ClientAPI:
    client = chromadb.HttpClient(host=cfg.CHROMADB_HOST, port=cfg.CHROMADB_PORT)
    _set_request_timeout(client, cfg.CHROMA_TIMEOUT_S)
    return client


def _set_request_timeout(client: chromadb.ClientAPI, seconds: float) -> None:
    # chromadb 1.0.x builds its httpx session with timeout=None and has no setting to change
    # it, so a hung server would block a tool call forever. The session is private API; the
    # chromadb<1.1 pin in pyproject.toml keeps this stable, and we warn if it ever moves.
    session = getattr(getattr(client, "_server", None), "_session", None)
    if isinstance(session, httpx.Client):
        session.timeout = httpx.Timeout(seconds)
    else:
        logger.warning(
            "could not set the Chroma request timeout; requests are unbounded"
        )


def get_collection(client: chromadb.ClientAPI, name: str) -> Collection:
    """Get or create a collection that measures cosine distance (Chroma defaults to L2).

    Always pass `embeddings=` / `query_embeddings=` yourself. These collections have a
    default embedding function that would otherwise download a different model.
    """
    return client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
    )
