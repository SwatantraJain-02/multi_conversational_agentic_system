import argparse
import math

import chromadb
from chromadb import Collection

from multi_agenti_conversation_system.config import get_chroma_settings
from multi_agenti_conversation_system.llm import get_embedding_model
from multi_agenti_conversation_system.vectorstore import build_chroma_client

COLLECTION_NAME = "phase1_check"

S1 = "I am a king and I like to rule the city."
S2 = "The city's ruler collect taxes."
S3 = "India is a democratic country."
QUERY = "Prime Minister of India is Narendra Modi"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def delete_if_exists(client: chromadb.ClientAPI) -> None:
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)


def create_sample_embedding(collection: Collection) -> list[list[float]]:
    # We compute the vectors ourselves and pass them in. If we only passed documents,
    # Chroma would silently use its own default model instead of nomic-embed-text.
    documents = [S1, S2, S3]
    vectors = get_embedding_model().embed_documents(documents)

    collection.add(
        ids=["s1", "s2", "s3"],
        embeddings=vectors,
        documents=documents,
    )

    return vectors


def phase_write(client: chromadb.ClientAPI) -> None:
    print("heartbeat:", client.heartbeat())

    delete_if_exists(client)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    space = collection.configuration["hnsw"]["space"]
    assert space == "cosine", f"Correct space not assigned, got {space!r}"
    print("PASS: collection space is", space)

    vectors = create_sample_embedding(collection)

    # Chroma needs a vector to search with, so embed the question with the same model.
    query_vector = get_embedding_model().embed_query(QUERY)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=3,
        include=["distances", "documents"],
    )

    print(f"\nquery: {QUERY!r}")
    for doc, distance in zip(
        result["documents"][0], result["distances"][0], strict=True
    ):
        print(f"  distance={distance:.4f}  similarity={1 - distance:.4f}  {doc}")

    # The proof: Chroma's number must match the cosine similarity we compute by hand.
    best_doc = result["documents"][0][0]
    best_similarity = 1 - result["distances"][0][0]
    hand_similarity = cosine_similarity(
        query_vector, vectors[[S1, S2, S3].index(best_doc)]
    )
    print(
        f"\nby hand for the closest document: {hand_similarity:.4f}  (Chroma: {best_similarity:.4f})"
    )
    assert math.isclose(hand_similarity, best_similarity, abs_tol=1e-3), (
        "1 - distance != cosine similarity"
    )
    print("PASS: 1 - distance matches the hand-computed cosine similarity")

    assert best_doc == S3, (
        f"expected the India sentence to be closest, got {best_doc!r}"
    )
    print("PASS: the most similar document is the India sentence")

    print(
        "\nNow run: docker compose restart chroma, wait a few seconds, then run with --phase verify"
    )


def phase_verify(client: chromadb.ClientAPI) -> None:
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    assert count == 3, f"expected 3 items after restart, found {count}"
    print(f"PASS: {count} items survived the restart")

    client.delete_collection(COLLECTION_NAME)
    print("cleaned up the test collection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["write", "verify"], default="write")
    args = parser.parse_args()

    client = build_chroma_client(get_chroma_settings())
    if args.phase == "write":
        phase_write(client)
    else:
        phase_verify(client)


if __name__ == "__main__":
    main()
