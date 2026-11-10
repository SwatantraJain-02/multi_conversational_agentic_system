"""In-memory stand-ins for Pinecone and the embedding model, for the memory tests.

`FakeBackend` has the methods of `multi_agenti_conversation_system.vectorstore` that the memory
stores use, and evaluates the same `$eq` / `$gt` filters. Search is exact and instantly
consistent (the real index lags behind a write, G14: that is covered by the live smoke).
`FakeEmbedder` is a normalised bag of words, so similar sentences score close to 1.
"""

import hashlib
import math
import re
from typing import Any

DIM = 256


class FakeEmbedder:
    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class Doc:
    def __init__(self, doc_id: str, fields: dict[str, Any], score: float | None = None):
        self.id, self._fields, self.score = doc_id, fields, score

    def get(self, key, default=None):
        return self._fields.get(key, default)

    def to_dict(self):
        return {"_id": self.id, **self._fields}


def matches(fields: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    for field, ops in (flt or {}).items():
        value = fields.get(field)
        for op, expected in ops.items():
            if value is None:
                return False
            if op == "$eq" and value != expected:
                return False
            if op == "$gt" and not value > expected:
                return False
    return True


class FakeBackend:
    def __init__(self) -> None:
        self.ns: dict[str, dict[str, dict[str, Any]]] = {}
        self.filters: list[dict[str, Any] | None] = []  # every filter a search used

    def _docs(self, namespace):
        return self.ns.setdefault(namespace, {})

    def upsert(self, namespace, documents):
        for d in documents:
            if any(v is None for v in d.values()):
                raise ValueError("metadata values may not be None (G15)")
            self._docs(namespace)[d["_id"]] = {k: v for k, v in d.items() if k != "_id"}

    def fetch(self, namespace, ids=(), filter=None, limit=100):
        docs = self._docs(namespace)
        found = (
            {i: docs[i] for i in ids if i in docs}
            if ids
            else {i: f for i, f in docs.items() if matches(f, filter)}
        )
        return {
            i: {k: v for k, v in f.items() if k != "embedding"}
            for i, f in found.items()
        }

    def update(self, namespace, doc_id, **fields):
        if doc_id in self._docs(namespace):
            self._docs(namespace)[doc_id].update(fields)

    def dense_search(self, namespace, vector, top_k, filter=None, include_fields=None):
        self.filters.append(filter)
        scored = []
        for doc_id, f in self._docs(namespace).items():
            if matches(f, filter):
                sim = sum(a * b for a, b in zip(vector, f["embedding"], strict=True))
                scored.append(Doc(doc_id, f, sim))
        return sorted(scored, key=lambda d: d.score, reverse=True)[:top_k]

    def delete(self, namespace, ids):
        for i in ids:
            self._docs(namespace).pop(i, None)

    def delete_where(self, namespace, filter):
        for i in [i for i, f in self._docs(namespace).items() if matches(f, filter)]:
            del self._docs(namespace)[i]
