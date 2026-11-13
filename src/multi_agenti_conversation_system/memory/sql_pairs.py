"""Learned SQL pairs: validated question -> SQL examples the SQL agent keeps for itself.

Episodic memory shared by all users (`scope=global`): Pinecone namespace `episodes`,
`kind=sql_pair`. Unlike the seed few-shots (written by hand, served by the DB MCP server),
these are added after a query ran successfully, and they only help the model WRITE the next
query: the query still runs against live data (that is what separates them from the
semantic cache, which returns a finished answer).

Retrieval is hybrid (BM25 + dense, fused with RRF), the same function as the seed few-shots.

Hygiene, all enforced here or by the caller:
- a pair is identified by the *fingerprint* of its SQL with the literals blanked out, so
  "details of product 100" and "... product 150" share one entry (the newest wins) instead
  of piling up near-duplicates;
- `schema_version` is stored, and only pairs for the current schema are returned, so a
  schema change retires old pairs automatically;
- nothing is written when EVAL_MODE is on (G23: eval and smoke runs must not pollute the
  shared namespace);
- the caller never writes a pair for a query that touched PII, was rejected, or was refused.
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from multi_agenti_conversation_system import vectorstore
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.llm import get_embedding_model
from multi_agenti_conversation_system.retrieval.hybrid import (
    DenseHit,
    candidate_depth,
    hybrid_search,
    keyword_query,
)

logger = logging.getLogger(__name__)

NAMESPACE = vectorstore.EPISODES
KIND = "sql_pair"
SCOPE = "global"
FIELDS = ("text", "sql", "domain", "tables")


@dataclass(frozen=True)
class Pair:
    question: str
    sql: str
    domain: str | None = None


def fingerprint(sql: str) -> str:
    """Stable id of a query's shape: literals blanked, identifiers case-folded, whitespace gone."""
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        for literal in list(tree.find_all(exp.Literal)):
            literal.replace(exp.Placeholder())
        canonical = tree.sql(dialect="postgres", normalize=True)
    except SqlglotError:
        canonical = re.sub(r"\s+", " ", sql).strip().lower()
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def pair_id(sql: str) -> str:
    return f"sqlpair-{fingerprint(sql)}"


def _filter(schema_version: str, domain: str | None) -> dict[str, Any] | None:
    return vectorstore.eq_filter(
        kind=KIND, scope=SCOPE, schema_version=schema_version, domain=domain
    )


def search(
    question: str, k: int, schema_version: str, domain: str | None = None
) -> list[Pair]:
    """The k learned pairs most relevant to the question (blocking: runs two Pinecone searches)."""
    flt = _filter(schema_version, domain)
    found: dict[str, Pair] = {}

    def remember(match) -> None:
        found[match.id] = Pair(
            match.get("text", ""), match.get("sql", ""), match.get("domain")
        )

    def dense(n: int) -> list[DenseHit]:
        matches = vectorstore.dense_search(
            NAMESPACE,
            get_embedding_model().embed_query(question),
            n,
            filter=flt,
            include_fields=FIELDS,
        )
        for m in matches:
            remember(m)
        return [DenseHit(id=m.id, similarity=round(m.score, 4)) for m in matches]

    def keyword(n: int) -> list[tuple[str, float]]:
        matches = vectorstore.keyword_search(
            NAMESPACE, keyword_query(question), n, filter=flt, include_fields=FIELDS
        )
        for m in matches:
            remember(m)
        return [(m.id, m.score) for m in matches]

    result = hybrid_search(dense, keyword, k, candidate_depth(10**6, k))
    return [
        found[hit.id] for hit in result.hits if hit.id in found and found[hit.id].sql
    ]


def write(
    question: str,
    sql: str,
    tables: list[str],
    domain: str | None,
    schema_version: str,
) -> str | None:
    """Store (or refresh) the pair; returns its id, or None when writes are off (EVAL_MODE)."""
    if settings.EVAL_MODE:
        return None
    doc_id = pair_id(sql)
    vectorstore.upsert(
        NAMESPACE,
        [
            {
                "_id": doc_id,
                vectorstore.KEYWORDS_FIELD: keyword_query(f"{question}\n{sql}"),
                vectorstore.EMBEDDING_FIELD: get_embedding_model().embed_documents(
                    [question]
                )[0],
                "text": question,
                "sql": sql.strip(),
                "kind": KIND,
                "scope": SCOPE,
                "domain": domain,
                "tables": ",".join(sorted(tables)),
                "fingerprint": fingerprint(sql),
                "schema_version": schema_version,
                "created_at": int(time.time()),
            }
        ],
    )
    return doc_id


def delete(sql: str) -> None:
    """Forget a pair (a thumbs-down, or test cleanup)."""
    vectorstore.delete(NAMESPACE, [pair_id(sql)])


class SqlPairStore:
    """The agent-facing, async view of this module (tests swap in a fake with the same methods)."""

    async def search(
        self, question: str, k: int, schema_version: str, domain: str | None = None
    ) -> list[Pair]:
        try:
            return await asyncio.to_thread(search, question, k, schema_version, domain)
        except Exception as exc:  # noqa: BLE001 - learned pairs are a bonus, never a blocker
            logger.warning(
                "learned-pair search failed: %s: %s", type(exc).__name__, exc
            )
            return []

    async def write(
        self,
        question: str,
        sql: str,
        tables: list[str],
        domain: str | None,
        schema_version: str,
    ) -> str | None:
        return await asyncio.to_thread(
            write, question, sql, tables, domain, schema_version
        )
