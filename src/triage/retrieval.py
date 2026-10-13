"""Search over the documents: by vector, by full text, and the two fused.

All three run in PostgreSQL. Ties are broken by document id everywhere, so a ranking is a
function of the data and never of the order in which rows happen to come back.
"""

import re
from collections.abc import Sequence

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from triage.embeddings import Embedder
from triage.models import Document, Evidence, Incident, SearchMode

# How many rows each of the two searches contributes to the fusion.
CANDIDATES = 20

# The constant of reciprocal rank fusion. 60 is the value of the paper that introduced it
# (Cormack, Clarke and Buettcher, 2009) and is not tuned here.
RRF_K = 60

_LOG_PREFIX = re.compile(r"^\S+\s+(?:DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+\S+\s+")

_VECTOR_SQL = """
SELECT id, kind, service, title, body, 1 - (embedding <=> %(vector)s) AS score
FROM documents
ORDER BY embedding <=> %(vector)s, id
LIMIT %(limit)s
"""

# The query is the incident text itself, which runs to dozens of words, so its lexemes are
# joined with OR: a document should rank by how much of the incident it shares, not be dropped
# because it lacks one word. Each lexeme is quoted for the tsquery syntax, where a quote and a
# backslash are doubled. An incident with no lexeme at all gives a NULL query and no row.
_FULLTEXT_SQL = r"""
WITH query AS (
    SELECT NULLIF(
        string_agg(
            '''' || replace(replace(lexeme, '\', '\\'), '''', '''''') || '''',
            ' | '
        ),
        ''
    )::tsquery AS q
    FROM unnest(to_tsvector('english', %(text)s))
)
SELECT d.id, d.kind, d.service, d.title, d.body, ts_rank(d.tsv, query.q, 1) AS score
FROM documents d, query
WHERE d.tsv @@ query.q
ORDER BY score DESC, d.id
LIMIT %(limit)s
"""


def incident_query(incident: Incident) -> str:
    """The text an incident is searched with.

    The alert and the ticket go in as they are. From the logs only the messages go in, each one
    once: a timestamp or a host name says nothing about the cause, and a line that is repeated
    forty times should not count forty times.
    """
    messages: list[str] = []
    for line in incident.log_excerpt.splitlines():
        message = _LOG_PREFIX.sub("", line.strip())
        if message and message not in messages:
            messages.append(message)
    parts = [incident.alert.strip(), incident.ticket_text.strip(), *messages]
    return "\n".join(part for part in parts if part)


def fuse(rankings: Sequence[Sequence[str]], limit: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: a document scores 1 / (60 + rank) in each ranking that has it."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit]


class Retriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder) -> None:
        self._pool = pool
        self._embedder = embedder

    async def search(self, query: str, mode: SearchMode, limit: int) -> list[Evidence]:
        if not query.strip() or limit < 1:
            return []
        if mode == "vector":
            return _ranked(await self._vector(query, limit))
        if mode == "fulltext":
            return _ranked(await self._fulltext(query, limit))
        by_vector = await self._vector(query, CANDIDATES)
        by_text = await self._fulltext(query, CANDIDATES)
        rows = {str(row["id"]): row for row in [*by_vector, *by_text]}
        fused = fuse(
            [[str(row["id"]) for row in by_vector], [str(row["id"]) for row in by_text]], limit
        )
        return _ranked([{**rows[doc_id], "score": score} for doc_id, score in fused])

    async def get(self, doc_id: str) -> Document | None:
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                "SELECT id, kind, service, title, body FROM documents WHERE id = %s", (doc_id,)
            )
            row = await cursor.fetchone()
        return Document.model_validate(row) if row else None

    async def _vector(self, query: str, limit: int) -> list[dict[str, object]]:
        vector = self._embedder.embed([query])[0]
        return await self._rows(_VECTOR_SQL, {"vector": vector, "limit": limit})

    async def _fulltext(self, query: str, limit: int) -> list[dict[str, object]]:
        return await self._rows(_FULLTEXT_SQL, {"text": query, "limit": limit})

    async def _rows(self, sql: str, parameters: dict[str, object]) -> list[dict[str, object]]:
        async with (
            self._pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(sql, parameters)
            return list(await cursor.fetchall())


def _ranked(rows: Sequence[dict[str, object]]) -> list[Evidence]:
    return [
        Evidence.model_validate({**row, "rank": rank, "score": float(row["score"])})  # type: ignore[arg-type]
        for rank, row in enumerate(rows, start=1)
    ]
