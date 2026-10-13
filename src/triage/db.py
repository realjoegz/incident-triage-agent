"""PostgreSQL: the schema, loading the corpus, and a connection pool for searches."""

from collections.abc import Sequence
from importlib import resources

import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg_pool import AsyncConnectionPool

from triage.embeddings import Embedder
from triage.models import Document


def _schema() -> str:
    return resources.files("triage").joinpath("schema.sql").read_text(encoding="utf-8")


def load_corpus(database_url: str, corpus: Sequence[Document], embedder: Embedder) -> int:
    """Replaces the content of `documents` with the corpus, in one transaction."""
    vectors = embedder.embed([document.text for document in corpus])
    with psycopg.connect(database_url) as connection:
        connection.execute(_schema())
        register_vector(connection)
        with connection.transaction():
            connection.execute("DELETE FROM documents")
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO documents (id, kind, service, title, body, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (
                            document.id,
                            document.kind,
                            document.service,
                            document.title,
                            document.body,
                            vector,
                        )
                        for document, vector in zip(corpus, vectors, strict=True)
                    ],
                )
    return len(corpus)


async def _configure(connection: psycopg.AsyncConnection[object]) -> None:
    await register_vector_async(connection)


def open_pool(database_url: str, max_size: int = 8) -> AsyncConnectionPool:
    """A pool that is not open yet; use it as `async with open_pool(url) as pool`."""
    return AsyncConnectionPool(
        database_url, min_size=1, max_size=max_size, configure=_configure, open=False
    )
