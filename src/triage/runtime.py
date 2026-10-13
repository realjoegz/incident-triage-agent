"""Builds the parts that the CLI, the API and the evaluation share."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from triage.db import open_pool
from triage.embeddings import (
    SNAPSHOT_FILE,
    Embedder,
    LocalEmbedder,
    Snapshot,
    SnapshotEmbedder,
)
from triage.retrieval import Retriever
from triage.settings import Settings


def build_embedder(settings: Settings, *, snapshot_only: bool) -> Embedder:
    """The snapshot first, then the local model unless `snapshot_only`.

    The evaluation passes `snapshot_only=True`: see `triage.embeddings` for why.
    """
    path = settings.data_dir / SNAPSHOT_FILE
    snapshot = Snapshot.load(path) if path.exists() else Snapshot()
    fallback = None if snapshot_only else LocalEmbedder(settings.embedding_cache_dir)
    return SnapshotEmbedder(snapshot, fallback)


@asynccontextmanager
async def open_retriever(settings: Settings, *, snapshot_only: bool) -> AsyncIterator[Retriever]:
    embedder = build_embedder(settings, snapshot_only=snapshot_only)
    async with open_pool(settings.database_url) as pool:
        yield Retriever(pool, embedder)
