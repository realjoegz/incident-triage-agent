import os
from collections.abc import AsyncIterator
from pathlib import Path

import psycopg
import pytest

from triage import db
from triage.dataset import load_corpus, load_labelled
from triage.embeddings import SNAPSHOT_FILE, LocalEmbedder, Snapshot, SnapshotEmbedder
from triage.models import Document, LabelledIncident
from triage.retrieval import Retriever
from triage.settings import Settings

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(data_dir=REPO / "data", eval_dir=REPO / "eval")


@pytest.fixture(scope="session")
def corpus(settings: Settings) -> list[Document]:
    return load_corpus(settings.data_dir)


@pytest.fixture(scope="session")
def labelled(settings: Settings) -> list[LabelledIncident]:
    return load_labelled(settings.data_dir)


@pytest.fixture(scope="session")
def snapshot(settings: Settings) -> Snapshot:
    return Snapshot.load(settings.data_dir / SNAPSHOT_FILE)


@pytest.fixture(scope="session")
def loaded_database(settings: Settings, corpus: list[Document], snapshot: Snapshot) -> str:
    """The database of compose.yaml with the corpus in it.

    On a developer machine a test that needs it is skipped while the database is down. In CI
    that would let a broken build pass, so there it is a failure.
    """
    try:
        psycopg.connect(settings.database_url, connect_timeout=3).close()
    except psycopg.OperationalError as error:
        message = (
            f"PostgreSQL is not reachable; run `docker compose up --detach --wait db` ({error})"
        )
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    db.load_corpus(settings.database_url, corpus, SnapshotEmbedder(snapshot))
    return settings.database_url


@pytest.fixture
async def retriever(loaded_database: str, snapshot: Snapshot) -> AsyncIterator[Retriever]:
    """Searches as the evaluation does: only with vectors from the snapshot."""
    async with db.open_pool(loaded_database) as pool:
        yield Retriever(pool, SnapshotEmbedder(snapshot))


@pytest.fixture(scope="session")
def local_embedder(settings: Settings) -> LocalEmbedder:
    return LocalEmbedder(settings.embedding_cache_dir)


@pytest.fixture
async def live_retriever(
    loaded_database: str, snapshot: Snapshot, local_embedder: LocalEmbedder
) -> AsyncIterator[Retriever]:
    """Searches as the API does: text that the snapshot does not hold goes to the model."""
    async with db.open_pool(loaded_database) as pool:
        yield Retriever(pool, SnapshotEmbedder(snapshot, local_embedder))
