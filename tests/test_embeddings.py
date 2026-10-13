from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from triage.embeddings import (
    LocalEmbedder,
    Snapshot,
    SnapshotEmbedder,
    SnapshotMiss,
    Vector,
)
from triage.models import Document, LabelledIncident
from triage.retrieval import incident_query
from triage.settings import EMBEDDING_DIMENSIONS

# How far a vector computed on this machine may be from the committed one, as cosine distance.
# The model is quantised, and ONNX Runtime uses different kernels on different CPUs.
DRIFT = 0.01


def _vector(seed: int) -> Vector:
    values = np.random.default_rng(seed).standard_normal(EMBEDDING_DIMENSIONS)
    return (values / np.linalg.norm(values)).astype(np.float32)


def test_a_snapshot_is_read_back_exactly(tmp_path: Path) -> None:
    vectors = {"one": _vector(1), "two": _vector(2)}
    written = Snapshot()
    for text, vector in vectors.items():
        written.put(text, vector)
    written.save(tmp_path / "s.jsonl")

    read = Snapshot.load(tmp_path / "s.jsonl")

    for text, vector in vectors.items():
        loaded = read.get(text)
        assert loaded is not None
        assert loaded.tobytes() == vector.tobytes()
    assert read.get("three") is None


def test_without_a_fallback_a_missing_text_is_an_error() -> None:
    snapshot = Snapshot()
    snapshot.put("known", _vector(1))

    with pytest.raises(SnapshotMiss, match="1 text"):
        SnapshotEmbedder(snapshot).embed(["known", "new"])


def test_only_the_missing_texts_go_to_the_fallback() -> None:
    class Recording:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def embed(self, texts: Sequence[str]) -> list[Vector]:
            self.asked.extend(texts)
            return [_vector(9) for _ in texts]

    snapshot = Snapshot()
    snapshot.put("known", _vector(1))
    fallback = Recording()

    vectors = SnapshotEmbedder(snapshot, fallback).embed(["new", "known", "newer"])

    assert fallback.asked == ["new", "newer"]
    assert vectors[1].tobytes() == _vector(1).tobytes()
    assert vectors[0].tobytes() == vectors[2].tobytes() == _vector(9).tobytes()


def test_the_snapshot_holds_exactly_the_texts_of_the_data(
    snapshot: Snapshot, corpus: list[Document], labelled: list[LabelledIncident]
) -> None:
    texts = [document.text for document in corpus]
    texts += [incident_query(item.incident) for item in labelled]

    assert all(snapshot.get(text) is not None for text in texts)
    assert len(snapshot.vectors) == len(set(texts))


def test_the_model_still_computes_the_vectors_of_the_snapshot(
    local_embedder: LocalEmbedder,
    snapshot: Snapshot,
    corpus: list[Document],
    labelled: list[LabelledIncident],
) -> None:
    """Fails when the text, the model or the library has moved away from the committed file."""
    texts = [document.text for document in corpus]
    texts += [incident_query(item.incident) for item in labelled]

    computed = local_embedder.embed(texts)

    distances = []
    for text, vector in zip(texts, computed, strict=True):
        committed = snapshot.get(text)
        assert committed is not None
        distances.append(1.0 - float(np.dot(committed, vector)))
    assert max(distances) < DRIFT, f"largest cosine distance {max(distances):.5f}"
