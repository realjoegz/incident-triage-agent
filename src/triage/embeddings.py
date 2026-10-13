"""Text embeddings from a local model, with a committed snapshot for the evaluation.

The model runs on ONNX Runtime, which does not give bit-identical floats on every CPU. A
difference in the sixth decimal is enough to swap two neighbours in a ranking, rank fusion
turns that into a different top five, and a different top five is a different prompt. The
evaluation replays recorded model responses by the hash of the prompt, so it needs retrieval to
be the same on a laptop and on a CI runner. The snapshot gives it that: the vectors of the
corpus and of the labelled incidents are computed once and committed, and a test recomputes
them and fails when the model or the text has drifted away from the file.
"""

import base64
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from triage.settings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

Vector = NDArray[np.float32]

SNAPSHOT_FILE = "embeddings.jsonl"


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """One unit-length vector per text."""
        ...


def _unit(vector: NDArray[np.floating]) -> Vector:
    as_float = np.asarray(vector, dtype=np.float32)
    return (as_float / np.linalg.norm(as_float)).astype(np.float32)


class LocalEmbedder:
    """bge-small-en-v1.5 through fastembed. The model file is fetched on first use."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._model: object | None = None

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=EMBEDDING_MODEL,
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
            )
        vectors = [_unit(vector) for vector in self._model.embed(list(texts))]  # type: ignore[attr-defined]
        for vector in vectors:
            if vector.shape != (EMBEDDING_DIMENSIONS,):
                raise ValueError(f"expected {EMBEDDING_DIMENSIONS} dimensions, got {vector.shape}")
        return vectors


def text_key(text: str) -> str:
    return hashlib.sha256(f"{EMBEDDING_MODEL}\n{text}".encode()).hexdigest()


class Snapshot:
    """Vectors by the hash of the model name and the text."""

    def __init__(self, vectors: dict[str, Vector] | None = None) -> None:
        self.vectors: dict[str, Vector] = vectors or {}

    @classmethod
    def load(cls, path: Path) -> "Snapshot":
        vectors = {}
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line)
                raw = base64.b64decode(row["vector"])
                vectors[row["key"]] = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        return cls(vectors)

    def save(self, path: Path) -> None:
        lines = [
            json.dumps(
                {
                    "key": key,
                    "vector": base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii"),
                }
            )
            for key, vector in sorted(self.vectors.items())
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def get(self, text: str) -> Vector | None:
        return self.vectors.get(text_key(text))

    def put(self, text: str, vector: Vector) -> None:
        self.vectors[text_key(text)] = vector


class SnapshotMiss(LookupError):
    pass


class SnapshotEmbedder:
    """Answers from the snapshot. A text that is not in it goes to `fallback`, or is an error.

    The evaluation runs without a fallback, so a text the snapshot does not hold stops the run
    instead of quietly getting a vector that another machine would compute differently.
    """

    def __init__(self, snapshot: Snapshot, fallback: Embedder | None = None) -> None:
        self._snapshot = snapshot
        self._fallback = fallback

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        found = [self._snapshot.get(text) for text in texts]
        missing = [text for text, vector in zip(texts, found, strict=True) if vector is None]
        if missing and self._fallback is None:
            raise SnapshotMiss(
                f"{len(missing)} text(s) are not in the embedding snapshot; "
                "run `triage data embed` after changing the data"
            )
        computed = iter(self._fallback.embed(missing)) if missing and self._fallback else iter(())
        return [vector if vector is not None else next(computed) for vector in found]
