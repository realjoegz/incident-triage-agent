"""The synthetic data set: a fault catalogue, and what is rendered from it."""

from triage.dataset.files import (
    CORPUS_FILE,
    INCIDENTS_FILE,
    LABELS_FILE,
    load_corpus,
    load_incidents,
    load_labelled,
    load_labels,
)

__all__ = [
    "CORPUS_FILE",
    "INCIDENTS_FILE",
    "LABELS_FILE",
    "load_corpus",
    "load_incidents",
    "load_labelled",
    "load_labels",
]
