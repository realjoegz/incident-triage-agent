"""Reading and writing the data files.

Incidents and labels are separate files on purpose. The agent, the API and the MCP server read
`incidents.jsonl` at most; only the evaluation reads `labels.jsonl`.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel

from triage.models import Document, Incident, Label, LabelledIncident

CORPUS_FILE = "corpus.jsonl"
INCIDENTS_FILE = "incidents.jsonl"
LABELS_FILE = "labels.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def dump_jsonl(rows: Iterable[BaseModel | dict[str, object]]) -> str:
    """One JSON object per line, keys in a fixed order, so that a regenerated file is identical."""
    out = []
    for row in rows:
        data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
        out.append(json.dumps(data, ensure_ascii=False))
    return "\n".join(out) + "\n"


def load_corpus(data_dir: Path) -> list[Document]:
    return [Document.model_validate(row) for row in _read_jsonl(data_dir / CORPUS_FILE)]


def load_incidents(data_dir: Path) -> dict[str, Incident]:
    incidents = {}
    for row in _read_jsonl(data_dir / INCIDENTS_FILE):
        incident_id = str(row.pop("id"))
        incidents[incident_id] = Incident.model_validate(row)
    return incidents


def load_labels(data_dir: Path) -> dict[str, Label]:
    labels = [Label.model_validate(row) for row in _read_jsonl(data_dir / LABELS_FILE)]
    return {label.id: label for label in labels}


def load_labelled(data_dir: Path) -> list[LabelledIncident]:
    incidents = load_incidents(data_dir)
    labels = load_labels(data_dir)
    if incidents.keys() != labels.keys():
        raise ValueError("incidents.jsonl and labels.jsonl do not hold the same ids")
    return [
        LabelledIncident(id=incident_id, incident=incidents[incident_id], label=labels[incident_id])
        for incident_id in sorted(incidents)
    ]


def incident_rows(labelled: Sequence[LabelledIncident]) -> list[dict[str, object]]:
    return [{"id": item.id, **item.incident.model_dump()} for item in labelled]
