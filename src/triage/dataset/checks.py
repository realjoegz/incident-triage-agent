"""Checks over the whole data set. Each returns the problems it found, as sentences.

These are the checks that can be exhaustive because they are code: counts, balance, duplicates,
whether the answer key can leak into what the agent reads, and whether the files on disk are
what the generator produces.
"""

import re
from collections import Counter
from pathlib import Path

from triage.dataset.catalogue import Catalogue, load_catalogue
from triage.dataset.files import (
    INCIDENTS_FILE,
    _read_jsonl,
    load_corpus,
    load_labelled,
)
from triage.dataset.generate import render_files
from triage.models import Document, LabelledIncident
from triage.taxonomy import KNOWN_CATEGORIES, Category

EXPECTED_DOCUMENTS = 80
EXPECTED_INCIDENTS = 50
EXPECTED_WITHOUT_EVIDENCE = 12

# What a log code looks like in this catalogue: CHK-POOL-0417, PGW-4012.
CODE = re.compile(r"\b[A-Z]{3,4}(?:-[A-Z]{2,4})?-\d{4}\b")


def stale_files(data_dir: Path) -> list[str]:
    problems = []
    for name, expected in render_files(data_dir).items():
        path = data_dir / name
        if not path.exists():
            problems.append(f"{name} is missing; run `triage data generate`")
        elif path.read_text(encoding="utf-8") != expected:
            problems.append(f"{name} is not what the catalogue renders; run `triage data generate`")
    return problems


def shape(corpus: list[Document], labelled: list[LabelledIncident]) -> list[str]:
    problems = []
    if len(corpus) != EXPECTED_DOCUMENTS:
        problems.append(f"expected {EXPECTED_DOCUMENTS} documents, found {len(corpus)}")
    if len(labelled) != EXPECTED_INCIDENTS:
        problems.append(f"expected {EXPECTED_INCIDENTS} incidents, found {len(labelled)}")
    without_evidence = sum(1 for item in labelled if not item.label.answerable)
    if without_evidence != EXPECTED_WITHOUT_EVIDENCE:
        problems.append(
            f"expected {EXPECTED_WITHOUT_EVIDENCE} incidents without evidence, "
            f"found {without_evidence}"
        )
    for name, ids in (
        ("document", [document.id for document in corpus]),
        ("incident", [item.id for item in labelled]),
    ):
        repeated = [value for value, count in Counter(ids).items() if count > 1]
        if repeated:
            problems.append(f"{name} ids are repeated: {repeated}")
    return problems


def labels(corpus: list[Document], labelled: list[LabelledIncident]) -> list[str]:
    problems = []
    known = {document.id for document in corpus}
    for item in labelled:
        label = item.label
        if label.answerable:
            if label.category is Category.UNKNOWN:
                problems.append(f"{item.id} has evidence but no category")
            if len(label.doc_ids) != 3:
                problems.append(f"{item.id} should name three documents, names {label.doc_ids}")
        else:
            if label.category is not Category.UNKNOWN or label.doc_ids:
                problems.append(f"{item.id} has no evidence but carries a category or documents")
        missing = [doc_id for doc_id in label.doc_ids if doc_id not in known]
        if missing:
            problems.append(f"{item.id} names documents that are not in the corpus: {missing}")
        incident = item.incident
        if not (incident.alert or incident.log_excerpt or incident.ticket_text):
            problems.append(f"{item.id} is empty")
    return problems


def duplicates(labelled: list[LabelledIncident]) -> list[str]:
    seen: dict[str, str] = {}
    problems = []
    for item in labelled:
        text = item.incident.model_dump_json()
        if text in seen:
            problems.append(f"{item.id} repeats {seen[text]} word for word")
        seen[text] = item.id
    return problems


def leakage(
    data_dir: Path, catalogue: Catalogue, corpus: list[Document], labelled: list[LabelledIncident]
) -> list[str]:
    """Nothing the agent reads may carry the answer key."""
    problems = []
    for row in _read_jsonl(data_dir / INCIDENTS_FILE):
        if set(row) != {"id", "alert", "log_excerpt", "ticket_text"}:
            problems.append(f"{INCIDENTS_FILE} row {row.get('id')} has fields beyond the input")

    # `unknown` is left out: it is also an ordinary word.
    category_names = [category.value for category in KNOWN_CATEGORIES]
    fault_ids = [fault.id for fault in catalogue.faults] + [novel.id for novel in catalogue.novel]
    readable = {document.id: document.text for document in corpus} | {
        item.id: item.incident.model_dump_json() for item in labelled
    }
    for where, text in readable.items():
        for name in category_names:
            if name in text:
                problems.append(f"{where} contains the category name {name}")
        for fault_id in fault_ids:
            if re.search(rf"\b{fault_id}\b", text):
                problems.append(f"{where} contains the catalogue id {fault_id}")

    corpus_text = "\n".join(document.text for document in corpus)
    for novel in catalogue.novel:
        for code in CODE.findall(novel.signature):
            if code in corpus_text:
                problems.append(
                    f"the corpus contains {code}, the code of the undocumented {novel.id}"
                )

    signatures = {fault.id: fault.signature for fault in catalogue.faults}
    signatures |= {novel.id: novel.signature for novel in catalogue.novel}
    for item in labelled:
        # Only the signature presentation shows the fault's own log line. The other two may
        # carry one code: that of the unrelated background line.
        if item.label.presentation == "signature":
            continue
        background = item.label.background_fault
        allowed = set(CODE.findall(signatures[background])) if background else set()
        own = set(CODE.findall(signatures[item.label.fault_id]))
        found = set(CODE.findall(item.incident.log_excerpt + item.incident.ticket_text))
        if found & own:
            problems.append(f"{item.id} should not show its fault's code {sorted(found & own)}")
        if item.label.presentation == "paraphrase" and found - allowed:
            problems.append(f"{item.id} is a paraphrase but contains {sorted(found - allowed)}")
    return problems


def run_all(data_dir: Path) -> list[str]:
    problems = stale_files(data_dir)
    if problems:
        return problems
    catalogue = load_catalogue(data_dir)
    corpus = load_corpus(data_dir)
    labelled = load_labelled(data_dir)
    return [
        *shape(corpus, labelled),
        *labels(corpus, labelled),
        *duplicates(labelled),
        *leakage(data_dir, catalogue, corpus, labelled),
    ]


def summary(data_dir: Path) -> dict[str, object]:
    corpus = load_corpus(data_dir)
    labelled = load_labelled(data_dir)
    with_evidence = [item for item in labelled if item.label.answerable]
    return {
        "documents": len(corpus),
        "documents_by_kind": dict(sorted(Counter(document.kind for document in corpus).items())),
        "incidents": len(labelled),
        "with_evidence": len(with_evidence),
        "without_evidence": len(labelled) - len(with_evidence),
        "by_presentation": dict(
            sorted(Counter(item.label.presentation for item in labelled).items())
        ),
        "by_category": dict(
            sorted(Counter(item.label.category.value for item in with_evidence).items())
        ),
        "majority_category_share": round(
            max(Counter(item.label.category for item in with_evidence).values())
            / len(with_evidence),
            3,
        ),
    }
