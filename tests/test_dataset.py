import json
from collections import Counter
from pathlib import Path

from triage.dataset import INCIDENTS_FILE, checks
from triage.dataset.catalogue import load_catalogue
from triage.dataset.generate import build_corpus, build_incidents, render_files
from triage.models import Document, LabelledIncident
from triage.settings import Settings
from triage.taxonomy import KNOWN_CATEGORIES, Category


def test_the_files_on_disk_pass_every_check(settings: Settings) -> None:
    assert checks.run_all(settings.data_dir) == []


def test_rendering_twice_gives_the_same_bytes(settings: Settings) -> None:
    assert render_files(settings.data_dir) == render_files(settings.data_dir)


def test_a_changed_file_is_reported_as_stale(settings: Settings, tmp_path: Path) -> None:
    for name in ("catalogue.yaml", *render_files(settings.data_dir)):
        (tmp_path / name).write_text((settings.data_dir / name).read_text())
    assert checks.stale_files(tmp_path) == []

    incidents = tmp_path / INCIDENTS_FILE
    incidents.write_text(incidents.read_text().replace("PENDING_REVIEW", "PENDING"))

    assert checks.stale_files(tmp_path) == [
        f"{INCIDENTS_FILE} is not what the catalogue renders; run `triage data generate`"
    ]


def test_what_the_agent_reads_holds_no_part_of_the_answer_key(settings: Settings) -> None:
    with (settings.data_dir / INCIDENTS_FILE).open() as lines:
        rows = [json.loads(line) for line in lines]

    assert {frozenset(row) for row in rows} == {
        frozenset({"id", "alert", "log_excerpt", "ticket_text"})
    }


def test_the_leak_check_catches_a_category_name_in_a_document(
    settings: Settings, corpus: list[Document], labelled: list[LabelledIncident]
) -> None:
    catalogue = load_catalogue(settings.data_dir)
    leaking = corpus[0].model_copy(update={"body": corpus[0].body + " (rate_limiting)"})

    problems = checks.leakage(settings.data_dir, catalogue, [leaking, *corpus[1:]], labelled)

    assert problems == [f"{leaking.id} contains the category name rate_limiting"]


def test_every_known_fault_has_two_past_incidents_and_a_runbook_section(
    settings: Settings,
) -> None:
    catalogue = load_catalogue(settings.data_dir)
    by_id = {document.id: document for document in build_corpus(catalogue)}

    for item in build_incidents(catalogue):
        if item.label.answerable:
            kinds = Counter(by_id[doc_id].kind for doc_id in item.label.doc_ids)
            assert kinds == {"incident": 2, "runbook": 1}, item.id
            assert {by_id[doc_id].service for doc_id in item.label.doc_ids} == {item.label.service}


def test_incidents_without_evidence_have_no_category_and_no_document(
    labelled: list[LabelledIncident],
) -> None:
    without = [item for item in labelled if not item.label.answerable]

    assert len(without) == 12
    assert {item.label.category for item in without} == {Category.UNKNOWN}
    assert all(item.label.doc_ids == [] for item in without)


def test_every_category_has_at_least_two_incidents(labelled: list[LabelledIncident]) -> None:
    counts = Counter(item.label.category for item in labelled if item.label.answerable)

    assert set(counts) == set(KNOWN_CATEGORIES)
    assert min(counts.values()) >= 2
