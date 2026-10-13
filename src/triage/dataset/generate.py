"""Renders the fault catalogue into the corpus, the incidents and their labels.

Nothing here is random. Every choice that looks arbitrary (a date, a host, which faults get a
second incident) is taken from a SHA-256 of a fixed seed and the thing being chosen, so the
output is the same on every machine and Python version, and a test can require that the files on
disk are what this module produces.

The label of an incident is not read off its text. The text is rendered from a catalogue fault,
and the fault's category is the label.
"""

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from triage.dataset.catalogue import Catalogue, Fault, NovelFault, load_catalogue
from triage.dataset.files import (
    CORPUS_FILE,
    INCIDENTS_FILE,
    LABELS_FILE,
    dump_jsonl,
    incident_rows,
)
from triage.models import Document, Incident, Label, LabelledIncident
from triage.taxonomy import Category

SEED = "larkhill-1"

Presentation = Literal["signature", "paraphrase", "downstream"]
PRESENTATIONS: tuple[Presentation, ...] = ("signature", "paraphrase", "downstream")

# How many of the 24 known faults get a second incident. 24 + 14 = 38 incidents with evidence.
FAULTS_WITH_TWO_INCIDENTS = 14


def _pick(*parts: object) -> int:
    digest = hashlib.sha256("|".join([SEED, *map(str, parts)]).encode()).hexdigest()
    return int(digest[:12], 16)


# ----------------------------------------------------------------------------------------------
# Corpus


def _past_incident_dates(faults: list[Fault]) -> dict[tuple[str, int], date]:
    """The first occurrence of a fault is in 2025 and the second in 2026, as its triggers say."""
    dates = {}
    for fault in faults:
        dates[fault.id, 0] = date(2025, 1, 10) + timedelta(days=_pick("date", fault.id, 0) % 320)
        dates[fault.id, 1] = date(2026, 1, 5) + timedelta(days=_pick("date", fault.id, 1) % 265)
    return dates


def _past_incident_ids(faults: list[Fault]) -> dict[tuple[str, int], str]:
    """Numbers rise with the date within a year, as they would in an incident tracker."""
    dates = _past_incident_dates(faults)
    ids = {}
    for year in (2025, 2026):
        in_year = sorted((when, key) for key, when in dates.items() if when.year == year)
        for rank, (_, key) in enumerate(in_year):
            number = 20 + rank * 23 + _pick("number", *key) % 19
            ids[key] = f"INC-{year}-{number:04d}"
    return ids


def _runbook_ids(catalogue: Catalogue) -> dict[str, str]:
    ids = {}
    for service in catalogue.services:
        of_service = [fault for fault in catalogue.faults if fault.service == service.name]
        for position, fault in enumerate(of_service, start=1):
            ids[fault.id] = f"RB-{service.name}-{position:02d}"
    return ids


def gold_documents(catalogue: Catalogue) -> dict[str, list[str]]:
    """The three documents that hold the evidence for each known fault."""
    incident_ids = _past_incident_ids(catalogue.faults)
    runbook_ids = _runbook_ids(catalogue)
    return {
        fault.id: [incident_ids[fault.id, 0], incident_ids[fault.id, 1], runbook_ids[fault.id]]
        for fault in catalogue.faults
    }


def build_corpus(catalogue: Catalogue) -> list[Document]:
    dates = _past_incident_dates(catalogue.faults)
    incident_ids = _past_incident_ids(catalogue.faults)
    runbook_ids = _runbook_ids(catalogue)
    documents = [
        Document(
            id=f"RB-{service.name}-00",
            kind="runbook",
            service=service.name,
            title=f"{service.name}: overview and escalation",
            body=f"{service.overview} {service.escalation}",
        )
        for service in catalogue.services
    ]
    for fault in catalogue.faults:
        signals = f'the alert "{fault.alert}" fired, and the logs showed "{fault.signature}".'
        for occurrence in (0, 1):
            first_id = incident_ids[fault.id, 0]
            severity = ("SEV1", "SEV2", "SEV3")[_pick("severity", fault.id, occurrence) % 3]
            happened = fault.symptoms
            title = fault.title
            resolution = fault.resolution
            if occurrence == 1:
                happened = f"The same pattern as {first_id}. {fault.symptoms}"
                title = f"Repeat of {first_id}: {fault.title}"
                # The resolution names a lasting fix, so a second occurrence has to say why
                # there was one.
                resolution = f"What was done after {first_id} was no longer in place. {resolution}"
            documents.append(
                Document(
                    id=incident_ids[fault.id, occurrence],
                    kind="incident",
                    service=fault.service,
                    title=title,
                    body="\n".join(
                        [
                            f"Service: {fault.service}. "
                            f"Date: {dates[fault.id, occurrence].isoformat()}. "
                            f"Severity: {severity}.",
                            f"What happened: {happened}",
                            f"Signals: {signals}",
                            f"Root cause: {fault.cause} "
                            f"The trigger was {fault.triggers[occurrence]}.",
                            f"Resolution: {resolution}",
                        ]
                    ),
                )
            )
        steps = [f"{number}. {step}" for number, step in enumerate(fault.diagnose, start=1)]
        documents.append(
            Document(
                id=runbook_ids[fault.id],
                kind="runbook",
                service=fault.service,
                title=fault.title,
                body="\n".join(
                    [
                        f"Applies to: {fault.service}.",
                        f"Use this section when: {fault.symptoms}",
                        f'Signals: the alert "{fault.alert}", and log lines such as '
                        f'"{fault.signature}".',
                        f"Why it happens: {fault.cause}",
                        "Steps:",
                        *steps,
                    ]
                ),
            )
        )
    return sorted(documents, key=lambda document: document.id)


# ----------------------------------------------------------------------------------------------
# Incidents


def _log_lines(key: str, clock: str | None, lines: list[tuple[str, str, str]]) -> str:
    """Renders (level, service, message) triples as a log excerpt with rising timestamps."""
    day = date(2026, 10, 1) + timedelta(days=_pick("day", key) % 12)
    if clock is None:
        clock = f"{_pick('hour', key) % 24:02d}:{_pick('minute', key) % 60:02d}"
    moment = datetime.fromisoformat(f"{day.isoformat()}T{clock}:00")
    rendered = []
    for position, (level, service, message) in enumerate(lines):
        moment += timedelta(seconds=7 + _pick("gap", key, position) % 34)
        host = f"{service}-{1 + _pick('host', key, position) % 4}"
        rendered.append(f"{moment.isoformat()}Z {level} {host} {message}")
    return "\n".join(rendered)


def _background(catalogue: Catalogue, fault: Fault | NovelFault, key: str) -> Fault | None:
    """The unrelated fault whose log line is mixed into this incident, for half of them.

    A real excerpt is not clean: it carries the odd declined card or deadlock victim that has
    nothing to do with the page. The line comes from a fault marked `blip` on a service that
    this incident does not involve, and it appears once, as a warning.
    """
    if _pick("background", key) % 2:
        return None
    involved = {fault.service, fault.downstream if isinstance(fault, Fault) else ""}
    pool = [other for other in catalogue.faults if other.blip and other.service not in involved]
    return pool[_pick("background-fault", key) % len(pool)]


def _render(
    catalogue: Catalogue, fault: Fault | NovelFault, presentation: Presentation
) -> tuple[Incident, Fault | None]:
    key = f"{fault.id}:{presentation}"
    noise = catalogue.noise

    def quiet(service: str, position: int) -> tuple[str, str, str]:
        return ("INFO", service, noise[_pick("noise", key, position) % len(noise)])

    signature = ("ERROR", fault.service, fault.signature)
    generic = ("WARN", fault.service, fault.generic_log)
    clock = fault.time if isinstance(fault, Fault) else None

    if presentation == "signature":
        # Paged by the alert: the fault's own log line is in the excerpt, and nobody has
        # written a ticket yet.
        alert, ticket = fault.alert, ""
        lines = [quiet(fault.service, 0), signature, generic, signature, quiet(fault.service, 4)]
    elif presentation == "paraphrase":
        # Raised by a person before any alert fired: the ticket describes what users see, and
        # the logs show the effect without the line that names the fault.
        alert, ticket = "", fault.tickets[0]
        lines = [quiet(fault.service, 0), generic, generic, quiet(fault.service, 3), generic]
    else:
        if not isinstance(fault, Fault):
            raise ValueError("only a known fault has a downstream service")
        # Seen from the service that depends on the faulty one: its alert and its log lines.
        # The faulty service's own line is not in the excerpt.
        effect = ("ERROR", fault.downstream, fault.downstream_log)
        alert = f"{fault.downstream}: elevated errors that involve {fault.service}"
        ticket = fault.tickets[1]
        lines = [effect, quiet(fault.downstream, 1), effect, generic, effect]

    background = _background(catalogue, fault, key)
    if background is not None:
        lines.insert(
            1 + _pick("background-line", key) % (len(lines) - 1),
            ("WARN", background.service, background.signature),
        )
    incident = Incident(alert=alert, log_excerpt=_log_lines(key, clock, lines), ticket_text=ticket)
    return incident, background


def _plan(catalogue: Catalogue) -> list[tuple[Fault | NovelFault, Presentation]]:
    """Which fault is shown in which way. Known faults are spread over the three presentations."""
    plan: list[tuple[Fault | NovelFault, Presentation]] = []
    shuffled = sorted(catalogue.faults, key=lambda fault: _pick("order", fault.id))
    for position, fault in enumerate(shuffled):
        plan.append((fault, PRESENTATIONS[position % 3]))
        if position < FAULTS_WITH_TWO_INCIDENTS:
            plan.append((fault, PRESENTATIONS[(position + 1) % 3]))
    for novel in catalogue.novel:
        plan.append((novel, "signature"))
        plan.append((novel, "paraphrase"))
    return sorted(plan, key=lambda item: _pick("incident", item[0].id, item[1]))


def build_incidents(catalogue: Catalogue) -> list[LabelledIncident]:
    gold = gold_documents(catalogue)
    labelled = []
    for number, (fault, presentation) in enumerate(_plan(catalogue), start=1):
        incident_id = f"T{number:02d}"
        known = isinstance(fault, Fault)
        incident, background = _render(catalogue, fault, presentation)
        labelled.append(
            LabelledIncident(
                id=incident_id,
                incident=incident,
                label=Label(
                    id=incident_id,
                    fault_id=fault.id,
                    answerable=known,
                    category=fault.category if isinstance(fault, Fault) else Category.UNKNOWN,
                    doc_ids=gold[fault.id] if known else [],
                    presentation=presentation,
                    service=fault.service,
                    background_fault=background.id if background else None,
                ),
            )
        )
    return labelled


# ----------------------------------------------------------------------------------------------
# Files


def render_files(data_dir: Path) -> dict[str, str]:
    """The content of the three generated files, by file name."""
    catalogue = load_catalogue(data_dir)
    labelled = build_incidents(catalogue)
    return {
        CORPUS_FILE: dump_jsonl(build_corpus(catalogue)),
        INCIDENTS_FILE: dump_jsonl(incident_rows(labelled)),
        LABELS_FILE: dump_jsonl(item.label for item in labelled),
    }


def write_files(data_dir: Path) -> None:
    for name, content in render_files(data_dir).items():
        (data_dir / name).write_text(content, encoding="utf-8")
