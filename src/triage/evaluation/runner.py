"""Runs the evaluation: against the API, or again from the recordings.

Both ways run the same code. The agent searches, builds its prompts, checks its citations and
is graded; the only difference is whether a model call that has no recording is paid for
(`live`) or is the end of the run (a replay). A replay therefore proves two things at once: the
committed results are what the current code computes from the recorded answers, and the current
code still sends exactly the prompts those answers were given to.
"""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from mcp import Client

from triage import db
from triage.agent import Agent
from triage.dataset import load_corpus, load_labelled
from triage.dataset.catalogue import load_catalogue
from triage.embeddings import SNAPSHOT_FILE, Snapshot, SnapshotEmbedder
from triage.evaluation import judge as judging
from triage.evaluation import retrieval_eval
from triage.evaluation.batch import fill_from_batch
from triage.evaluation.metrics import summarise
from triage.evaluation.rows import ErrorRow, Row
from triage.evaluation.variants import VARIANTS, Variant
from triage.llm.base import LLM, LLMError, LLMRequest, LLMResponse
from triage.llm.claude import REQUEST_TIMEOUT_S, ClaudeLLM
from triage.llm.pricing import cost_usd
from triage.llm.recording import (
    Budget,
    BudgetExceeded,
    RecordingLLM,
    RecordingMiss,
    Recordings,
    ReplayLLM,
)
from triage.mcp_server import EvidenceTools, build_server
from triage.models import Document, Incident, LabelledIncident
from triage.runtime import open_retriever
from triage.settings import Settings

REPETITIONS = 3
CONCURRENCY = 4

# A hard limit on one triage, whatever the connection is doing. An attempt that runs into it is
# an error row, never an answer.
CASE_TIMEOUT_S = 300.0

RECORDINGS_DIR = "recordings"
JUDGE_RECORDINGS = "judge"

JudgeMode = Literal["batch", "direct", "none"]


class StaleResults(RuntimeError):
    """A replay asked for a model response that was never recorded."""


@dataclass
class Evaluation:
    rows: list[Row]
    errors: list[ErrorRow]
    controls: list[dict[str, object]]
    summary: dict[str, object]
    # Wall-clock measurements of this machine. Not part of the reproducible results.
    timings: dict[str, float] = field(default_factory=dict)


def is_full_run(
    variants: Sequence[Variant],
    cases: Sequence[LabelledIncident] | None,
    reps: int,
    judge: JudgeMode,
    control_count: int | None,
) -> bool:
    """Whether a run covers everything that the committed results cover.

    Only such a run may replace the results or drop recordings that it did not use. A run with
    fewer incidents, repetitions, variants or controls, or without the judge, used less than
    there is, and what it did not use is still paid for and still wanted.
    """
    return (
        cases is None
        and reps == REPETITIONS
        and tuple(variants) == VARIANTS
        and judge != "none"
        and control_count is None
    )


def sample_cases(labelled: Sequence[LabelledIncident], count: int) -> list[LabelledIncident]:
    """The first `count` incidents when taking turns across evidence and presentation."""
    groups: dict[tuple[bool, str], list[LabelledIncident]] = {}
    for item in sorted(labelled, key=lambda item: item.id):
        groups.setdefault((item.label.answerable, item.label.presentation), []).append(item)
    order = sorted(groups, key=lambda key: (not key[0], key[1]))
    chosen: list[LabelledIncident] = []
    while len(chosen) < min(count, len(labelled)):
        for key in order:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop(0))
    return chosen


async def evaluate(
    settings: Settings,
    *,
    live: bool,
    variants: Sequence[Variant] = VARIANTS,
    cases: Sequence[LabelledIncident] | None = None,
    reps: int = REPETITIONS,
    budget: Budget | None = None,
    judge: JudgeMode = "direct",
    control_count: int | None = None,
    paid_llm: LLM | None = None,
    log: Callable[[str], None] = print,
) -> Evaluation:
    """`cases=None` and `control_count=None` mean all of them.

    Without a budget a live run may spend nothing: it can only reuse what is recorded.
    `paid_llm` stands in for the Claude API in tests.
    """
    data_dir = settings.data_dir
    recordings_dir = settings.eval_dir / RECORDINGS_DIR
    labelled = load_labelled(data_dir)
    labels = {item.id: item.label for item in labelled}
    corpus = load_corpus(data_dir)
    full_run = is_full_run(variants, cases, reps, judge, control_count)
    cases = list(cases) if cases is not None else labelled
    budget = budget or Budget(max_usd=0.0)
    paid = (paid_llm or ClaudeLLM()) if live else None

    # The database is brought to the committed data and vectors first, so that a search here is
    # the search of every other run.
    snapshot = Snapshot.load(data_dir / SNAPSHOT_FILE)
    db.load_corpus(settings.database_url, corpus, SnapshotEmbedder(snapshot))

    def llm_for(recordings: Recordings, rep: int) -> RecordingLLM | ReplayLLM:
        if paid is not None:
            return RecordingLLM(paid, recordings, budget, rep)
        return ReplayLLM(recordings, rep)

    async with (
        open_retriever(settings, snapshot_only=True) as retriever,
        Client(build_server(retriever)) as client,
    ):
        tools = EvidenceTools(client)
        retrieval, timings = await retrieval_eval.measure(tools.search, labelled)

        rows: list[Row] = []
        errors: list[ErrorRow] = []
        # The MCP client runs in a task group, and an exception that leaves its block comes out
        # wrapped in an ExceptionGroup. The two that callers handle are carried out by hand.
        stopped: BudgetExceeded | StaleResults | None = None
        try:
            for variant in variants:
                recordings = Recordings(recordings_dir / f"{variant.name}.jsonl")
                variant_rows, variant_errors, used = await _run_variant(
                    variant, cases, reps, tools, recordings, llm_for
                )
                rows += variant_rows
                errors += variant_errors
                if live:
                    if full_run and not variant_errors:
                        recordings.prune(used)
                    recordings.compact()
                log(
                    f"{variant.name}: {len(variant_rows)} rows, {len(variant_errors)} errors, "
                    f"${budget.spent_usd:.4f} paid so far"
                )
        except (BudgetExceeded, StaleResults) as error:
            stopped = error
    if stopped is not None:
        raise stopped

    by_id = {document.id: document for document in corpus}
    incidents = {item.id: item.incident for item in labelled}
    judge_recordings = Recordings(recordings_dir / f"{JUDGE_RECORDINGS}.jsonl")
    checks = judging.controls(load_catalogue(data_dir), corpus, labelled)
    if control_count is not None:
        checks = checks[:control_count]
    control_results: list[dict[str, object]] = []
    if judge != "none":
        to_judge = [row for row in rows if judging.needs_judging(row)]
        if live and judge == "batch" and paid_llm is None:
            requests = [
                judging.judge_input(row, incidents[row.case_id], by_id).request()
                for row in to_judge
            ]
            requests += [check.input.request() for check in checks]
            failed = await fill_from_batch(
                # No retries of its own: see `fill_from_batch`.
                anthropic.AsyncAnthropic(max_retries=0, timeout=REQUEST_TIMEOUT_S),
                {request.fingerprint(): request for request in requests},
                judge_recordings,
                budget,
                recordings_dir / ".pending-judge-batch",
                log,
            )
            for failure in failed:
                log(f"judge: the batch gave no verdict for {failure}; asking directly")
        judge_llm = llm_for(judge_recordings, 0)
        errors += await _judge_rows(to_judge, incidents, by_id, judge_llm)
        control_results, control_errors = await _judge_controls(checks, judge_llm)
        errors += control_errors
        if live:
            if full_run and not errors:
                judge_recordings.prune(judge_llm.used)
            judge_recordings.compact()
        log(
            f"judge: {len(to_judge)} answers, {len(checks)} controls, "
            f"${budget.spent_usd:.4f} paid so far"
        )

    order = {variant.name: position for position, variant in enumerate(variants)}
    rows.sort(key=lambda row: (order[row.variant], row.case_id, row.rep))
    summary = summarise(rows, labels, variants)
    summary["retrieval"] = retrieval
    summary["judge"] = {
        "model": judging.JUDGE_MODEL,
        "effort": judging.JUDGE_EFFORT,
        "controls": _control_summary(control_results),
    }
    summary["repetitions"] = reps
    summary["incidents"] = len(cases)
    return Evaluation(rows, errors, control_results, summary, timings)


async def _run_variant(
    variant: Variant,
    cases: Sequence[LabelledIncident],
    reps: int,
    tools: EvidenceTools,
    recordings: Recordings,
    llm_for: Callable[[Recordings, int], RecordingLLM | ReplayLLM],
) -> tuple[list[Row], list[ErrorRow], set[tuple[str, int]]]:
    gate = asyncio.Semaphore(CONCURRENCY)
    llms = [llm_for(recordings, rep) for rep in range(reps)]
    rows: list[Row] = []
    errors: list[ErrorRow] = []
    stopped: list[BudgetExceeded] = []

    async def one(item: LabelledIncident, rep: int) -> None:
        async with gate:
            if stopped:
                return
            agent = Agent(llms[rep], tools.search, variant.config())
            try:
                run = await asyncio.wait_for(agent.run(item.incident), CASE_TIMEOUT_S)
            except BudgetExceeded as error:
                # Calls already on their way finish and are recorded; no new one starts.
                stopped.append(error)
            except RecordingMiss as error:
                raise StaleResults(
                    f"{variant.name}, incident {item.id}, repetition {rep}: {error}. The prompt "
                    "that the current code builds has no recorded answer, so the committed "
                    "results do not describe this code. Run `triage eval run` with an API key."
                ) from error
            except LLMError as error:
                errors.append(_error(variant.name, item.id, rep, "api_error", str(error)))
            except TimeoutError:
                detail = f"no answer within {CASE_TIMEOUT_S:.0f} s"
                errors.append(_error(variant.name, item.id, rep, "timeout", detail))
            else:
                rows.append(Row.from_run(variant.name, item.id, rep, run))

    await asyncio.gather(*(one(item, rep) for item in cases for rep in range(reps)))
    if stopped:
        raise stopped[0]
    used: set[tuple[str, int]] = set()
    for llm in llms:
        used |= llm.used
    return rows, errors, used


def _error(variant: str, case_id: str, rep: int, kind: str, detail: str) -> ErrorRow:
    return ErrorRow(variant=variant, case_id=case_id, rep=rep, kind=kind, detail=detail)


async def _ask_judge(llm: LLM, request: LLMRequest) -> LLMResponse:
    try:
        return await llm.complete(request)
    except RecordingMiss as error:
        raise StaleResults(
            f"judge: {error}. An answer or a control has no recorded verdict; "
            "run `triage eval run` with an API key."
        ) from error


async def _verdicts(
    requests: Sequence[LLMRequest], llm: LLM
) -> tuple[dict[str, LLMResponse], dict[str, str]]:
    """One call per distinct request. Returns the responses and the failures, by fingerprint."""
    gate = asyncio.Semaphore(CONCURRENCY)
    distinct = {request.fingerprint(): request for request in requests}
    responses: dict[str, LLMResponse] = {}
    failures: dict[str, str] = {}

    async def one(key: str, request: LLMRequest) -> None:
        try:
            async with gate:
                responses[key] = await _ask_judge(llm, request)
        except LLMError as error:
            failures[key] = str(error)

    await asyncio.gather(*(one(key, request) for key, request in distinct.items()))
    return responses, failures


async def _judge_rows(
    rows: Sequence[Row],
    incidents: Mapping[str, Incident],
    corpus: Mapping[str, Document],
    llm: LLM,
) -> list[ErrorRow]:
    """Sets the verdict on each row. Rows with the same claim and citations share one call,
    one verdict and its cost."""
    keys = [judging.judge_input(row, incidents[row.case_id], corpus).request() for row in rows]
    responses, failures = await _verdicts(keys, llm)
    sharing: dict[str, int] = {}
    for request in keys:
        sharing[request.fingerprint()] = sharing.get(request.fingerprint(), 0) + 1
    errors: list[ErrorRow] = []
    for row, request in zip(rows, keys, strict=True):
        key = request.fingerprint()
        if key in failures:
            errors.append(_error(row.variant, row.case_id, row.rep, "judge_error", failures[key]))
            continue
        response = responses[key]
        row.judge = judging.parse_verdict(response)
        cost = cost_usd(response.model, response.usage, batch=response.batch)
        row.judge_cost_usd = round(cost / sharing[key], 8)
    return errors


async def _judge_controls(
    checks: Sequence[judging.Control], llm: LLM
) -> tuple[list[dict[str, object]], list[ErrorRow]]:
    requests = [check.input.request() for check in checks]
    responses, failures = await _verdicts(requests, llm)
    results: list[dict[str, object]] = []
    errors: list[ErrorRow] = []
    for check, request in zip(checks, requests, strict=True):
        key = request.fingerprint()
        if key in failures:
            errors.append(_error("judge-controls", check.name, 0, "judge_error", failures[key]))
        verdict = judging.parse_verdict(responses[key]) if key in responses else None
        results.append(
            {
                "name": check.name,
                "kind": check.kind,
                "expected": check.expected,
                "supported": verdict.supported if verdict else None,
                "reason": verdict.reason if verdict else None,
            }
        )
    return results, errors


def _control_summary(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    kinds: dict[str, dict[str, int]] = {}
    for result in results:
        counts = kinds.setdefault(str(result["kind"]), {"controls": 0, "as_expected": 0})
        counts["controls"] += 1
        counts["as_expected"] += int(result["supported"] == result["expected"])
    return {
        "controls": len(results),
        "as_expected": sum(counts["as_expected"] for counts in kinds.values()),
        "by_kind": dict(sorted(kinds.items())),
    }


# ----------------------------------------------------------------------------------------------
# Files

ROWS_FILE = "rows.jsonl"
SUMMARY_FILE = "summary.json"
CONTROLS_FILE = "controls.json"
ERRORS_FILE = "errors.jsonl"
RUN_FILE = "run.json"


def _json(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def rendered(evaluation: Evaluation) -> dict[str, str]:
    """The reproducible result files, by name: what a replay must compute again exactly."""
    return {
        ROWS_FILE: "".join(row.model_dump_json() + "\n" for row in evaluation.rows),
        SUMMARY_FILE: _json(evaluation.summary),
        CONTROLS_FILE: _json(evaluation.controls),
    }


def write(evaluation: Evaluation, out_dir: Path, run_info: Mapping[str, object] | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered(evaluation).items():
        (out_dir / name).write_text(content, encoding="utf-8")
    errors = out_dir / ERRORS_FILE
    if evaluation.errors:
        errors.write_text(
            "".join(error.model_dump_json() + "\n" for error in evaluation.errors), encoding="utf-8"
        )
    else:
        errors.unlink(missing_ok=True)
    if run_info is not None:
        (out_dir / RUN_FILE).write_text(_json(run_info), encoding="utf-8")
