"""The gate must pass on a repository that is in order and fail on each way of being worse.

The second half of this file builds such a repository state in a temporary directory, with a
stand-in model in place of the API, and then breaks it one way at a time.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import triage.retrieval
from tests.fakes import RuleLLM, oracle
from triage.agent import prompts
from triage.dataset.catalogue import load_catalogue
from triage.evaluation import gate, report, runner
from triage.llm.recording import Budget
from triage.models import LabelledIncident
from triage.settings import Settings


def variant(**changes: Any) -> dict[str, Any]:
    values = {
        "root_cause_accuracy": 0.90,
        "abstention_without_evidence": 0.75,
        "citation_faithfulness": 0.96,
        "cost_usd_per_incident": 0.0008,
        "unusable_answers": {"refused": 0, "truncated": 0, "invalid_output": 0},
        "answers_judged": 100,
        "answers_to_judge": 100,
    }
    return {**values, **changes}


def summary(**changes: Any) -> dict[str, Any]:
    return {
        "retrieval": {"hybrid": {"recall_at_5": 0.9035, "hit_at_5": 0.9737}},
        "variants": {"hybrid-haiku": variant(**changes)},
        "judge": {"controls": {"controls": 144, "as_expected": 144}},
    }


BASELINE = gate.make_baseline(summary())


def failures(findings: Sequence[gate.Finding]) -> list[str]:
    return [finding.message for finding in findings if not finding.ok]


# ----------------------------------------------------------------------------------------------
# Thresholds


def test_a_baseline_allows_each_metric_its_tolerance_and_no_more() -> None:
    assert BASELINE == {
        "retrieval": {
            "hybrid": {
                "recall_at_5": {"baseline": 0.9035, "min": 0.8835},
                "hit_at_5": {"baseline": 0.9737, "min": 0.9537},
            }
        },
        "variants": {
            "hybrid-haiku": {
                "root_cause_accuracy": {"baseline": 0.90, "min": 0.85},
                "abstention_without_evidence": {"baseline": 0.75, "min": 0.65},
                "citation_faithfulness": {"baseline": 0.96, "min": 0.91},
                "cost_usd_per_incident": {"baseline": 0.0008, "max": 0.001},
            }
        },
    }


def test_a_metric_that_has_no_value_gets_no_threshold() -> None:
    baseline = gate.make_baseline(summary(citation_faithfulness=None))

    assert "citation_faithfulness" not in baseline["variants"]["hybrid-haiku"]


def test_results_at_their_own_baseline_pass() -> None:
    assert failures(gate.variant_findings(summary(), BASELINE)) == []
    assert failures(gate.retrieval_findings(summary()["retrieval"], BASELINE)) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"root_cause_accuracy": 0.84},
            "hybrid-haiku root_cause_accuracy: 0.84 (baseline 0.9, minimum 0.85)",
        ),
        (
            {"abstention_without_evidence": 0.5833},
            "hybrid-haiku abstention_without_evidence: 0.5833 (baseline 0.75, minimum 0.65)",
        ),
        (
            {"citation_faithfulness": 0.9},
            "hybrid-haiku citation_faithfulness: 0.9 (baseline 0.96, minimum 0.91)",
        ),
        (
            {"cost_usd_per_incident": 0.0011},
            "hybrid-haiku cost_usd_per_incident: 0.0011 (baseline 0.0008, maximum 0.001)",
        ),
        (
            {"citation_faithfulness": None},
            "hybrid-haiku citation_faithfulness: no value, baseline 0.96",
        ),
        (
            {"unusable_answers": {"refused": 2, "truncated": 0, "invalid_output": 1}},
            "hybrid-haiku: 3 answers were refused, cut off or invalid",
        ),
        ({"answers_judged": 97}, "hybrid-haiku: 97 of 100 cited answers were judged"),
    ],
)
def test_each_way_of_being_worse_than_the_baseline_fails(
    change: dict[str, Any], message: str
) -> None:
    assert failures(gate.variant_findings(summary(**change), BASELINE)) == [message]


@pytest.mark.parametrize(
    "change",
    [
        {"root_cause_accuracy": 0.85},
        {"root_cause_accuracy": 1.0},
        {"cost_usd_per_incident": 0.001},
        {"cost_usd_per_incident": 0.0001},
    ],
)
def test_a_change_within_the_tolerance_or_for_the_better_passes(change: dict[str, Any]) -> None:
    assert failures(gate.variant_findings(summary(**change), BASELINE)) == []


def test_a_variant_that_vanished_from_the_results_fails() -> None:
    results = summary()
    results["variants"] = {}

    assert failures(gate.variant_findings(results, BASELINE)) == [
        "hybrid-haiku: in the baseline but not in the results"
    ]


def test_retrieval_below_its_minimum_fails_and_a_missing_mode_fails() -> None:
    worse = {"hybrid": {"recall_at_5": 0.8772, "hit_at_5": 0.9737}}

    assert failures(gate.retrieval_findings(worse, BASELINE)) == [
        "retrieval hybrid recall_at_5: 0.8772 (baseline 0.9035, minimum 0.8835)"
    ]
    assert len(failures(gate.retrieval_findings({}, BASELINE))) == 2


@pytest.mark.parametrize(("expected", "ok"), [(144, True), (137, True), (136, False), (0, False)])
def test_the_judge_must_agree_with_nearly_all_of_its_controls(expected: int, ok: bool) -> None:
    results = summary()
    results["judge"]["controls"]["as_expected"] = expected

    assert gate.control_finding(results).ok is ok


def test_a_judge_without_controls_fails() -> None:
    results = summary()
    results["judge"]["controls"] = {"controls": 0, "as_expected": 0}

    assert gate.control_finding(results).ok is False


# ----------------------------------------------------------------------------------------------
# The whole gate, on a repository state built with a stand-in model


@pytest.fixture(scope="module")
async def recorded(
    settings: Settings,
    loaded_database: str,
    labelled: list[LabelledIncident],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """An `eval` directory with recordings, results, a baseline and a README that agree."""
    eval_dir = tmp_path_factory.mktemp("recorded") / "eval"
    scratch = settings.model_copy(update={"eval_dir": eval_dir})
    catalogue = load_catalogue(settings.data_dir)
    evaluation = await runner.evaluate(
        scratch,
        live=True,
        paid_llm=RuleLLM(labelled, catalogue, oracle(catalogue)),
        budget=Budget(max_usd=1000.0),
        log=lambda _: None,
    )
    runner.write(evaluation, eval_dir / gate.RESULTS_DIR, {"paid_usd": 0.0})
    baseline = gate.make_baseline(evaluation.summary)
    (eval_dir / gate.BASELINE_FILE).write_text(json.dumps(baseline))
    table = report.results_table(evaluation.summary)
    (eval_dir / "README.md").write_text(f"{report.START}\n{table}\n{report.END}\n")
    return eval_dir


@pytest.fixture
def state(settings: Settings, recorded: Path, tmp_path: Path) -> Settings:
    """A copy of the recorded state that a test may break."""
    shutil.copytree(recorded, tmp_path / "eval")
    return settings.model_copy(update={"eval_dir": tmp_path / "eval"})


def committed_summary(state: Settings) -> dict[str, Any]:
    path = state.eval_dir / gate.RESULTS_DIR / runner.SUMMARY_FILE
    return dict(json.loads(path.read_text()))


async def run_gate(state: Settings) -> list[gate.Finding]:
    return await gate.run(state, state.eval_dir / "README.md")


async def test_a_repository_in_order_passes_every_check(state: Settings) -> None:
    findings = await run_gate(state)

    assert failures(findings) == []
    messages = [finding.message for finding in findings]
    assert "replay: 750 rows rebuilt from the recorded answers" in messages
    assert "results: summary.json is what the replay computes" in messages
    assert "readme: the table is that of the results" in messages
    assert "judge: the expected verdict on 144 of 144 controls" in messages
    # A run that pays nothing still writes run.json, which the gate does not compare.
    assert (state.eval_dir / gate.RESULTS_DIR / runner.RUN_FILE).exists()


async def test_a_search_that_got_worse_fails_on_retrieval_and_on_stale_recordings(
    state: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fuse = triage.retrieval.fuse
    # The hybrid search quietly stops using the vector ranking.
    monkeypatch.setattr(
        triage.retrieval, "fuse", lambda rankings, limit: real_fuse(rankings[1:], limit)
    )

    failed = failures(await run_gate(state))

    # Without the vector ranking the hybrid search is the full-text search.
    fulltext = committed_summary(state)["retrieval"]["fulltext"]["recall_at_5"]
    assert any(
        message.startswith(f"retrieval hybrid recall_at_5: {fulltext} ") for message in failed
    )
    assert any(message.startswith("replay: hybrid-haiku, incident T") for message in failed)
    assert not any("vector" in message or "fulltext" in message for message in failed)


async def test_a_changed_prompt_fails_as_stale_without_touching_retrieval(
    state: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prompts, "SYSTEM_WITH_EVIDENCE", prompts.SYSTEM_WITH_EVIDENCE + "\nAnswer briefly."
    )

    failed = failures(await run_gate(state))

    assert len(failed) == 1
    assert failed[0].startswith("replay: vector-haiku, incident T")
    assert "Run `triage eval run` with an API key" in failed[0]


async def test_a_number_edited_in_the_results_fails(state: Settings) -> None:
    path = state.eval_dir / gate.RESULTS_DIR / runner.SUMMARY_FILE
    edited = json.loads(path.read_text())
    edited["variants"]["hybrid-haiku"]["root_cause_accuracy"] = 0.5
    path.write_text(json.dumps(edited, indent=2) + "\n")

    failed = failures(await run_gate(state))

    assert any(
        message.startswith("results: summary.json is not what the replay computes: line")
        and "root_cause_accuracy" in message
        for message in failed
    )
    # The README still shows the number that the replay computes, not the edited one.
    assert any(message.startswith("readme: the table is not that of") for message in failed)


async def test_an_answer_edited_in_the_rows_fails(state: Settings) -> None:
    path = state.eval_dir / gate.RESULTS_DIR / runner.ROWS_FILE
    path.write_text(
        path.read_text().replace('"decision":"insufficient_evidence"', '"decision":"triage"', 1)
    )

    failed = failures(await run_gate(state))

    assert [message.split(":")[1].strip() for message in failed] == [
        "rows.jsonl is not what the replay computes"
    ]


async def test_results_below_the_baseline_fail_until_a_new_baseline_is_accepted(
    state: Settings,
) -> None:
    path = state.eval_dir / gate.BASELINE_FILE
    baseline = json.loads(path.read_text())
    measured = committed_summary(state)["variants"]["vector-haiku"]["root_cause_accuracy"]
    assert measured < 0.99
    accuracy = baseline["variants"]["vector-haiku"]["root_cause_accuracy"]
    accuracy["baseline"], accuracy["min"] = 1.0, 0.99
    path.write_text(json.dumps(baseline))

    assert failures(await run_gate(state)) == [
        f"vector-haiku root_cause_accuracy: {measured} (baseline 1.0, minimum 0.99)"
    ]

    path.write_text(json.dumps(gate.make_baseline(committed_summary(state))))
    assert failures(await run_gate(state)) == []


async def test_a_readme_table_that_is_not_the_rendered_one_fails(state: Settings) -> None:
    readme = state.eval_dir / "README.md"
    readme.write_text(readme.read_text().replace("100%", "99%", 1))

    assert failures(await run_gate(state)) == [
        "readme: the table is not that of eval/results/summary.json; run `triage eval report`"
    ]


async def test_a_recording_that_was_deleted_fails_as_stale(state: Settings) -> None:
    path = state.eval_dir / runner.RECORDINGS_DIR / "hybrid-sonnet.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[1:]))

    failed = failures(await run_gate(state))

    assert len(failed) == 1
    assert failed[0].startswith("replay: hybrid-sonnet, incident T")


async def test_results_with_an_attempt_that_got_no_response_fail(state: Settings) -> None:
    errors = state.eval_dir / gate.RESULTS_DIR / runner.ERRORS_FILE
    errors.write_text('{"variant": "hybrid-haiku", "case_id": "T01", "rep": 0}\n')

    assert failures(await run_gate(state)) == [
        "results: errors.jsonl lists attempts that got no response"
    ]
