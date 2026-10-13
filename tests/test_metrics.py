from typing import Any

import pytest

from triage.evaluation.graders import grade
from triage.evaluation.metrics import (
    Graded,
    interval,
    paired,
    percentile,
    summarise,
)
from triage.evaluation.rows import CallSummary, Row, Verdict
from triage.evaluation.variants import BY_NAME
from triage.llm.base import Usage
from triage.models import Citation, Label
from triage.taxonomy import Category

GOLD = ["D1", "D2", "D3"]
HYBRID = BY_NAME["hybrid-haiku"]
NONE = BY_NAME["none-haiku"]


def label(case_id: str, *, answerable: bool = True) -> Label:
    return Label(
        id=case_id,
        # Each incident of these tests is of its own fault unless a test says otherwise.
        fault_id=f"fault-of-{case_id}",
        answerable=answerable,
        category=Category.RATE_LIMITING if answerable else Category.UNKNOWN,
        doc_ids=GOLD if answerable else [],
        presentation="signature",
        service="payments-gateway",
    )


def row(case_id: str, rep: int = 0, **changes: Any) -> Row:
    """A row that names the labelled cause and cites a document of the fault."""
    call = CallSummary(
        step="draft",
        fingerprint="f",
        model="claude-haiku-5-5",
        stop_reason="end_turn",
        usage=Usage(input_tokens=3000, output_tokens=400),
        latency_s=changes.pop("latency", 2.0),
        attempts=1,
    )
    values: dict[str, Any] = {
        "variant": "hybrid-haiku",
        "case_id": case_id,
        "rep": rep,
        "retrieved": ["D1", "X1", "D2", "X2", "X3"],
        "outcome": "answered",
        "decision": "triage",
        "category": Category.RATE_LIMITING,
        "likely_cause": "The acquirer throttled the account.",
        "next_step": "Compare the traffic with the limit.",
        "citations": [Citation(doc_id="D1", quote="a quote of twenty characters")],
        "abstained_because": None,
        "draft_citations": 1,
        "draft_citations_valid": 1,
        "repaired": False,
        "calls": [call],
        "cost_usd": 0.001,
        "latency_s": call.latency_s,
    }
    return Row(**{**values, **changes})


def abstaining(case_id: str, rep: int = 0, **changes: Any) -> Row:
    return row(
        case_id,
        rep,
        decision="insufficient_evidence",
        category=Category.UNKNOWN,
        likely_cause="",
        citations=[],
        draft_citations=0,
        draft_citations_valid=0,
        **changes,
    )


# ----------------------------------------------------------------------------------------------
# One row against its label


def test_with_evidence_the_labelled_category_is_the_only_right_answer() -> None:
    right = grade(row("A"), label("A"), retrieves=True)
    other = grade(row("A", category=Category.DNS_FAILURE), label("A"), retrieves=True)
    silent = grade(abstaining("A"), label("A"), retrieves=True)

    assert (right.correct, right.wrong_cause, right.abstained) == (True, False, False)
    assert (other.correct, other.wrong_cause, other.abstained) == (False, True, False)
    assert (silent.correct, silent.wrong_cause, silent.abstained) == (False, False, True)
    assert [item.false_attribution for item in (right, other, silent)] == [False] * 3


def test_without_evidence_abstaining_is_the_only_right_answer() -> None:
    silent = grade(abstaining("N"), label("N", answerable=False), retrieves=True)
    named = grade(row("N"), label("N", answerable=False), retrieves=True)

    assert (silent.correct, silent.false_attribution) == (True, False)
    assert (named.correct, named.false_attribution, named.wrong_cause) == (False, True, False)
    assert (named.recall, named.hit, named.cites_gold) == (None, None, None)


def test_recall_is_the_share_of_the_faults_documents_among_those_retrieved() -> None:
    found_two = grade(row("A"), label("A"), retrieves=True)
    found_none = grade(row("A", retrieved=["X1", "X2"]), label("A"), retrieves=True)
    late = grade(row("A", retrieved=["X1", "X2", "D3"]), label("A"), retrieves=True)

    assert (found_two.recall, found_two.hit, found_two.reciprocal_rank) == (2 / 3, True, 1.0)
    assert (found_none.recall, found_none.hit, found_none.reciprocal_rank) == (0.0, False, 0.0)
    assert (late.recall, late.reciprocal_rank) == (1 / 3, 1 / 3)


def test_a_variant_without_retrieval_has_no_retrieval_grade() -> None:
    result = grade(row("A", retrieved=[], citations=[]), label("A"), retrieves=False)

    assert (result.recall, result.hit, result.reciprocal_rank, result.cites_gold) == (
        None,
        None,
        None,
        None,
    )
    assert result.correct


def test_citing_a_document_of_the_fault_is_graded_apart_from_naming_the_category() -> None:
    elsewhere = [Citation(doc_id="X1", quote="a quote of twenty characters")]

    cited_right = grade(row("A"), label("A"), retrieves=True)
    cited_wrong = grade(row("A", citations=elsewhere), label("A"), retrieves=True)
    abstained = grade(abstaining("A"), label("A"), retrieves=True)

    assert (cited_right.cites_gold, cited_wrong.cites_gold, abstained.cites_gold) == (
        True,
        False,
        None,
    )
    # The category is right either way. Only one of the two got there by the fault's document.
    assert (cited_right.correct, cited_wrong.correct) == (True, True)
    assert (cited_right.correct_citing_the_fault, cited_wrong.correct_citing_the_fault) == (
        True,
        False,
    )
    assert abstained.correct_citing_the_fault is False


@pytest.mark.parametrize("outcome", ["refused", "truncated", "invalid_output"])
def test_an_answer_the_model_never_gave_is_not_scored(outcome: str) -> None:
    assert grade(abstaining("A", outcome=outcome), label("A"), retrieves=True).scored is False
    assert grade(row("A"), label("A"), retrieves=True).scored is True


# ----------------------------------------------------------------------------------------------
# Many rows


def summary(rows: list[Row], *labels: Label) -> dict[str, Any]:
    return Graded(HYBRID, rows, {item.id: item for item in labels}).summary()


def test_an_incident_counts_once_however_often_it_was_run() -> None:
    rows = [
        row("A", 0),
        row("A", 1, category=Category.DNS_FAILURE),
        row("A", 2, category=Category.DNS_FAILURE),
        row("B", 0),
    ]

    result = summary(rows, label("A"), label("B"))

    # A is right once in three and B once in one: (1/3 + 1) / 2, not 2 of 4 rows.
    assert result["root_cause_accuracy"] == 0.6667
    assert result["wrong_cause_rate"] == 0.3333
    assert result["root_cause_accuracy_by_repetition"] == [1.0, 0.0, 0.0]
    assert (result["rows"], result["repetitions"], result["incidents_with_evidence"]) == (4, 3, 2)


def test_an_unusable_answer_is_counted_and_left_out_of_every_rate() -> None:
    rows = [
        abstaining("A", 0, outcome="refused"),
        row("A", 1),
        abstaining("B", 0, outcome="truncated"),
    ]

    result = summary(rows, label("A"), label("B"))

    assert result["unusable_answers"] == {"refused": 1, "truncated": 1, "invalid_output": 0}
    # A is judged on its one usable answer. B has none and is in no rate at all.
    assert result["root_cause_accuracy"] == 1.0
    assert result["abstained_despite_evidence_rate"] == 0.0
    assert result["incidents_with_evidence"] == 1


def test_both_kinds_of_incident_have_their_own_rate() -> None:
    rows = [row("A"), abstaining("B"), abstaining("N"), row("M")]

    result = summary(
        rows, label("A"), label("B"), label("N", answerable=False), label("M", answerable=False)
    )

    assert result["root_cause_accuracy"] == 0.5
    assert result["abstained_despite_evidence_rate"] == 0.5
    assert result["abstention_without_evidence"] == 0.5
    assert (result["incidents_with_evidence"], result["incidents_without_evidence"]) == (2, 2)


def test_retrieval_rates_are_over_the_incidents_that_have_documents() -> None:
    rows = [row("A"), row("B", retrieved=["X1"]), abstaining("N", retrieved=["X1"])]

    result = summary(rows, label("A"), label("B"), label("N", answerable=False))

    assert result["retrieval_recall_at_5"] == 0.3333
    assert result["retrieval_hit_at_5"] == 0.5
    assert result["retrieval_mrr"] == 0.5


def test_faithfulness_is_over_the_answers_that_were_judged() -> None:
    yes = Verdict(reason="stated", supported=True)
    no = Verdict(reason="absent", supported=False)
    rows = [
        row("A", 0, judge=yes),
        row("A", 1, judge=no),
        row("B", 0, judge=yes),
        row("C"),
        abstaining("D"),
    ]

    result = summary(rows, label("A"), label("B"), label("C"), label("D"))

    assert result["citation_faithfulness"] == 0.75
    assert (result["answers_judged"], result["answers_to_judge"]) == (3, 4)


def test_first_draft_validity_and_the_repair_rate_are_over_rows() -> None:
    rows = [
        row("A", draft_citations=2, draft_citations_valid=1, repaired=True),
        row("B", draft_citations=2, draft_citations_valid=2),
        abstaining("C", calls=[]),
    ]

    result = summary(rows, label("A"), label("B"), label("C"))

    assert result["draft_citations"] == 4
    assert result["draft_citation_validity"] == 0.75
    # C made no model call, so it could not have been repaired.
    assert result["repair_rate"] == 0.5


def test_cost_is_per_incident_over_all_rows_and_latency_over_those_that_called_the_model() -> None:
    rows = [
        row("A", cost_usd=0.002, latency=1.0),
        row("B", cost_usd=0.004, latency=3.0),
        abstaining("C", calls=[], cost_usd=0.0, latency_s=0.0),
    ]

    result = summary(rows, label("A"), label("B"), label("C"))

    assert result["cost_usd_per_incident"] == 0.002
    assert (result["latency_p50_s"], result["latency_p95_s"]) == (1.0, 3.0)
    assert result["input_tokens_per_incident"] == 2000
    assert result["output_tokens_per_incident"] == pytest.approx(266.6667)


def test_a_rate_with_nothing_to_measure_is_absent_and_not_zero() -> None:
    result = Graded(
        NONE, [row("A", variant="none-haiku", retrieved=[], citations=[])], {"A": label("A")}
    ).summary()

    assert result["retrieval_recall_at_5"] is None
    assert result["citation_faithfulness"] is None
    assert result["abstention_without_evidence"] is None
    assert result["draft_citation_validity"] == 1.0


def test_variants_are_compared_with_the_default_over_the_same_incidents() -> None:
    labels = {name: label(name) for name in "ABCD"}
    rows = [row(name) for name in "ABCD"]
    rows += [
        row(name, variant="none-haiku", category=category, retrieved=[], citations=[])
        for name, category in zip(
            "ABCD", [Category.RATE_LIMITING] + [Category.DNS_FAILURE] * 3, strict=True
        )
    ]

    result = summarise(rows, labels, [NONE, HYBRID])

    assert list(result["variants"]) == ["none-haiku", "hybrid-haiku"]  # type: ignore[call-overload]
    comparison = result["default_minus"]["none-haiku"]["root_cause_accuracy"]  # type: ignore[index]
    assert comparison["difference"] == 0.75
    assert comparison["incidents"] == 4
    assert comparison["interval"][0] <= 0.75 <= comparison["interval"][1]


# ----------------------------------------------------------------------------------------------
# Statistics


def test_a_percentile_is_the_nearest_rank() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.50) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0
    assert percentile([7.0], 0.50) == 7.0
    assert percentile([], 0.50) is None


def test_an_interval_is_the_same_every_time_and_brackets_the_mean() -> None:
    per_case = {f"T{n:02d}": float(n % 3 == 0) for n in range(30)}

    first = interval(per_case)

    assert first == interval(dict(reversed(per_case.items())))
    assert first is not None
    assert first[0] < 10 / 30 < first[1]
    assert 0.0 <= first[0] < first[1] <= 1.0


def test_an_interval_of_identical_values_has_no_width_and_one_value_has_no_interval() -> None:
    assert interval({"A": 1.0, "B": 1.0, "C": 1.0}) == [1.0, 1.0]
    assert interval({"A": 1.0}) is None
    assert interval({}) is None


def test_a_paired_difference_uses_only_the_incidents_both_sides_have() -> None:
    result = paired({"A": 1.0, "B": 1.0, "C": 0.0}, {"B": 0.0, "C": 0.0, "D": 1.0})

    assert result is not None
    assert (result["difference"], result["incidents"]) == (0.5, 2)
    assert paired({"A": 1.0}, {"A": 0.0}) is None


def test_the_stricter_accuracy_drops_an_answer_that_cites_another_faults_document() -> None:
    elsewhere = [Citation(doc_id="X1", quote="a quote of twenty characters")]
    rows = [row("A"), row("B", citations=elsewhere)]

    result = summary(rows, label("A"), label("B"))

    assert result["root_cause_accuracy"] == 1.0
    assert result["root_cause_accuracy_citing_the_fault"] == 0.5
    assert result["cites_a_document_of_the_fault"] == 0.5


def test_an_interval_resamples_whole_faults_when_it_is_told_which_incident_is_whose() -> None:
    # Six incidents from three faults, and each fault is all right or all wrong.
    per_case = {"A1": 1.0, "A2": 1.0, "B1": 1.0, "B2": 1.0, "C1": 0.0, "C2": 0.0}
    fault_of = {case: case[0] for case in per_case}

    by_incident = interval(per_case)
    by_fault = interval(per_case, fault_of)

    assert by_incident is not None
    assert by_fault is not None
    # Three independent faults say less than six independent incidents would.
    assert by_fault[1] - by_fault[0] > by_incident[1] - by_incident[0]
    assert by_fault == [0.0, 1.0]
    assert interval({"A1": 1.0, "A2": 0.0}, {"A1": "A", "A2": "A"}) is None


def test_accuracy_is_also_given_for_each_way_an_incident_is_shown() -> None:
    told = label("B").model_copy(update={"presentation": "paraphrase"})
    rows = [row("A"), row("B", 0), abstaining("B", 1)]

    result = summary(rows, label("A"), told)

    assert result["root_cause_accuracy_by_presentation"] == {"paraphrase": 0.5, "signature": 1.0}
