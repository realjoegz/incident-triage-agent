from typing import Any

import pytest

from triage.evaluation.report import END, START, readme_problem, results_table, with_table


def variant(label: str, **changes: Any) -> dict[str, Any]:
    values = {
        "label": label,
        "retrieval_recall_at_5": 0.9035,
        "root_cause_accuracy": 0.9211,
        "abstention_without_evidence": 0.8333,
        "citation_faithfulness": 0.9655,
        "cost_usd_per_incident": 0.000612,
        "latency_p50_s": 2.413,
        "latency_p95_s": 6.98,
    }
    return {**values, **changes}


SUMMARY = {
    "variants": {
        "none-haiku": variant(
            "No retrieval, Haiku 5.5",
            retrieval_recall_at_5=None,
            citation_faithfulness=None,
            root_cause_accuracy=0.3158,
        ),
        "hybrid-haiku": variant("Hybrid, Haiku 5.5"),
    }
}

TABLE = """\
| Variant | Recall@5 | Root-cause accuracy | Abstains without evidence | Citation faithfulness \
| Cost per incident | Latency p50 / p95 |
|---|---|---|---|---|---|---|
| No retrieval, Haiku 5.5 | n/a | 32% | 83% | n/a | $0.0006 | 2.4 s / 7.0 s |
| Hybrid, Haiku 5.5 | 0.90 | 92% | 83% | 97% | $0.0006 | 2.4 s / 7.0 s |"""


def test_the_table_has_one_row_per_variant_in_the_order_of_the_summary() -> None:
    assert results_table(SUMMARY) == TABLE


def test_a_readme_whose_table_is_the_rendered_one_has_no_problem() -> None:
    readme = f"# Title\n\n{START}\n{TABLE}\n{END}\n\nMore text.\n"

    assert readme_problem(readme, SUMMARY) is None


def test_a_number_typed_into_the_readme_is_a_problem() -> None:
    readme = f"{START}\n{TABLE.replace('92%', '95%')}\n{END}\n"

    assert readme_problem(readme, SUMMARY) == (
        "the table is not that of eval/results/summary.json; run `triage eval report`"
    )


@pytest.mark.parametrize(
    "readme", ["no markers at all", f"{START}\nonly a start", f"{START}{START}{END}"]
)
def test_a_readme_without_exactly_one_pair_of_markers_is_a_problem(readme: str) -> None:
    assert readme_problem(readme, SUMMARY) == f"expected exactly one {START} and one {END}"


def test_updating_replaces_what_is_between_the_markers_and_nothing_else() -> None:
    readme = f"Before.\n\n{START}\nan old table\n{END}\n\nAfter.\n"

    updated = with_table(readme, SUMMARY)

    assert updated == f"Before.\n\n{START}\n{TABLE}\n{END}\n\nAfter.\n"
    assert readme_problem(updated, SUMMARY) is None
    assert with_table(updated, SUMMARY) == updated
