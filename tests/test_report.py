from typing import Any

import pytest

from triage.evaluation.report import (
    END,
    START,
    presentation_table,
    problems,
    results_table,
    with_tables,
)


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

    assert problems(readme, SUMMARY, require="results") == []


def test_a_number_typed_into_the_readme_is_a_problem() -> None:
    readme = f"{START}\n{TABLE.replace('92%', '95%')}\n{END}\n"

    assert problems(readme, SUMMARY) == [
        "the results table is not that of eval/results/summary.json; "
        "run `triage eval report --update`"
    ]


@pytest.mark.parametrize(
    "readme", ["no markers at all", f"{START}\nonly a start", f"{START}{START}{END}"]
)
def test_a_readme_without_exactly_one_pair_of_markers_is_a_problem(readme: str) -> None:
    assert problems(readme, SUMMARY, require="results") == [
        f"expected exactly one {START} and one {END}"
    ]


def test_a_document_without_markers_has_no_table_to_be_wrong() -> None:
    assert problems("prose only", SUMMARY) == []


def test_updating_replaces_what_is_between_the_markers_and_nothing_else() -> None:
    readme = f"Before.\n\n{START}\nan old table\n{END}\n\nAfter.\n"

    updated = with_tables(readme, SUMMARY)

    assert updated == f"Before.\n\n{START}\n{TABLE}\n{END}\n\nAfter.\n"
    assert problems(updated, SUMMARY, require="results") == []
    assert with_tables(updated, SUMMARY) == updated


def test_each_table_a_document_has_markers_for_is_checked_and_updated() -> None:
    summary = {
        "variants": {
            name: {
                **values,
                "root_cause_accuracy_by_presentation": {"signature": 1.0, "paraphrase": 0.5897},
            }
            for name, values in SUMMARY["variants"].items()
        }
    }
    marked = "<!-- presentation:start -->\nold\n<!-- presentation:end -->"
    document = f"{START}\n{TABLE}\n{END}\n\n{marked}\n"

    assert problems(document, summary) == [
        "the presentation table is not that of eval/results/summary.json; "
        "run `triage eval report --update`"
    ]
    updated = with_tables(document, summary)
    assert problems(updated, summary) == []
    assert presentation_table(summary) in updated
    assert "| Hybrid, Haiku 5.5 | 100% | 59% | n/a |" in updated
