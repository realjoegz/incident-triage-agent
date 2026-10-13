"""The results table, rendered from the summary file.

The README does not hold numbers that someone typed. Its table sits between two markers, and
the gate fails when the text between them is not what this module renders from the committed
results.
"""

from collections.abc import Mapping
from typing import Any

START = "<!-- results:start -->"
END = "<!-- results:end -->"

_NONE = "n/a"


def _percent(value: float | None) -> str:
    return _NONE if value is None else f"{value * 100:.0f}%"


def _ratio(value: float | None) -> str:
    return _NONE if value is None else f"{value:.2f}"


def _seconds(value: float | None) -> str:
    return _NONE if value is None else f"{value:.1f} s"


def _range(interval: list[float] | None) -> str:
    if interval is None:
        return ""
    return f" ({interval[0] * 100:.0f} to {interval[1] * 100:.0f})"


def results_table(summary: Mapping[str, Any]) -> str:
    """The table of the README: one row per variant."""
    lines = [
        "| Variant | Recall@5 | Root-cause accuracy | Abstains without evidence "
        "| Citation faithfulness | Cost per incident | Latency p50 / p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for values in summary["variants"].values():
        lines.append(
            "| "
            + " | ".join(
                [
                    values["label"],
                    _ratio(values["retrieval_recall_at_5"]),
                    _percent(values["root_cause_accuracy"]),
                    _percent(values["abstention_without_evidence"]),
                    _percent(values["citation_faithfulness"]),
                    f"${values['cost_usd_per_incident']:.4f}",
                    f"{_seconds(values['latency_p50_s'])} / {_seconds(values['latency_p95_s'])}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def detail_table(summary: Mapping[str, Any]) -> str:
    """The longer table of docs/evaluation.md: intervals and what went wrong, per variant."""
    lines = [
        "| Variant | Root-cause accuracy (95% interval) | Wrong cause | Abstained despite "
        "evidence | Abstains without evidence (95% interval) | Citation faithfulness "
        "| Cites a document of the fault | First-draft citations valid | Repaired |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for values in summary["variants"].values():
        lines.append(
            "| "
            + " | ".join(
                [
                    values["label"],
                    _percent(values["root_cause_accuracy"])
                    + _range(values["root_cause_accuracy_interval"]),
                    _percent(values["wrong_cause_rate"]),
                    _percent(values["abstained_despite_evidence_rate"]),
                    _percent(values["abstention_without_evidence"])
                    + _range(values["abstention_without_evidence_interval"]),
                    _percent(values["citation_faithfulness"]),
                    _percent(values["cites_a_document_of_the_fault"]),
                    _percent(values["draft_citation_validity"]),
                    _percent(values["repair_rate"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _between_markers(text: str) -> str | None:
    if text.count(START) != 1 or text.count(END) != 1:
        return None
    return text.split(START)[1].split(END)[0].strip()


def readme_problem(text: str, summary: Mapping[str, Any]) -> str | None:
    found = _between_markers(text)
    if found is None:
        return f"expected exactly one {START} and one {END}"
    if found != results_table(summary):
        return "the table is not that of eval/results/summary.json; run `triage eval report`"
    return None


def with_table(text: str, summary: Mapping[str, Any]) -> str:
    """The text with what is between the markers replaced by the current table."""
    if _between_markers(text) is None:
        raise ValueError(f"expected exactly one {START} and one {END}")
    before, rest = text.split(START)
    _, after = rest.split(END)
    return f"{before}{START}\n{results_table(summary)}\n{END}{after}"
