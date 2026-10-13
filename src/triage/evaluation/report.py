"""The tables of the README and of the docs, rendered from the summary file.

No document holds a table that someone typed. A table sits between two markers, such as
`<!-- results:start -->` and `<!-- results:end -->`, and the gate fails when the text between
them is not what this module renders from the committed results.
"""

from collections.abc import Callable, Mapping
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
        "| Variant | Root-cause accuracy (95% interval) | Right and citing the fault "
        "| Wrong cause | Abstained despite evidence | Abstains without evidence (95% interval) "
        "| First-draft citations valid | Repaired |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for values in summary["variants"].values():
        lines.append(
            "| "
            + " | ".join(
                [
                    values["label"],
                    _percent(values["root_cause_accuracy"])
                    + _range(values["root_cause_accuracy_interval"]),
                    _percent(values["root_cause_accuracy_citing_the_fault"]),
                    _percent(values["wrong_cause_rate"]),
                    _percent(values["abstained_despite_evidence_rate"]),
                    _percent(values["abstention_without_evidence"])
                    + _range(values["abstention_without_evidence_interval"]),
                    _percent(values["draft_citation_validity"]),
                    _percent(values["repair_rate"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def presentation_table(summary: Mapping[str, Any]) -> str:
    """Root-cause accuracy by the way an incident is shown."""
    kinds = ["signature", "paraphrase", "downstream"]
    lines = ["| Variant | " + " | ".join(f"`{kind}`" for kind in kinds) + " |", "|---|---|---|---|"]
    for values in summary["variants"].values():
        by_kind = values["root_cause_accuracy_by_presentation"]
        cells = [_percent(by_kind.get(kind)) for kind in kinds]
        lines.append(f"| {values['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


TABLES: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "results": results_table,
    "detail": detail_table,
    "presentation": presentation_table,
}


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:start -->", f"<!-- {name}:end -->"


def _between(text: str, name: str) -> str | None:
    """What is between the markers of a table, or None when they are not there exactly once."""
    start, end = _markers(name)
    if text.count(start) != 1 or text.count(end) != 1:
        return None
    return text.split(start)[1].split(end)[0].strip()


def problems(text: str, summary: Mapping[str, Any], require: str | None = None) -> list[str]:
    """What is wrong with the tables of a document. `require` names one that must be there."""
    found = []
    for name, render in TABLES.items():
        start, end = _markers(name)
        if start not in text and end not in text and name != require:
            continue
        between = _between(text, name)
        if between is None:
            found.append(f"expected exactly one {start} and one {end}")
        elif between != render(summary):
            found.append(
                f"the {name} table is not that of eval/results/summary.json; "
                "run `triage eval report --update`"
            )
    return found


def with_tables(text: str, summary: Mapping[str, Any]) -> str:
    """The text with every table that it has markers for replaced by the current one."""
    for name, render in TABLES.items():
        if _between(text, name) is None:
            continue
        start, end = _markers(name)
        before, rest = text.split(start)
        _, after = rest.split(end)
        text = f"{before}{start}\n{render(summary)}\n{end}{after}"
    return text
