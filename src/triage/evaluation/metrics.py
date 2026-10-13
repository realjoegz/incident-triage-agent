"""From rows to the numbers of the results table.

Every rate is first averaged within an incident, over its repetitions, and then across
incidents, so that an incident counts once however often it was run.

Confidence intervals come from resampling faults, not rows and not incidents. Three answers to
one incident are not three independent observations, and neither are two incidents that were
rendered from the same fault: they share its log lines and its documents.
"""

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from triage.evaluation.graders import Grade, grade
from triage.evaluation.rows import Row
from triage.evaluation.variants import DEFAULT, Variant
from triage.models import Label

DECIMALS = 4
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261013


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, DECIMALS)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest rank: the smallest value with at least q of the values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _resamples(count: int) -> list[list[int]]:
    """The same index draws on every machine: Python guarantees `Random.random` for a seed."""
    generator = random.Random(BOOTSTRAP_SEED)
    return [
        [int(generator.random() * count) for _ in range(count)] for _ in range(BOOTSTRAP_SAMPLES)
    ]


def interval(
    per_case: Mapping[str, float], cluster_of: Mapping[str, str] | None = None
) -> list[float] | None:
    """A 95% percentile bootstrap interval for the mean over incidents.

    `cluster_of` names the fault of each incident. A draw takes whole faults, with all of their
    incidents. Without it every incident is its own cluster.
    """
    groups: dict[str, list[float]] = {}
    for case in sorted(per_case):
        groups.setdefault(cluster_of[case] if cluster_of else case, []).append(per_case[case])
    clusters = [groups[name] for name in sorted(groups)]
    if len(clusters) < 2:
        return None
    means = []
    for draw in _resamples(len(clusters)):
        values = [value for index in draw for value in clusters[index]]
        means.append(sum(values) / len(values))
    means.sort()
    low = means[int(0.025 * len(means))]
    high = means[int(0.975 * len(means)) - 1]
    return [round(low, DECIMALS), round(high, DECIMALS)]


def paired(
    ours: Mapping[str, float],
    theirs: Mapping[str, float],
    cluster_of: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    """The mean of `ours - theirs` over the incidents both have, with its interval."""
    shared = sorted(set(ours) & set(theirs))
    if len(shared) < 2:
        return None
    differences = {case: ours[case] - theirs[case] for case in shared}
    return {
        "difference": _round(mean(list(differences.values()))),
        "interval": interval(differences, cluster_of),
        "incidents": len(shared),
    }


def _per_case(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case, value in pairs:
        grouped[case].append(value)
    return {case: sum(values) / len(values) for case, values in grouped.items()}


def _rate(per_case: Mapping[str, float]) -> float | None:
    return _round(mean(list(per_case.values())))


class Graded:
    """The rows of one variant with their grades."""

    def __init__(self, variant: Variant, rows: Sequence[Row], labels: Mapping[str, Label]) -> None:
        self.variant = variant
        self.fault_of = {case: label.fault_id for case, label in labels.items()}
        self.rows = sorted(rows, key=lambda row: (row.case_id, row.rep))
        self.grades: list[tuple[Row, Grade]] = [
            (row, grade(row, labels[row.case_id], retrieves=variant.mode != "none"))
            for row in self.rows
        ]

    def per_case(self, value: str, *, answerable: bool) -> dict[str, float]:
        """An incident's mean of a boolean grade, over the repetitions that produced an answer."""
        return _per_case(
            [
                (row.case_id, float(getattr(result, value)))
                for row, result in self.grades
                if result.scored and result.answerable is answerable
            ]
        )

    def summary(self) -> dict[str, object]:
        accuracy = self.per_case("correct", answerable=True)
        abstention = self.per_case("correct", answerable=False)
        recall = _per_case(
            [(row.case_id, g.recall) for row, g in self.grades if g.recall is not None]
        )
        hit = _per_case(
            [(row.case_id, float(g.hit)) for row, g in self.grades if g.hit is not None]
        )
        reciprocal = _per_case(
            [
                (row.case_id, g.reciprocal_rank)
                for row, g in self.grades
                if g.reciprocal_rank is not None
            ]
        )
        judged = [(row.case_id, float(row.judge.supported)) for row in self.rows if row.judge]
        faithfulness = _per_case(judged)
        cites_gold = _per_case(
            [
                (row.case_id, float(g.cites_gold))
                for row, g in self.grades
                if g.scored and g.cites_gold is not None
            ]
        )
        citing_the_fault = _per_case(
            [
                (row.case_id, float(g.correct_citing_the_fault))
                for row, g in self.grades
                if g.scored and g.correct_citing_the_fault is not None
            ]
        )
        with_calls = [row for row in self.rows if row.calls]
        drafted = sum(row.draft_citations for row in self.rows)
        reps = sorted({row.rep for row in self.rows})
        by_rep = []
        for rep in reps:
            of_rep = [
                float(g.correct)
                for row, g in self.grades
                if row.rep == rep and g.scored and g.answerable
            ]
            by_rep.append(_round(mean(of_rep)))
        unusable = {
            outcome: sum(1 for row in self.rows if row.outcome == outcome)
            for outcome in ("refused", "truncated", "invalid_output")
        }
        return {
            "label": self.variant.label,
            "mode": self.variant.mode,
            "model": self.variant.model,
            "rows": len(self.rows),
            "repetitions": len(reps),
            "incidents_with_evidence": len(accuracy),
            "incidents_without_evidence": len(abstention),
            "unusable_answers": unusable,
            "retrieval_recall_at_5": _rate(recall),
            "retrieval_hit_at_5": _rate(hit),
            "retrieval_mrr": _rate(reciprocal),
            "root_cause_accuracy": _rate(accuracy),
            "root_cause_accuracy_interval": interval(accuracy, self.fault_of),
            "root_cause_accuracy_by_repetition": by_rep,
            "root_cause_accuracy_citing_the_fault": _rate(citing_the_fault),
            "wrong_cause_rate": _rate(self.per_case("wrong_cause", answerable=True)),
            "abstained_despite_evidence_rate": _rate(self.per_case("abstained", answerable=True)),
            "abstention_without_evidence": _rate(abstention),
            "abstention_without_evidence_interval": interval(abstention, self.fault_of),
            "citation_faithfulness": _rate(faithfulness),
            "citation_faithfulness_interval": interval(faithfulness, self.fault_of),
            "answers_judged": len(judged),
            "answers_to_judge": sum(1 for row in self.rows if _makes_a_cited_claim(row)),
            "cites_a_document_of_the_fault": _rate(cites_gold),
            "draft_citations": drafted,
            "draft_citation_validity": _round(
                sum(row.draft_citations_valid for row in self.rows) / drafted if drafted else None
            ),
            "repair_rate": _round(
                sum(1 for row in with_calls if row.repaired) / len(with_calls)
                if with_calls
                else None
            ),
            "cost_usd_per_incident": round(mean([row.cost_usd for row in self.rows]) or 0.0, 6),
            "judge_cost_usd_total": round(sum(row.judge_cost_usd for row in self.rows), 6),
            "latency_p50_s": percentile([row.latency_s for row in with_calls], 0.50),
            "latency_p95_s": percentile([row.latency_s for row in with_calls], 0.95),
            "input_tokens_per_incident": _round(
                mean([sum(call.usage.input_tokens for call in row.calls) for row in self.rows])
            ),
            "output_tokens_per_incident": _round(
                mean([sum(call.usage.output_tokens for call in row.calls) for row in self.rows])
            ),
        }


def _makes_a_cited_claim(row: Row) -> bool:
    return row.outcome == "answered" and row.decision == "triage" and bool(row.citations)


def summarise(
    rows: Sequence[Row], labels: Mapping[str, Label], variants: Sequence[Variant]
) -> dict[str, object]:
    graded = {
        variant.name: Graded(variant, [row for row in rows if row.variant == variant.name], labels)
        for variant in variants
        if any(row.variant == variant.name for row in rows)
    }
    comparisons: dict[str, object] = {}
    reference = graded.get(DEFAULT.name)
    if reference is not None:
        for name, other in graded.items():
            if name == DEFAULT.name:
                continue
            comparisons[name] = {
                "root_cause_accuracy": paired(
                    reference.per_case("correct", answerable=True),
                    other.per_case("correct", answerable=True),
                    reference.fault_of,
                ),
                "abstention_without_evidence": paired(
                    reference.per_case("correct", answerable=False),
                    other.per_case("correct", answerable=False),
                    reference.fault_of,
                ),
            }
    return {
        "variants": {name: item.summary() for name, item in graded.items()},
        # Each entry is the default variant minus the named one, over the same incidents.
        "default_variant": DEFAULT.name,
        "default_minus": comparisons,
    }
