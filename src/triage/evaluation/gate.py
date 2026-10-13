"""The gate that CI runs: does the repository still measure at least what it claims?

It needs no API key. Four things are checked, in this order:

1. Retrieval is measured again and compared with the baseline. A change to the search shows up
   here at once.
2. The evaluation is replayed from the recordings. If the current code builds a prompt that has
   no recorded answer, the committed results are stale and the paid evaluation must be run
   again; a change to a prompt, to the agent or to what the search returns ends here.
3. What the replay computes must be the committed result files, byte for byte. This is what
   catches an edited number, and a change to the grading that moves one.
4. The metrics of the committed results are compared with the baseline. A new paid run that is
   worse than the baseline by more than the tolerance fails, until someone accepts it by
   committing a new baseline, where a reviewer can see it.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client

from triage import db
from triage.dataset import checks, load_corpus, load_labelled
from triage.embeddings import SNAPSHOT_FILE, Snapshot, SnapshotEmbedder
from triage.evaluation import report, retrieval_eval, runner
from triage.mcp_server import EvidenceTools, build_server
from triage.runtime import open_retriever
from triage.settings import Settings

RESULTS_DIR = "results"
BASELINE_FILE = "baseline.json"

# How far below its baseline a metric may fall before the gate fails. Retrieval is exact, so
# its tolerance only has to absorb one incident moving one document. The others are measured on
# model answers that vary from run to run: 0.05 is two of the 38 incidents with evidence, and
# 0.10 is a little more than one of the 12 without.
RETRIEVAL_TOLERANCE = 0.02
MINIMUMS = {
    "root_cause_accuracy": 0.05,
    "abstention_without_evidence": 0.10,
    "citation_faithfulness": 0.05,
}
# Cost per incident may rise by a quarter.
COST_RATIO = 1.25
RETRIEVAL_METRICS = ("recall_at_5", "hit_at_5")

# The judge must give the expected verdict on at least this share of its controls.
CONTROL_AGREEMENT = 0.95


@dataclass(frozen=True)
class Finding:
    ok: bool
    message: str

    def __str__(self) -> str:
        return f"{'ok  ' if self.ok else 'FAIL'}  {self.message}"


def make_baseline(summary: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = {
        mode: {
            metric: {
                "baseline": values[metric],
                "min": round(values[metric] - RETRIEVAL_TOLERANCE, 4),
            }
            for metric in RETRIEVAL_METRICS
        }
        for mode, values in summary["retrieval"].items()
    }
    variants: dict[str, Any] = {}
    for name, values in summary["variants"].items():
        thresholds: dict[str, Any] = {}
        for metric, tolerance in MINIMUMS.items():
            if values.get(metric) is not None:
                thresholds[metric] = {
                    "baseline": values[metric],
                    "min": round(max(0.0, values[metric] - tolerance), 4),
                }
        thresholds["cost_usd_per_incident"] = {
            "baseline": values["cost_usd_per_incident"],
            "max": round(values["cost_usd_per_incident"] * COST_RATIO, 6),
        }
        variants[name] = thresholds
    return {"retrieval": retrieval, "variants": variants}


def _against(name: str, value: float | None, threshold: Mapping[str, float]) -> Finding:
    if value is None:
        return Finding(False, f"{name}: no value, baseline {threshold['baseline']}")
    if "min" in threshold:
        ok = value >= threshold["min"]
        bound = f"minimum {threshold['min']}"
    else:
        ok = value <= threshold["max"]
        bound = f"maximum {threshold['max']}"
    return Finding(ok, f"{name}: {value} (baseline {threshold['baseline']}, {bound})")


def retrieval_findings(
    measured: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Any]
) -> list[Finding]:
    return [
        _against(f"retrieval {mode} {metric}", measured.get(mode, {}).get(metric), threshold)
        for mode, thresholds in baseline["retrieval"].items()
        for metric, threshold in thresholds.items()
    ]


def variant_findings(summary: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[Finding]:
    findings = []
    for name, thresholds in baseline["variants"].items():
        values = summary["variants"].get(name)
        if values is None:
            findings.append(Finding(False, f"{name}: in the baseline but not in the results"))
            continue
        findings += [
            _against(f"{name} {metric}", values.get(metric), threshold)
            for metric, threshold in thresholds.items()
        ]
        unusable = sum(values["unusable_answers"].values())
        findings.append(
            Finding(unusable == 0, f"{name}: {unusable} answers were refused, cut off or invalid")
        )
        judged, to_judge = values["answers_judged"], values["answers_to_judge"]
        findings.append(
            Finding(judged == to_judge, f"{name}: {judged} of {to_judge} cited answers were judged")
        )
    return findings


def control_finding(summary: Mapping[str, Any]) -> Finding:
    controls = summary["judge"]["controls"]
    total, expected = controls["controls"], controls["as_expected"]
    ok = total > 0 and expected / total >= CONTROL_AGREEMENT
    return Finding(ok, f"judge: the expected verdict on {expected} of {total} controls")


def _first_difference(committed: str, computed: str) -> str:
    old, new = committed.splitlines(), computed.splitlines()
    for number, (a, b) in enumerate(zip(old, new, strict=False), start=1):
        if a != b:
            return f"line {number}: committed `{a.strip()[:90]}`, computed `{b.strip()[:90]}`"
    return f"{len(old)} lines committed, {len(new)} computed"


async def measure_retrieval(settings: Settings) -> dict[str, dict[str, object]]:
    corpus = load_corpus(settings.data_dir)
    snapshot = Snapshot.load(settings.data_dir / SNAPSHOT_FILE)
    db.load_corpus(settings.database_url, corpus, SnapshotEmbedder(snapshot))
    async with (
        open_retriever(settings, snapshot_only=True) as retriever,
        Client(build_server(retriever)) as client,
    ):
        metrics, _ = await retrieval_eval.measure(
            EvidenceTools(client).search, load_labelled(settings.data_dir)
        )
    return metrics


async def run(settings: Settings, readme: Path | None) -> list[Finding]:
    results_dir = settings.eval_dir / RESULTS_DIR
    baseline = json.loads((settings.eval_dir / BASELINE_FILE).read_text(encoding="utf-8"))
    findings = []

    problems = checks.run_all(settings.data_dir)
    findings.append(Finding(not problems, "data: " + ("; ".join(problems) or "the files check")))
    if problems:
        return findings

    findings += retrieval_findings(await measure_retrieval(settings), baseline)

    try:
        evaluation = await runner.evaluate(settings, live=False, log=lambda _: None)
    except runner.StaleResults as error:
        findings.append(Finding(False, f"replay: {error}"))
        return findings
    findings.append(
        Finding(True, f"replay: {len(evaluation.rows)} rows rebuilt from the recorded answers")
    )

    for name, computed in runner.rendered(evaluation).items():
        committed = (results_dir / name).read_text(encoding="utf-8")
        same = committed == computed
        detail = "" if same else ": " + _first_difference(committed, computed)
        verb = "is" if same else "is not"
        findings.append(Finding(same, f"results: {name} {verb} what the replay computes{detail}"))

    # A paid run writes this file only when an attempt got no response at all.
    unanswered = (results_dir / runner.ERRORS_FILE).exists()
    findings.append(
        Finding(
            not unanswered, f"results: {runner.ERRORS_FILE} is absent, so every attempt answered"
        )
        if not unanswered
        else Finding(False, f"results: {runner.ERRORS_FILE} lists attempts that got no response")
    )

    summary = json.loads((results_dir / runner.SUMMARY_FILE).read_text(encoding="utf-8"))
    findings += variant_findings(summary, baseline)
    findings.append(control_finding(summary))

    if readme is not None:
        problem = report.readme_problem(readme.read_text(encoding="utf-8"), summary)
        findings.append(
            Finding(problem is None, f"readme: {problem or 'the table is that of the results'}")
        )
    return findings
