"""The `triage` command."""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import uvicorn

from triage import db
from triage.api import create_app
from triage.dataset import checks, load_corpus, load_labelled
from triage.dataset.catalogue import load_catalogue
from triage.dataset.generate import write_files
from triage.embeddings import SNAPSHOT_FILE, LocalEmbedder, Snapshot
from triage.evaluation import gate, judge, report, runner
from triage.evaluation.rows import Row
from triage.evaluation.variants import BY_NAME, VARIANTS
from triage.llm.recording import Budget, BudgetExceeded
from triage.mcp_server import serve_stdio
from triage.retrieval import incident_query
from triage.runtime import build_embedder, open_retriever
from triage.settings import Settings


def _data_generate(settings: Settings, args: argparse.Namespace) -> int:
    write_files(settings.data_dir)
    print(f"wrote the corpus, the incidents and the labels to {settings.data_dir}/")
    return 0


def _data_check(settings: Settings, args: argparse.Namespace) -> int:
    problems = checks.run_all(settings.data_dir)
    for problem in problems:
        print(f"problem: {problem}")
    if problems:
        return 1
    print(json.dumps(checks.summary(settings.data_dir), indent=2))
    return 0


def _data_embed(settings: Settings, args: argparse.Namespace) -> int:
    """Brings the snapshot in line with the data.

    A vector that is already in the snapshot is kept, so that running this on another machine
    does not move every vector by a rounding error. `--rebuild` computes all of them again.
    """
    path = settings.data_dir / SNAPSHOT_FILE
    old = Snapshot() if args.rebuild or not path.exists() else Snapshot.load(path)
    corpus = load_corpus(settings.data_dir)
    labelled = load_labelled(settings.data_dir)
    texts = [document.text for document in corpus]
    texts += [incident_query(item.incident) for item in labelled]
    missing = [text for text in texts if old.get(text) is None]
    computed = LocalEmbedder(settings.embedding_cache_dir).embed(missing) if missing else []
    fresh = dict(zip(missing, computed, strict=True))
    new = Snapshot()
    for text in texts:
        kept = old.get(text)
        new.put(text, kept if kept is not None else fresh[text])
    new.save(path)
    print(f"{path}: {len(texts)} vectors, {len(missing)} computed")
    return 0


def _db_load(settings: Settings, args: argparse.Namespace) -> int:
    corpus = load_corpus(settings.data_dir)
    embedder = build_embedder(settings, snapshot_only=False)
    count = db.load_corpus(settings.database_url, corpus, embedder)
    print(f"loaded {count} documents")
    return 0


def _search(settings: Settings, args: argparse.Namespace) -> int:
    async def run() -> None:
        async with open_retriever(settings, snapshot_only=False) as retriever:
            for hit in await retriever.search(args.query, args.mode, args.limit):
                print(f"{hit.rank}. {hit.id}  {hit.score:.4f}  {hit.title}")

    asyncio.run(run())
    return 0


def _serve(settings: Settings, args: argparse.Namespace) -> int:
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def _mcp(settings: Settings, args: argparse.Namespace) -> int:
    asyncio.run(serve_stdio(settings))
    return 0


# ----------------------------------------------------------------------------------------------
# Evaluation


def _results_dir(settings: Settings) -> Path:
    return settings.eval_dir / gate.RESULTS_DIR


def _committed_summary(settings: Settings) -> dict[str, Any]:
    path = _results_dir(settings) / runner.SUMMARY_FILE
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _eval_retrieval(settings: Settings, args: argparse.Namespace) -> int:
    print(json.dumps(asyncio.run(gate.measure_retrieval(settings)), indent=2))
    return 0


def _eval_run(settings: Settings, args: argparse.Namespace) -> int:
    """A run against the API. It pays only for calls that have no recording yet."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; `triage eval replay` needs no key", file=sys.stderr)
        return 2
    labelled = load_labelled(settings.data_dir)
    cases = runner.sample_cases(labelled, args.cases) if args.cases else None
    variants = VARIANTS
    if args.variants:
        variants = tuple(BY_NAME[name] for name in args.variants.split(","))
    partial = cases is not None or args.reps != runner.REPETITIONS or variants != VARIANTS
    if partial and args.out is None:
        print("a partial run needs --out, so that it cannot replace the results", file=sys.stderr)
        return 2
    budget = Budget(max_usd=args.max_usd)
    try:
        evaluation = asyncio.run(
            runner.evaluate(
                settings,
                live=True,
                variants=variants,
                cases=cases,
                reps=args.reps,
                budget=budget,
                judge=args.judge,
                control_count=args.controls,
            )
        )
    except BudgetExceeded as error:
        print(f"stopped: {error}. What was answered is recorded and will not be paid for again.")
        return 3
    run_info = {
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "paid_usd": round(budget.spent_usd, 4),
        "paid_calls": budget.paid_calls,
        "judge": args.judge,
        "anthropic_sdk": anthropic.__version__,
        **evaluation.timings,
    }
    out = args.out or _results_dir(settings)
    runner.write(evaluation, out, run_info)
    print(report.results_table(evaluation.summary))
    print(f"paid ${budget.spent_usd:.4f} for {budget.paid_calls} calls; results in {out}/")
    if evaluation.errors:
        print(f"{len(evaluation.errors)} attempts gave no answer; see {out}/{runner.ERRORS_FILE}")
        return 1
    return 0


def _eval_estimate(settings: Settings, args: argparse.Namespace) -> int:
    """Projects the cost of the full run from the rows of a sample run."""
    with (args.sample / runner.ROWS_FILE).open(encoding="utf-8") as lines:
        rows = [Row.model_validate_json(line) for line in lines if line.strip()]
    incidents = len(load_labelled(settings.data_dir))
    planned = incidents * runner.REPETITIONS
    agents = 0.0
    judged_answers = 0.0
    print(f"per variant, for {incidents} incidents x {runner.REPETITIONS} repetitions:")
    for variant in VARIANTS:
        of_variant = [row for row in rows if row.variant == variant.name]
        if not of_variant:
            print(f"  {variant.name}: not in the sample")
            continue
        per_row = sum(row.cost_usd for row in of_variant) / len(of_variant)
        judged = sum(1 for row in of_variant if row.judge is not None) / len(of_variant)
        agents += per_row * planned
        judged_answers += judged * planned
        print(
            f"  {variant.name}: ${per_row:.5f} per incident over {len(of_variant)} sampled, "
            f"${per_row * planned:.2f} in all"
        )
    verdicts = [row for row in rows if row.judge is not None]
    per_verdict = sum(row.judge_cost_usd for row in verdicts) / len(verdicts) if verdicts else 0.0
    controls = len(
        judge.controls(load_catalogue(settings.data_dir), load_corpus(settings.data_dir))
    )
    judge_direct = per_verdict * (judged_answers + controls)
    print(
        f"judge: ${per_verdict:.5f} per verdict over {len(verdicts)} sampled, about "
        f"{judged_answers:.0f} answers and {controls} controls: ${judge_direct:.2f} direct, "
        f"${judge_direct / 2:.2f} through the Batch API"
    )
    print(f"estimate for one full run: ${agents + judge_direct / 2:.2f} with the judge in a batch")
    return 0


def _eval_replay(settings: Settings, args: argparse.Namespace) -> int:
    try:
        evaluation = asyncio.run(runner.evaluate(settings, live=False, log=lambda _: None))
    except runner.StaleResults as error:
        print(f"stale: {error}", file=sys.stderr)
        return 1
    if args.out:
        runner.write(evaluation, args.out, None)
    print(report.results_table(evaluation.summary))
    return 0


def _eval_baseline(settings: Settings, args: argparse.Namespace) -> int:
    path = settings.eval_dir / gate.BASELINE_FILE
    baseline = gate.make_baseline(_committed_summary(settings))
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} from {_results_dir(settings)}/{runner.SUMMARY_FILE}")
    return 0


def _eval_gate(settings: Settings, args: argparse.Namespace) -> int:
    findings = asyncio.run(gate.run(settings, args.readme))
    for finding in findings:
        print(finding)
    failed = [finding for finding in findings if not finding.ok]
    print(f"{len(findings) - len(failed)} checks passed, {len(failed)} failed")
    return 1 if failed else 0


def _eval_report(settings: Settings, args: argparse.Namespace) -> int:
    summary = _committed_summary(settings)
    if args.update:
        args.update.write_text(
            report.with_table(args.update.read_text(encoding="utf-8"), summary), encoding="utf-8"
        )
        print(f"updated the table in {args.update}")
        return 0
    print(report.detail_table(summary) if args.detail else report.results_table(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="the synthetic data set").add_subparsers(
        dest="data_command", required=True
    )
    data.add_parser("generate", help="render the catalogue into the data files").set_defaults(
        run=_data_generate
    )
    data.add_parser("check", help="check the data files and print their shape").set_defaults(
        run=_data_check
    )
    embed = data.add_parser("embed", help="bring the embedding snapshot in line with the data")
    embed.add_argument("--rebuild", action="store_true", help="compute every vector again")
    embed.set_defaults(run=_data_embed)

    database = commands.add_parser("db", help="the database").add_subparsers(
        dest="db_command", required=True
    )
    database.add_parser("load", help="create the schema and load the corpus").set_defaults(
        run=_db_load
    )

    search = commands.add_parser("search", help="search the corpus")
    search.add_argument("query")
    search.add_argument("--mode", choices=["vector", "fulltext", "hybrid"], default="hybrid")
    search.add_argument("--limit", type=int, default=5)
    search.set_defaults(run=_search)

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(run=_serve)

    commands.add_parser("mcp", help="serve the evidence tools over stdio").set_defaults(run=_mcp)

    evaluation = commands.add_parser("eval", help="the evaluation").add_subparsers(
        dest="eval_command", required=True
    )
    evaluation.add_parser("retrieval", help="measure retrieval alone").set_defaults(
        run=_eval_retrieval
    )
    run = evaluation.add_parser("run", help="run against the API; pays for unrecorded calls")
    run.add_argument("--max-usd", type=float, required=True, help="stop paying at this amount")
    run.add_argument("--cases", type=int, help="only this many incidents, as a sample")
    run.add_argument("--reps", type=int, default=runner.REPETITIONS)
    run.add_argument("--variants", help="comma-separated names; all by default")
    run.add_argument("--judge", choices=["batch", "direct", "none"], default="batch")
    run.add_argument("--controls", type=int, help="only this many judge controls")
    run.add_argument("--out", type=Path, help="where to write; required for a partial run")
    run.set_defaults(run=_eval_run)
    estimate = evaluation.add_parser("estimate", help="project the cost of a full run")
    estimate.add_argument("sample", type=Path, help="the --out directory of a sample run")
    estimate.set_defaults(run=_eval_estimate)
    replay = evaluation.add_parser("replay", help="rebuild the results from the recordings")
    replay.add_argument("--out", type=Path, help="write the rebuilt results here")
    replay.set_defaults(run=_eval_replay)
    evaluation.add_parser(
        "baseline", help="accept the committed results as the new baseline"
    ).set_defaults(run=_eval_baseline)
    check = evaluation.add_parser("gate", help="what CI runs: fail if anything got worse")
    check.add_argument("--readme", type=Path, help="also check the results table of this file")
    check.set_defaults(run=_eval_gate)
    table = evaluation.add_parser("report", help="print the results table")
    table.add_argument("--detail", action="store_true", help="the longer table of the docs")
    table.add_argument("--update", type=Path, help="write the table into this file's markers")
    table.set_defaults(run=_eval_report)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.run(Settings(), args))


if __name__ == "__main__":
    main()
