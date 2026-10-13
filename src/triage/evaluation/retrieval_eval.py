"""Retrieval measured on its own: no model, no cost, the same on every machine.

This is the part of the evaluation that CI can run in full. A change to the search either
moves these numbers or it does not, and no API key is needed to find out.
"""

import time
from collections import defaultdict
from collections.abc import Sequence

from triage.agent.graph import Search
from triage.evaluation.metrics import DECIMALS, mean, percentile
from triage.models import LabelledIncident, SearchMode
from triage.retrieval import incident_query

MODES: tuple[SearchMode, ...] = ("vector", "fulltext", "hybrid")
TOP_K = 5


async def measure(
    search: Search, labelled: Sequence[LabelledIncident], k: int = TOP_K
) -> tuple[dict[str, dict[str, object]], dict[str, float]]:
    """Recall, hit rate and reciprocal rank per mode, and how long the searches took.

    The first value is deterministic. The timings are of this machine and are kept apart.
    """
    with_evidence = [item for item in labelled if item.label.answerable]
    metrics: dict[str, dict[str, object]] = {}
    durations_ms: list[float] = []
    for mode in MODES:
        recall, hit, reciprocal = [], [], []
        by_presentation: dict[str, list[float]] = defaultdict(list)
        for item in with_evidence:
            started = time.perf_counter()
            retrieved = [hit_.id for hit_ in await search(incident_query(item.incident), mode, k)]
            durations_ms.append((time.perf_counter() - started) * 1000)
            gold = set(item.label.doc_ids)
            found = [doc_id for doc_id in retrieved if doc_id in gold]
            first = next((rank for rank, doc in enumerate(retrieved, 1) if doc in gold), None)
            recall.append(len(found) / len(gold))
            hit.append(float(bool(found)))
            reciprocal.append(1.0 / first if first else 0.0)
            by_presentation[item.label.presentation].append(len(found) / len(gold))
        metrics[mode] = {
            "incidents": len(with_evidence),
            "recall_at_5": round(mean(recall) or 0.0, DECIMALS),
            "hit_at_5": round(mean(hit) or 0.0, DECIMALS),
            "mrr": round(mean(reciprocal) or 0.0, DECIMALS),
            "recall_at_5_by_presentation": {
                name: round(mean(values) or 0.0, DECIMALS)
                for name, values in sorted(by_presentation.items())
            },
        }
    timings = {
        "searches": float(len(durations_ms)),
        "search_ms_p50": round(percentile(durations_ms, 0.50) or 0.0, 2),
        "search_ms_p95": round(percentile(durations_ms, 0.95) or 0.0, 2),
    }
    return metrics, timings
