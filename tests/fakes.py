"""Test doubles for the model and for the search."""

from collections.abc import Sequence

from triage.llm.base import LLMRequest, LLMResponse, Usage
from triage.models import Citation, Evidence, SearchMode, TriageDraft
from triage.taxonomy import Category

DOC_A = Evidence(
    id="INC-A",
    kind="incident",
    service="payments-gateway",
    title="Card authorisations declined in bursts",
    body="Root cause: Meridian Acquiring limits our merchant account to 300 authorisations "
    "per second. It answers traffic above that limit with R29 declines.",
    rank=1,
    score=0.9,
)
DOC_B = Evidence(
    id="RB-B",
    kind="runbook",
    service="payments-gateway",
    title="Card authorisations declined in bursts",
    body="Steps:\n1. Compare authorisations per second on 'Gateway traffic' with the limit "
    "of 300 per second.",
    rank=2,
    score=0.8,
)

QUOTE_A = "Meridian Acquiring limits our merchant account to 300 authorisations per second"
QUOTE_B = "Compare authorisations per second on 'Gateway traffic'"


def draft(
    *citations: tuple[str, str],
    decision: str = "triage",
    category: Category = Category.RATE_LIMITING,
    likely_cause: str = "The acquirer throttled the account.",
    next_step: str = "Compare the traffic with the limit.",
) -> TriageDraft:
    return TriageDraft.model_validate(
        {
            "decision": decision,
            "cause_category": category,
            "likely_cause": likely_cause,
            "citations": [Citation(doc_id=doc_id, quote=quote) for doc_id, quote in citations],
            "next_step": next_step,
        }
    )


def abstention() -> TriageDraft:
    return draft(
        decision="insufficient_evidence",
        category=Category.UNKNOWN,
        likely_cause="",
        next_step="Check the service dashboard.",
    )


def response(
    answer: TriageDraft | str | None, *, stop_reason: str = "end_turn", model: str = "test-model"
) -> LLMResponse:
    text = answer.model_dump_json() if isinstance(answer, TriageDraft) else answer
    return LLMResponse(
        text=text,
        stop_reason=stop_reason,
        model=model,
        usage=Usage(input_tokens=1000, output_tokens=200),
        latency_s=0.5,
        attempts=1,
    )


class ScriptedLLM:
    """Returns the given responses in order and keeps the requests it was sent."""

    def __init__(self, *responses: LLMResponse) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("the model was called more often than the test scripted")
        return self._responses.pop(0)


class FixedSearch:
    def __init__(self, evidence: Sequence[Evidence]) -> None:
        self._evidence = list(evidence)
        self.calls: list[tuple[str, SearchMode, int]] = []

    async def __call__(self, query: str, mode: SearchMode, limit: int) -> list[Evidence]:
        self.calls.append((query, mode, limit))
        return self._evidence[:limit]
