"""Test doubles for the model and for the search."""

from collections.abc import Callable, Sequence

from triage.agent.prompts import incident_block
from triage.dataset.catalogue import Catalogue
from triage.evaluation.judge import SEEN
from triage.evaluation.rows import Verdict
from triage.llm.base import LLMRequest, LLMResponse, Usage
from triage.models import (
    Citation,
    Evidence,
    Label,
    LabelledIncident,
    SearchMode,
    TriageDraft,
)
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


# ----------------------------------------------------------------------------------------------
# Stand-ins for the model in tests of the evaluation: each answers every request by a rule.


class RuleLLM:
    """Answers triage requests with `triage(label, prompt)` and judge requests by construction.

    The judge rule is the one the controls are built on. A stated cause is supported when it
    is, word for word, a cause of the catalogue that a cited document states, followed by
    nothing or by a sentence about the incident that the incident contains.
    """

    def __init__(
        self,
        labelled: Sequence[LabelledIncident],
        catalogue: Catalogue,
        triage: Callable[[Label, str], TriageDraft],
    ) -> None:
        self._labels = {incident_block(item.incident): item.label for item in labelled}
        self._causes = [fault.cause for fault in catalogue.faults]
        self._triage = triage
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if request.schema is Verdict:
            before, documents = request.user.split("</stated_cause>")
            incident, stated = before.split("<stated_cause", 1)
            stated = stated.split(">\n", 1)[1].strip()
            supported = False
            for cause in self._causes:
                if stated.startswith(cause) and cause in documents:
                    rest = stated[len(cause) :].strip()
                    supported = not rest or (
                        rest.startswith(SEEN) and rest[len(SEEN) :] in incident
                    )
            text = Verdict(reason="by construction", supported=supported).model_dump_json()
        else:
            block = request.user.split("\n\n<evidence>")[0].split("\n\n<previous_answer>")[0]
            text = self._triage(self._labels[block], request.user).model_dump_json()
        return LLMResponse(
            text=text,
            stop_reason="end_turn",
            model=request.model,
            usage=Usage(input_tokens=2000, output_tokens=500),
            latency_s=1.0,
            attempts=1,
        )


def oracle(catalogue: Catalogue) -> Callable[[Label, str], TriageDraft]:
    """Knows the answer key. Names the labelled cause whenever it can cite a document for it."""
    faults = {fault.id: fault for fault in catalogue.faults}

    def answer(label: Label, prompt: str) -> TriageDraft:
        if not label.answerable:
            return abstention()
        fault = faults[label.fault_id]
        if "<evidence>" not in prompt:
            return draft(category=label.category, likely_cause=fault.cause)
        shown = [doc_id for doc_id in label.doc_ids if f'<document id="{doc_id}"' in prompt]
        if not shown:
            return abstention()
        quote = fault.cause.split(". ")[0] + "."
        return draft((shown[0], quote), category=label.category, likely_cause=fault.cause)

    return answer


def always_abstains(label: Label, prompt: str) -> TriageDraft:
    return abstention()


def always_rate_limiting(label: Label, prompt: str) -> TriageDraft:
    """Names one cause for everything and cites words that no document contains."""
    return draft(("INC-2025-0173", "the acquirer confirmed that it throttled every request"))
