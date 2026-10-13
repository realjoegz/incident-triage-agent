"""One row per (variant, incident, repetition): what the agent did, with nothing aggregated.

A row holds no wall-clock measurement of this machine. Its latency is that of the recorded model
calls, so a row rebuilt from the recordings in CI is byte for byte the row of the paid run.
"""

from pydantic import BaseModel, ConfigDict

from triage.agent import AgentRun, Outcome
from triage.llm.base import Usage
from triage.llm.pricing import cost_usd
from triage.models import Citation, Decision
from triage.taxonomy import Category


class CallSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    fingerprint: str
    model: str
    stop_reason: str
    usage: Usage
    latency_s: float
    attempts: int


class Verdict(BaseModel):
    """What the judge returns. `reason` comes first so that it is written before the verdict."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    supported: bool


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: str
    case_id: str
    rep: int
    retrieved: list[str]
    outcome: Outcome
    decision: Decision
    category: Category
    likely_cause: str
    next_step: str
    citations: list[Citation]
    abstained_because: str | None
    draft_citations: int
    draft_citations_valid: int
    repaired: bool
    calls: list[CallSummary]
    cost_usd: float
    latency_s: float
    # Filled by the judge for an answer that names a cause and cites for it.
    judge: Verdict | None = None
    judge_cost_usd: float = 0.0

    @classmethod
    def from_run(cls, variant: str, case_id: str, rep: int, run: AgentRun) -> "Row":
        calls = [
            CallSummary(
                step=call.step,
                fingerprint=call.fingerprint,
                model=call.response.model,
                stop_reason=call.response.stop_reason,
                usage=call.response.usage,
                latency_s=call.response.latency_s,
                attempts=call.response.attempts,
            )
            for call in run.calls
        ]
        return cls(
            variant=variant,
            case_id=case_id,
            rep=rep,
            retrieved=[item.id for item in run.evidence],
            outcome=run.outcome,
            decision=run.triage.decision,
            category=run.triage.cause_category,
            likely_cause=run.triage.likely_cause,
            next_step=run.triage.next_step,
            citations=run.triage.citations,
            abstained_because=run.triage.abstained_because,
            draft_citations=run.draft_citations,
            draft_citations_valid=run.draft_citations_valid,
            repaired=run.repaired,
            calls=calls,
            cost_usd=round(sum(cost_usd(call.model, call.usage) for call in calls), 8),
            latency_s=round(sum(call.latency_s for call in calls), 3),
        )


class ErrorRow(BaseModel):
    """An attempt that produced no answer to score: the API failed or the time ran out.

    These never go among the rows, where a missing answer would be counted as a wrong one.
    """

    variant: str
    case_id: str
    rep: int
    kind: str
    detail: str
