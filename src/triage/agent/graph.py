"""The agent: a fixed pipeline, written as a LangGraph state graph.

    retrieve -> draft -> check -> finalize
                  ^        |
                  +- repair (once)

The model is called at `draft`, and once more at `repair` when a citation of the draft failed
the check. Everything else is code. In particular it is code, not the model, that decides
whether an answer that names a cause is allowed out: it needs at least one citation whose quote
is in its document, and otherwise the agent says that the evidence is insufficient.
"""

import operator
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ValidationError

from triage.agent import citations, prompts
from triage.llm.base import LLM, Effort, LLMRequest, LLMResponse
from triage.models import Evidence, Incident, SearchMode, Triage, TriageDraft
from triage.retrieval import incident_query
from triage.taxonomy import Category

# `none` is the variant without retrieval: the model sees the incident and nothing else.
AgentMode = Literal["none", "vector", "fulltext", "hybrid"]

# How a run ended as far as the model is concerned. Only `answered` produced a usable answer;
# the other three are reported apart from wrong answers and never scored as one.
Outcome = Literal["answered", "refused", "truncated", "invalid_output"]

Search = Callable[[str, SearchMode, int], Awaitable[list[Evidence]]]


@dataclass(frozen=True)
class AgentConfig:
    model: str
    effort: Effort = "low"
    mode: AgentMode = "hybrid"
    top_k: int = 5
    # Thinking counts towards this, so it is far above the size of the answer itself.
    max_tokens: int = 8000


class Call(BaseModel):
    """One call to the model."""

    step: Literal["draft", "repair"]
    fingerprint: str
    response: LLMResponse


class AgentRun(BaseModel):
    triage: Triage
    evidence: list[Evidence]
    calls: list[Call]
    outcome: Outcome
    # The citations of the first draft, before any repair: how many, and how many passed.
    draft_citations: int
    draft_citations_valid: int
    repaired: bool
    retrieval_s: float
    total_s: float


class _State(TypedDict, total=False):
    incident: Incident
    evidence: list[Evidence]
    retrieval_s: float
    prompt: str
    draft: TriageDraft | None
    outcome: Outcome
    calls: Annotated[list[Call], operator.add]
    checked: list[citations.Checked]
    first_check: tuple[int, int]
    repaired: bool
    triage: Triage


class Agent:
    def __init__(self, llm: LLM, search: Search, config: AgentConfig) -> None:
        self._llm = llm
        self._search = search
        self.config = config
        self._graph = self._build()

    async def run(self, incident: Incident) -> AgentRun:
        if not (
            incident.alert.strip() or incident.log_excerpt.strip() or incident.ticket_text.strip()
        ):
            raise ValueError("an incident needs an alert, a log excerpt or a ticket")
        started = time.perf_counter()
        state = cast(_State, await self._graph.ainvoke({"incident": incident, "calls": []}))
        drafted, valid = state.get("first_check", (0, 0))
        return AgentRun(
            triage=state["triage"],
            evidence=state.get("evidence", []),
            calls=state["calls"],
            outcome=state.get("outcome", "answered"),
            draft_citations=drafted,
            draft_citations_valid=valid,
            repaired=state.get("repaired", False),
            retrieval_s=round(state.get("retrieval_s", 0.0), 4),
            total_s=round(time.perf_counter() - started, 4),
        )

    # ------------------------------------------------------------------------------------------
    # Nodes

    async def _retrieve(self, state: _State) -> _State:
        if self.config.mode == "none":
            return {"evidence": [], "retrieval_s": 0.0}
        started = time.perf_counter()
        evidence = await self._search(
            incident_query(state["incident"]), self.config.mode, self.config.top_k
        )
        return {"evidence": evidence, "retrieval_s": time.perf_counter() - started}

    async def _draft(self, state: _State) -> _State:
        evidence = None if self.config.mode == "none" else state["evidence"]
        prompt = prompts.user_prompt(state["incident"], evidence)
        return {"prompt": prompt, **await self._ask("draft", prompt)}

    async def _repair(self, state: _State) -> _State:
        draft = state["draft"]
        assert draft is not None
        problems = [item.problem for item in state["checked"] if item.problem]
        if not state["checked"]:
            problems = ["the answer names a cause but cites no document"]
        prompt = prompts.repair_prompt(state["prompt"], draft, problems)
        return {"repaired": True, **await self._ask("repair", prompt)}

    async def _ask(self, step: Literal["draft", "repair"], user: str) -> _State:
        with_evidence = self.config.mode != "none"
        request = LLMRequest(
            model=self.config.model,
            effort=self.config.effort,
            max_tokens=self.config.max_tokens,
            system=prompts.SYSTEM_WITH_EVIDENCE
            if with_evidence
            else prompts.SYSTEM_WITHOUT_EVIDENCE,
            user=user,
            schema=TriageDraft,
        )
        response = await self._llm.complete(request)
        draft: TriageDraft | None = None
        outcome: Outcome
        if response.stop_reason == "refusal":
            outcome = "refused"
        elif response.stop_reason == "max_tokens":
            outcome = "truncated"
        else:
            try:
                draft = TriageDraft.model_validate_json(response.text or "")
                outcome = "answered"
            except ValidationError:
                outcome = "invalid_output"
        call = Call(step=step, fingerprint=request.fingerprint(), response=response)
        return {"draft": draft, "outcome": outcome, "calls": [call]}

    def _check(self, state: _State) -> _State:
        draft = state["draft"]
        assert draft is not None
        checked: list[citations.Checked] = []
        if self.config.mode != "none" and draft.decision == "triage":
            checked = citations.check(draft.citations, state["evidence"])
        update: _State = {"checked": checked}
        if "first_check" not in state:
            update["first_check"] = (len(checked), sum(item.valid for item in checked))
        return update

    def _finalize(self, state: _State) -> _State:
        evidence = state.get("evidence", [])
        draft = state.get("draft")
        outcome = state.get("outcome", "answered")

        def abstain(because: str | None, next_step: str = "") -> _State:
            return {
                "triage": Triage(
                    decision="insufficient_evidence",
                    cause_category=Category.UNKNOWN,
                    likely_cause="",
                    citations=[],
                    next_step=next_step,
                    evidence_ids=[item.id for item in evidence],
                    abstained_because=because,
                )
            }

        if self.config.mode != "none" and not evidence:
            return abstain("the search returned no document")
        if outcome != "answered" or draft is None:
            return abstain(f"the model gave no usable answer ({outcome})")
        if draft.decision == "insufficient_evidence":
            return abstain(None, draft.next_step)
        if draft.cause_category is Category.UNKNOWN:
            return abstain("the answer named a cause but no category", draft.next_step)

        valid = [item.citation for item in state.get("checked", []) if item.valid]
        if self.config.mode != "none" and not valid:
            return abstain("no citation could be verified", draft.next_step)
        return {
            "triage": Triage(
                decision="triage",
                cause_category=draft.cause_category,
                likely_cause=draft.likely_cause,
                citations=valid,
                next_step=draft.next_step,
                evidence_ids=[item.id for item in evidence],
            )
        }

    # ------------------------------------------------------------------------------------------
    # Edges

    def _after_retrieve(self, state: _State) -> Literal["draft", "finalize"]:
        # Nothing to show the model, so nothing to ask it.
        if self.config.mode != "none" and not state["evidence"]:
            return "finalize"
        return "draft"

    def _after_answer(self, state: _State) -> Literal["check", "finalize"]:
        return "check" if state["outcome"] == "answered" else "finalize"

    def _after_check(self, state: _State) -> Literal["repair", "finalize"]:
        draft = state["draft"]
        assert draft is not None
        if self.config.mode == "none" or draft.decision != "triage" or state.get("repaired"):
            return "finalize"
        every_citation_valid = bool(state["checked"]) and all(
            item.valid for item in state["checked"]
        )
        return "finalize" if every_citation_valid else "repair"

    def _build(self) -> CompiledStateGraph[_State, None, _State, _State]:
        graph = StateGraph(_State)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("draft", self._draft)
        graph.add_node("check", self._check)
        graph.add_node("repair", self._repair)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges("retrieve", self._after_retrieve, ["draft", "finalize"])
        graph.add_conditional_edges("draft", self._after_answer, ["check", "finalize"])
        graph.add_conditional_edges("check", self._after_check, ["repair", "finalize"])
        graph.add_conditional_edges("repair", self._after_answer, ["check", "finalize"])
        graph.add_edge("finalize", END)
        return graph.compile()
