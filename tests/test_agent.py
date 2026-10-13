import pytest

from tests.fakes import (
    DOC_A,
    DOC_B,
    QUOTE_A,
    QUOTE_B,
    FixedSearch,
    ScriptedLLM,
    abstention,
    draft,
    response,
)
from triage.agent import Agent, AgentConfig
from triage.agent.prompts import SYSTEM_WITH_EVIDENCE, SYSTEM_WITHOUT_EVIDENCE
from triage.models import Incident
from triage.taxonomy import KNOWN_CATEGORIES, Category

INCIDENT = Incident(
    alert="payments-gateway: authorisation decline rate above 8%",
    log_excerpt="2026-10-01T09:50:45Z ERROR payments-gateway-2 PGW-4012 declined code=R29",
    ticket_text="Cards are declined in waves.",
)


def agent(llm: ScriptedLLM, search: FixedSearch, mode: str = "hybrid") -> Agent:
    return Agent(llm, search, AgentConfig(model="test-model", mode=mode))  # type: ignore[arg-type]


async def test_an_answer_whose_quotes_are_in_the_documents_is_returned_as_given() -> None:
    llm = ScriptedLLM(response(draft(("INC-A", QUOTE_A), ("RB-B", QUOTE_B))))
    search = FixedSearch([DOC_A, DOC_B])

    run = await agent(llm, search).run(INCIDENT)

    assert run.outcome == "answered"
    assert run.triage.decision == "triage"
    assert run.triage.cause_category is Category.RATE_LIMITING
    assert [citation.doc_id for citation in run.triage.citations] == ["INC-A", "RB-B"]
    assert run.triage.evidence_ids == ["INC-A", "RB-B"]
    assert run.triage.abstained_because is None
    assert (run.draft_citations, run.draft_citations_valid, run.repaired) == (2, 2, False)
    assert [call.step for call in run.calls] == ["draft"]


async def test_the_search_is_given_the_incident_text_the_mode_and_the_limit() -> None:
    search = FixedSearch([DOC_A])

    await Agent(
        ScriptedLLM(response(abstention())),
        search,
        AgentConfig(model="test-model", mode="vector", top_k=3),
    ).run(INCIDENT)

    [(query, mode, limit)] = search.calls
    assert (mode, limit) == ("vector", 3)
    assert query.splitlines() == [
        INCIDENT.alert,
        INCIDENT.ticket_text,
        "PGW-4012 declined code=R29",
    ]


async def test_the_prompt_holds_the_incident_and_each_document_in_rank_order() -> None:
    llm = ScriptedLLM(response(abstention()))

    await agent(llm, FixedSearch([DOC_A, DOC_B])).run(INCIDENT)

    [request] = llm.requests
    assert request.system == SYSTEM_WITH_EVIDENCE
    assert all(category.value in request.system for category in KNOWN_CATEGORIES)
    for part in (
        INCIDENT.alert,
        INCIDENT.ticket_text,
        INCIDENT.log_excerpt,
        DOC_A.body,
        DOC_B.body,
    ):
        assert part in request.user
    assert request.user.index('<document id="INC-A"') < request.user.index('<document id="RB-B"')


async def test_a_quote_that_is_not_in_its_document_is_sent_back_once_for_repair() -> None:
    llm = ScriptedLLM(
        response(draft(("INC-A", "Meridian limits us to 300 authorisations a second"))),
        response(draft(("INC-A", QUOTE_A))),
    )

    run = await agent(llm, FixedSearch([DOC_A, DOC_B])).run(INCIDENT)

    assert run.triage.decision == "triage"
    assert [citation.quote for citation in run.triage.citations] == [QUOTE_A]
    assert (run.draft_citations, run.draft_citations_valid, run.repaired) == (1, 0, True)
    assert [call.step for call in run.calls] == ["draft", "repair"]
    repair = llm.requests[1].user
    assert repair.startswith(llm.requests[0].user)
    assert (
        'the quote "Meridian limits us to 300 authorisations a second" does not appear in INC-A'
        in repair
    )


async def test_when_the_repair_also_fails_the_agent_says_the_evidence_is_insufficient() -> None:
    invented = draft(("INC-A", "The acquirer had an outage in its primary region"))
    llm = ScriptedLLM(response(invented), response(invented))

    run = await agent(llm, FixedSearch([DOC_A])).run(INCIDENT)

    assert run.triage.decision == "insufficient_evidence"
    assert run.triage.cause_category is Category.UNKNOWN
    assert run.triage.likely_cause == ""
    assert run.triage.citations == []
    assert run.triage.abstained_because == "no citation could be verified"
    assert len(llm.requests) == 2


async def test_after_a_repair_the_valid_citations_are_kept_and_the_others_dropped() -> None:
    mixed = draft(("INC-A", QUOTE_A), ("RB-B", "Raise the limit with the acquirer at once"))
    llm = ScriptedLLM(response(mixed), response(mixed))

    run = await agent(llm, FixedSearch([DOC_A, DOC_B])).run(INCIDENT)

    assert run.triage.decision == "triage"
    assert [citation.doc_id for citation in run.triage.citations] == ["INC-A"]
    assert (run.draft_citations, run.draft_citations_valid, run.repaired) == (2, 1, True)


async def test_a_cause_without_any_citation_is_sent_back_and_then_refused() -> None:
    llm = ScriptedLLM(response(draft()), response(draft()))

    run = await agent(llm, FixedSearch([DOC_A])).run(INCIDENT)

    assert "names a cause but cites no document" in llm.requests[1].user
    assert run.triage.decision == "insufficient_evidence"
    assert run.triage.abstained_because == "no citation could be verified"


async def test_a_citation_of_a_document_that_was_not_retrieved_does_not_count() -> None:
    elsewhere = draft(("INC-2025-9999", QUOTE_A))
    llm = ScriptedLLM(response(elsewhere), response(elsewhere))

    run = await agent(llm, FixedSearch([DOC_A])).run(INCIDENT)

    assert "INC-2025-9999 is not one of the evidence documents" in llm.requests[1].user
    assert run.triage.decision == "insufficient_evidence"


async def test_when_the_model_abstains_its_answer_stands_without_a_repair() -> None:
    llm = ScriptedLLM(response(abstention()))

    run = await agent(llm, FixedSearch([DOC_A])).run(INCIDENT)

    assert run.triage.decision == "insufficient_evidence"
    assert run.triage.abstained_because is None
    assert run.triage.next_step == "Check the service dashboard."
    assert (run.draft_citations, run.repaired, len(llm.requests)) == (0, False, 1)


async def test_a_cause_with_the_category_unknown_is_not_a_triage() -> None:
    llm = ScriptedLLM(response(draft(("INC-A", QUOTE_A), category=Category.UNKNOWN)))

    run = await agent(llm, FixedSearch([DOC_A])).run(INCIDENT)

    assert run.triage.decision == "insufficient_evidence"
    assert run.triage.abstained_because == "the answer named a cause but no category"


async def test_when_the_search_returns_nothing_the_model_is_not_called() -> None:
    llm = ScriptedLLM()

    run = await agent(llm, FixedSearch([])).run(INCIDENT)

    assert llm.requests == []
    assert run.calls == []
    assert run.triage.decision == "insufficient_evidence"
    assert run.triage.abstained_because == "the search returned no document"


async def test_without_retrieval_the_model_sees_only_the_incident_and_cites_nothing() -> None:
    llm = ScriptedLLM(response(draft(("INC-A", QUOTE_A))))
    search = FixedSearch([DOC_A])

    run = await agent(llm, search, mode="none").run(INCIDENT)

    assert search.calls == []
    [request] = llm.requests
    assert request.system == SYSTEM_WITHOUT_EVIDENCE
    assert "<evidence>" not in request.user
    assert run.triage.decision == "triage"
    assert run.triage.cause_category is Category.RATE_LIMITING
    assert run.triage.citations == []
    assert run.triage.evidence_ids == []
    assert (run.draft_citations, run.repaired) == (0, False)


@pytest.mark.parametrize(
    ("stop_reason", "text", "outcome"),
    [
        ("refusal", None, "refused"),
        ("max_tokens", '{"decision": "tria', "truncated"),
        ("end_turn", "not json", "invalid_output"),
        ("end_turn", None, "invalid_output"),
        ("end_turn", '{"decision": "triage"}', "invalid_output"),
    ],
)
async def test_a_response_without_a_usable_answer_is_reported_as_what_it_was(
    stop_reason: str, text: str | None, outcome: str
) -> None:
    llm = ScriptedLLM(response(text, stop_reason=stop_reason))

    run = await agent(llm, FixedSearch([DOC_A])).run(INCIDENT)

    assert run.outcome == outcome
    assert run.triage.decision == "insufficient_evidence"
    assert run.triage.abstained_because == f"the model gave no usable answer ({outcome})"
    assert len(llm.requests) == 1


async def test_a_refusal_during_the_repair_ends_the_run_as_refused() -> None:
    llm = ScriptedLLM(
        response(draft(("INC-A", "words that are not in the document at all"))),
        response(None, stop_reason="refusal"),
    )

    run = await agent(llm, FixedSearch([DOC_A])).run(INCIDENT)

    assert run.outcome == "refused"
    assert run.triage.decision == "insufficient_evidence"
    assert [call.step for call in run.calls] == ["draft", "repair"]


async def test_an_empty_incident_is_rejected_before_anything_is_searched() -> None:
    search = FixedSearch([DOC_A])

    with pytest.raises(ValueError, match="needs an alert"):
        await agent(ScriptedLLM(), search).run(Incident(alert="  ", ticket_text="\n"))

    assert search.calls == []
