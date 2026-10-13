from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from tests.fakes import ScriptedLLM, abstention, draft, response
from triage.api import MAX_FIELD_CHARS, create_app
from triage.llm.base import LLMError, LLMRequest, LLMResponse
from triage.llm.pricing import cost_usd
from triage.models import LabelledIncident
from triage.settings import Settings


@contextmanager
def serving(settings: Settings, llm: object) -> Iterator[TestClient]:
    with TestClient(create_app(settings, llm)) as client:  # type: ignore[arg-type]
        yield client


@pytest.fixture
def incident(labelled: list[LabelledIncident]) -> LabelledIncident:
    return next(item for item in labelled if item.label.fault_id == "F04")


def test_health_reports_ok_when_the_database_answers(
    settings: Settings, loaded_database: str
) -> None:
    with serving(settings, ScriptedLLM()) as client:
        assert client.get("/healthz").json() == {
            "status": "ok",
            "llm": "anthropic",
            "model": "claude-haiku-5-5",
        }


def test_search_returns_ranked_documents(settings: Settings, loaded_database: str) -> None:
    with serving(settings, ScriptedLLM()) as client:
        reply = client.post(
            "/v1/search",
            json={"query": "PGW-4012 authorisation declined", "mode": "fulltext", "limit": 3},
        )

        assert reply.status_code == 200
        hits = reply.json()
        assert [hit["rank"] for hit in hits] == [1, 2, 3]
        assert all("PGW-4012" in hit["body"] for hit in hits)


def test_a_triage_returns_the_answer_the_evidence_and_what_it_cost(
    settings: Settings, loaded_database: str, incident: LabelledIncident
) -> None:
    gold = incident.label.doc_ids[2]
    quote = "Meridian Acquiring limits our merchant account to 300 authorisations per second"
    llm = ScriptedLLM(response(draft((gold, quote)), model="claude-haiku-5-5"))
    with serving(settings, llm) as client:
        reply = client.post("/v1/triage", json=incident.incident.model_dump())

        assert reply.status_code == 200
        body = reply.json()
        assert body["triage"]["decision"] == "triage"
        assert body["triage"]["cause_category"] == "rate_limiting"
        assert body["triage"]["citations"] == [{"doc_id": gold, "quote": quote}]
        assert [item["rank"] for item in body["evidence"]] == [1, 2, 3, 4, 5]
        assert gold in [item["id"] for item in body["evidence"]]
        assert "body" not in body["evidence"][0]
        assert body["outcome"] == "answered"
        assert body["model_calls"] == 1
        assert body["cost_usd"] == pytest.approx(
            cost_usd("claude-haiku-5-5", response(None).usage), abs=1e-6
        )
        # The prompt was built from what the search returned through the MCP tools.
        assert f'<document id="{gold}"' in llm.requests[0].user


def test_an_abstention_is_a_normal_answer(
    settings: Settings, loaded_database: str, incident: LabelledIncident
) -> None:
    with serving(settings, ScriptedLLM(response(abstention(), model="claude-haiku-5-5"))) as client:
        body = client.post("/v1/triage", json=incident.incident.model_dump()).json()

        assert body["triage"]["decision"] == "insufficient_evidence"
        assert body["triage"]["citations"] == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"alert": "  ", "ticket_text": ""},
        {"alert": "x", "severity": "high"},
        {"alert": "x" * (MAX_FIELD_CHARS + 1)},
        {"alert": 7},
    ],
)
def test_a_request_that_is_empty_oversized_or_misshapen_is_rejected_without_a_model_call(
    settings: Settings, loaded_database: str, body: dict[str, object]
) -> None:
    llm = ScriptedLLM()
    with serving(settings, llm) as client:
        assert client.post("/v1/triage", json=body).status_code == 422
        assert llm.requests == []


def test_a_failure_of_the_model_api_is_a_bad_gateway_and_says_why(
    settings: Settings, loaded_database: str, incident: LabelledIncident
) -> None:
    class Unreachable:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise LLMError("no API credentials: set ANTHROPIC_API_KEY")

    with serving(settings, Unreachable()) as client:
        reply = client.post("/v1/triage", json=incident.incident.model_dump())

        assert reply.status_code == 502
        assert reply.json() == {"detail": "no API credentials: set ANTHROPIC_API_KEY"}


def test_in_replay_mode_an_incident_without_a_recording_is_unavailable_not_invented(
    settings: Settings, loaded_database: str, incident: LabelledIncident
) -> None:
    replay = settings.model_copy(
        update={"llm": "replay", "eval_dir": settings.data_dir / "nowhere"}
    )
    with TestClient(create_app(replay)) as client:
        recorded_nowhere = client.post("/v1/triage", json=incident.incident.model_dump())
        never_embedded = client.post("/v1/triage", json={"ticket_text": "Checkout is slow today."})

    assert recorded_nowhere.status_code == 503
    assert "replay mode answers only the recorded prompts" in recorded_nowhere.json()["detail"]
    # Replay searches with the committed vectors alone, and this text has none.
    assert never_embedded.status_code == 503
    assert "not in the embedding snapshot" in never_embedded.json()["detail"]
