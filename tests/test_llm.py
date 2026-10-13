import json
from pathlib import Path

import anthropic
import httpx2
import pytest

from tests.fakes import ScriptedLLM, abstention, response
from triage.llm.base import LLMError, LLMRequest, LLMResponse, Usage
from triage.llm.claude import MAX_ATTEMPTS, ClaudeLLM
from triage.llm.pricing import cost_usd
from triage.llm.recording import (
    Budget,
    BudgetExceeded,
    RecordingLLM,
    RecordingMiss,
    Recordings,
    ReplayAll,
    ReplayLLM,
)
from triage.models import TriageDraft

REQUEST = LLMRequest(
    model="claude-haiku-5-5",
    effort="low",
    max_tokens=8000,
    system="You triage incidents.",
    user="<incident>…</incident>",
    schema=TriageDraft,
)


# ----------------------------------------------------------------------------------------------
# Fingerprints


def test_the_same_request_has_the_same_fingerprint() -> None:
    again = LLMRequest(**{**REQUEST.__dict__})

    assert again.fingerprint() == REQUEST.fingerprint()


@pytest.mark.parametrize(
    "change",
    [
        {"model": "claude-sonnet-5-5"},
        {"effort": "medium"},
        {"max_tokens": 4000},
        {"system": "You triage incidents. "},
        {"user": "<incident>...</incident>"},
        {"schema": Usage},
    ],
)
def test_any_change_to_what_is_sent_changes_the_fingerprint(change: dict[str, object]) -> None:
    changed = LLMRequest(**{**REQUEST.__dict__, **change})

    assert changed.fingerprint() != REQUEST.fingerprint()


# ----------------------------------------------------------------------------------------------
# Prices


def test_cost_is_tokens_times_the_price_of_the_model() -> None:
    usage = Usage(input_tokens=3_000, output_tokens=1_000)

    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(0.0003 + 0.0005)
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(0.006 + 0.010)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(0.012 + 0.020)


def test_a_batch_costs_half() -> None:
    usage = Usage(input_tokens=3_000, output_tokens=1_000)

    assert cost_usd("claude-opus-5-5", usage, batch=True) == pytest.approx(0.016)


def test_cached_tokens_are_priced_at_their_own_rates() -> None:
    usage = Usage(
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )

    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(2.50 + 0.20)


def test_a_model_without_a_recorded_price_has_no_cost() -> None:
    with pytest.raises(KeyError, match="no price is recorded"):
        cost_usd("claude-next", Usage(input_tokens=1, output_tokens=1))


def test_a_haiku_prompt_above_the_standard_tier_is_not_priced_at_the_standard_rate() -> None:
    with pytest.raises(ValueError, match="100,000 tokens"):
        cost_usd("claude-haiku-5-5", Usage(input_tokens=100_001, output_tokens=1))


# ----------------------------------------------------------------------------------------------
# Recordings


def test_a_recording_is_read_back_from_its_file(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    recorded = response(abstention())
    Recordings(path).put("abc", 2, recorded)

    reopened = Recordings(path)

    assert reopened.get("abc", 2) == recorded
    assert reopened.get("abc", 0) is None
    assert reopened.get("abd", 2) is None
    assert len(reopened) == 1


async def test_a_request_is_paid_for_once_and_then_answered_from_the_recording(
    tmp_path: Path,
) -> None:
    recordings = Recordings(tmp_path / "r.jsonl")
    inner = ScriptedLLM(response(abstention(), model="claude-haiku-5-5"))
    budget = Budget(max_usd=1.0)
    llm = RecordingLLM(inner, recordings, budget)

    first = await llm.complete(REQUEST)
    second = await llm.complete(REQUEST)

    assert first == second
    assert len(inner.requests) == 1
    assert budget.paid_calls == 1
    assert budget.spent_usd == pytest.approx(cost_usd("claude-haiku-5-5", first.usage))
    assert llm.used == {(REQUEST.fingerprint(), 0)}


async def test_each_repetition_of_a_request_is_recorded_separately(tmp_path: Path) -> None:
    recordings = Recordings(tmp_path / "r.jsonl")
    inner = ScriptedLLM(
        response(abstention(), model="claude-haiku-5-5"),
        response("second answer", model="claude-haiku-5-5"),
    )
    budget = Budget(max_usd=1.0)

    await RecordingLLM(inner, recordings, budget, rep=0).complete(REQUEST)
    await RecordingLLM(inner, recordings, budget, rep=1).complete(REQUEST)

    key = REQUEST.fingerprint()
    assert recordings.get(key, 0) != recordings.get(key, 1)
    assert (await ReplayLLM(recordings, rep=1).complete(REQUEST)).text == "second answer"


async def test_no_call_is_made_once_the_cap_is_reached(tmp_path: Path) -> None:
    recordings = Recordings(tmp_path / "r.jsonl")
    inner = ScriptedLLM(response(abstention(), model="claude-sonnet-5-5"))
    budget = Budget(max_usd=0.001)
    llm = RecordingLLM(inner, recordings, budget)
    await llm.complete(REQUEST)
    other = LLMRequest(**{**REQUEST.__dict__, "user": "another incident"})

    with pytest.raises(BudgetExceeded, match=r"\$0\.00 cap"):
        await llm.complete(other)

    assert len(inner.requests) == 1
    # What is already recorded costs nothing and is still answered.
    assert (await llm.complete(REQUEST)).text is not None


async def test_replay_never_calls_a_model_and_names_the_request_it_lacks(tmp_path: Path) -> None:
    replay = ReplayLLM(Recordings(tmp_path / "r.jsonl"))

    with pytest.raises(RecordingMiss, match=REQUEST.fingerprint()[:12]):
        await replay.complete(REQUEST)


async def test_every_file_of_a_directory_is_one_replay_source(tmp_path: Path) -> None:
    other = LLMRequest(**{**REQUEST.__dict__, "model": "claude-sonnet-5-5"})
    Recordings(tmp_path / "a.jsonl").put(REQUEST.fingerprint(), 0, response("from a"))
    Recordings(tmp_path / "b.jsonl").put(other.fingerprint(), 0, response("from b"))
    replay = ReplayAll(tmp_path)

    assert (await replay.complete(REQUEST)).text == "from a"
    assert (await replay.complete(other)).text == "from b"
    with pytest.raises(RecordingMiss, match="not one of them"):
        await replay.complete(LLMRequest(**{**REQUEST.__dict__, "user": "something new"}))


def test_compacting_orders_the_file_and_pruning_drops_what_a_run_did_not_use(
    tmp_path: Path,
) -> None:
    path = tmp_path / "r.jsonl"
    recordings = Recordings(path)
    for key, rep in (("b", 1), ("a", 0), ("b", 0)):
        recordings.put(key, rep, response(f"{key}{rep}"))

    assert recordings.prune(keep={("a", 0), ("b", 1)}) == 1
    recordings.compact()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(row["key"], row["rep"]) for row in rows] == [("a", 0), ("b", 1)]
    assert Recordings(path).get("b", 0) is None


# ----------------------------------------------------------------------------------------------
# The Claude API client, against a transport that answers in place of the API


def api_message(**overrides: object) -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": abstention().model_dump_json()},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1200, "output_tokens": 300},
        **overrides,
    }


class FakeAPI:
    def __init__(self, *replies: httpx2.Response) -> None:
        self._replies = list(replies)
        self.requests: list[dict[str, object]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]

    def llm(self) -> ClaudeLLM:
        return ClaudeLLM(
            anthropic.AsyncAnthropic(
                api_key="test-key",
                max_retries=0,
                http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self)),
            )
        )


def ok(**overrides: object) -> httpx2.Response:
    return httpx2.Response(200, json=api_message(**overrides), headers={"request-id": "req_1"})


def failure(status: int, kind: str, **headers: str) -> httpx2.Response:
    body = {"type": "error", "error": {"type": kind, "message": f"{kind} from the test"}}
    return httpx2.Response(status, json=body, headers=headers)


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replaces the sleep between attempts and collects how long each would have been."""
    waited: list[float] = []

    async def sleep(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr("triage.llm.claude.asyncio.sleep", sleep)
    return waited


async def test_the_request_carries_the_prompt_the_effort_and_the_output_schema() -> None:
    api = FakeAPI(ok())

    result = await api.llm().complete(REQUEST)

    [sent] = api.requests
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["max_tokens"] == 8000
    assert sent["system"] == "You triage incidents."
    assert sent["messages"] == [{"role": "user", "content": "<incident>…</incident>"}]
    config = sent["output_config"]
    assert isinstance(config, dict)
    assert config["effort"] == "low"
    assert config["format"]["type"] == "json_schema"
    assert config["format"]["schema"]["required"] == [
        "decision",
        "cause_category",
        "likely_cause",
        "citations",
        "next_step",
    ]
    # Nothing that would make the cost or the answers of one variant depend on another.
    assert "temperature" not in sent
    assert "fallbacks" not in sent
    assert "cache_control" not in json.dumps(sent)

    assert TriageDraft.model_validate_json(result.text or "") == abstention()
    assert result.usage == Usage(input_tokens=1200, output_tokens=300)
    assert (result.model, result.stop_reason, result.attempts) == (
        "claude-haiku-5-5",
        "end_turn",
        1,
    )
    assert result.request_id == "req_1"
    assert result.latency_s >= 0


async def test_a_rate_limit_is_retried_after_the_time_the_api_asked_for(waits: list[float]) -> None:
    api = FakeAPI(failure(429, "rate_limit_error", **{"retry-after": "7"}), ok())

    result = await api.llm().complete(REQUEST)

    assert result.attempts == 2
    assert len(api.requests) == 2
    assert len(waits) == 1
    assert waits[0] >= 7


async def test_a_server_error_is_retried_with_a_growing_wait(waits: list[float]) -> None:
    api = FakeAPI(failure(529, "overloaded_error"), failure(500, "api_error"), ok())

    result = await api.llm().complete(REQUEST)

    assert result.attempts == 3
    assert 1 <= waits[0] < 2
    assert 2 <= waits[1] < 3


async def test_after_the_last_attempt_the_failure_is_an_error_and_not_an_answer(
    waits: list[float],
) -> None:
    api = FakeAPI(failure(500, "api_error"))

    with pytest.raises(LLMError, match=f"no response after {MAX_ATTEMPTS} attempts"):
        await api.llm().complete(REQUEST)

    assert len(api.requests) == MAX_ATTEMPTS
    assert len(waits) == MAX_ATTEMPTS - 1


async def test_a_request_the_api_rejects_is_not_sent_again(waits: list[float]) -> None:
    api = FakeAPI(failure(400, "invalid_request_error"))

    with pytest.raises(LLMError, match="the API rejected the request"):
        await api.llm().complete(REQUEST)

    assert len(api.requests) == 1
    assert waits == []


async def test_an_answer_from_another_model_is_refused() -> None:
    api = FakeAPI(ok(model="claude-opus-5-5"))

    with pytest.raises(LLMError, match="asked for claude-haiku-5-5 and was served by claude-opus"):
        await api.llm().complete(REQUEST)


async def test_a_dated_snapshot_of_the_model_that_was_asked_for_is_accepted() -> None:
    api = FakeAPI(ok(model="claude-haiku-5-5-20260901"))

    assert (await api.llm().complete(REQUEST)).model == "claude-haiku-5-5"


async def test_a_refusal_comes_back_as_a_response_with_its_stop_reason() -> None:
    api = FakeAPI(ok(content=[], stop_reason="refusal"))

    result = await api.llm().complete(REQUEST)

    assert (result.text, result.stop_reason) == (None, "refusal")
    assert isinstance(result, LLMResponse)


async def test_without_credentials_the_call_fails_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(LLMError, match="no API credentials"):
        await ClaudeLLM().complete(REQUEST)
