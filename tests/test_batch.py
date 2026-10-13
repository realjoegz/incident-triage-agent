import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from triage.evaluation.batch import Pending, estimate_usd, fill_from_batch
from triage.evaluation.rows import Verdict
from triage.llm.base import LLMRequest
from triage.llm.pricing import cost_usd
from triage.llm.recording import Budget, BudgetExceeded, Recordings

BATCH_ID = "msgbatch_test"
SECOND_ID = "msgbatch_second"


def request(user: str) -> LLMRequest:
    return LLMRequest("claude-opus-5-5", "low", 4000, "Judge.", user, Verdict)


def batch(status: str, batch_id: str = BATCH_ID) -> dict[str, object]:
    return {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": status,
        "request_counts": {
            "processing": 0,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": "2026-10-13T20:00:00Z",
        "expires_at": "2026-10-14T20:00:00Z",
        "ended_at": "2026-10-13T20:05:00Z" if status == "ended" else None,
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": (
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
            if status == "ended"
            else None
        ),
    }


def succeeded(custom_id: str, supported: bool) -> dict[str, object]:
    verdict = Verdict(reason="because", supported=supported).model_dump_json()
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": verdict}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 400, "output_tokens": 100},
            },
        },
    }


def errored(custom_id: str) -> dict[str, object]:
    error = {"type": "error", "error": {"type": "api_error", "message": "internal"}}
    return {"custom_id": custom_id, "result": {"type": "errored", "error": error}}


class FakeBatchAPI:
    """Answers the three calls of a batch: create, retrieve (twice), results.

    `results` are those of the first batch. With `pending=True` that batch exists before
    anything is created, as one that an earlier run submitted. A later batch answers every
    request it is sent with a supported verdict.
    """

    def __init__(self, results: list[dict[str, object]], *, pending: bool = False) -> None:
        self._first = results
        self._results = {BATCH_ID: results} if pending else {}
        self.created: list[dict[str, Any]] = []
        self.retrieved = 0

    def __call__(self, http: httpx2.Request) -> httpx2.Response:
        path = http.url.path
        if http.method == "POST" and path == "/v1/messages/batches":
            body = json.loads(http.content)
            self.created.append(body)
            answered = [succeeded(item["custom_id"], True) for item in body["requests"]]
            if BATCH_ID in self._results:
                batch_id = SECOND_ID
                self._results[batch_id] = answered
            else:
                batch_id = BATCH_ID
                self._results[batch_id] = self._first or answered
            return httpx2.Response(200, json=batch("in_progress", batch_id=batch_id))
        batch_id = path.removeprefix("/v1/messages/batches/").removesuffix("/results")
        if batch_id not in self._results:
            error = {"type": "error", "error": {"type": "not_found_error", "message": "gone"}}
            return httpx2.Response(404, json=error)
        if http.method == "GET" and path.endswith("/results"):
            lines = "".join(json.dumps(result) + "\n" for result in self._results[batch_id])
            return httpx2.Response(
                200, content=lines, headers={"content-type": "application/x-jsonl"}
            )
        self.retrieved += 1
        done = self.retrieved > 1
        return httpx2.Response(
            200, json=batch("ended" if done else "in_progress", batch_id=batch_id)
        )

    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self)),
        )


async def fill(
    api: FakeBatchAPI,
    requests: dict[str, LLMRequest],
    recordings: Recordings,
    budget: Budget,
    tmp_path: Path,
) -> list[str]:
    return await fill_from_batch(
        api.client(),
        requests,
        recordings,
        budget,
        tmp_path / ".pending",
        log=lambda _: None,
        poll_seconds=0,
    )


async def test_missing_verdicts_are_fetched_in_one_batch_and_recorded_at_half_price(
    tmp_path: Path,
) -> None:
    one, two, three = request("one"), request("two"), request("three")
    requests = {item.fingerprint(): item for item in (one, two, three)}
    recordings = Recordings(tmp_path / "judge.jsonl")
    already = succeeded(three.fingerprint(), True)["result"]["message"]  # type: ignore[index]
    recordings.put(three.fingerprint(), 0, _as_response(three, already))
    api = FakeBatchAPI([succeeded(two.fingerprint(), False), succeeded(one.fingerprint(), True)])
    budget = Budget(max_usd=1.0)

    failed = await fill(api, requests, recordings, budget, tmp_path)

    assert failed == []
    # Only the two without a recording were sent, each under its fingerprint.
    [created] = api.created
    sent = created["requests"]
    assert isinstance(sent, list)
    assert sorted(item["custom_id"] for item in sent) == sorted(
        [one.fingerprint(), two.fingerprint()]
    )
    params = sent[0]["params"]
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["type"] == "json_schema"

    first = recordings.get(one.fingerprint(), 0)
    second = recordings.get(two.fingerprint(), 0)
    assert first is not None
    assert second is not None
    assert Verdict.model_validate_json(first.text or "").supported is True
    assert Verdict.model_validate_json(second.text or "").supported is False
    assert first.batch is True
    assert budget.paid_calls == 2
    assert budget.spent_usd == pytest.approx(2 * cost_usd("claude-opus-5-5", first.usage) / 2)
    assert not (tmp_path / ".pending").exists()


def _as_response(item: LLMRequest, message: object):  # type: ignore[no-untyped-def]
    from triage.llm.claude import to_response

    return to_response(item.model, anthropic.types.Message.model_validate(message), 0.0, 1)


async def test_nothing_is_submitted_when_every_verdict_is_recorded(tmp_path: Path) -> None:
    one = request("one")
    recordings = Recordings(tmp_path / "judge.jsonl")
    message = succeeded(one.fingerprint(), True)["result"]["message"]  # type: ignore[index]
    recordings.put(one.fingerprint(), 0, _as_response(one, message))
    api = FakeBatchAPI([])

    failed = await fill(api, {one.fingerprint(): one}, recordings, Budget(0.0), tmp_path)

    assert failed == []
    assert api.created == []


async def test_no_batch_is_submitted_once_the_cap_is_reached(tmp_path: Path) -> None:
    one = request("one")
    api = FakeBatchAPI([succeeded(one.fingerprint(), True)])

    with pytest.raises(BudgetExceeded):
        await fill(
            api, {one.fingerprint(): one}, Recordings(tmp_path / "j.jsonl"), Budget(0.0), tmp_path
        )

    assert api.created == []


async def test_a_request_the_batch_failed_is_reported_and_left_unrecorded(tmp_path: Path) -> None:
    one, two = request("one"), request("two")
    recordings = Recordings(tmp_path / "judge.jsonl")
    api = FakeBatchAPI([succeeded(one.fingerprint(), True), errored(two.fingerprint())])

    failed = await fill(
        api, {item.fingerprint(): item for item in (one, two)}, recordings, Budget(1.0), tmp_path
    )

    assert failed == [f"{two.fingerprint()[:12]} errored"]
    assert recordings.get(one.fingerprint(), 0) is not None
    assert recordings.get(two.fingerprint(), 0) is None


async def test_a_batch_that_was_already_submitted_is_picked_up_and_not_submitted_again(
    tmp_path: Path,
) -> None:
    one = request("one")
    Pending(BATCH_ID, {one.fingerprint(): one.model}).write(tmp_path / ".pending")
    recordings = Recordings(tmp_path / "judge.jsonl")
    api = FakeBatchAPI([succeeded(one.fingerprint(), True)], pending=True)

    failed = await fill(api, {one.fingerprint(): one}, recordings, Budget(1.0), tmp_path)

    assert failed == []
    assert api.created == []
    assert recordings.get(one.fingerprint(), 0) is not None
    assert not (tmp_path / ".pending").exists()


async def test_a_batch_of_an_earlier_run_is_collected_in_full_before_another_is_sent(
    tmp_path: Path,
) -> None:
    """The earlier run asked for one, two and three. This run asks for three and four."""
    one, two, three, four = (request(name) for name in ("one", "two", "three", "four"))
    earlier = {item.fingerprint(): item.model for item in (one, two, three)}
    Pending(BATCH_ID, earlier).write(tmp_path / ".pending")
    recordings = Recordings(tmp_path / "judge.jsonl")
    api = FakeBatchAPI(
        [succeeded(item.fingerprint(), True) for item in (one, two, three)], pending=True
    )
    budget = Budget(max_usd=1.0)

    failed = await fill(
        api, {item.fingerprint(): item for item in (three, four)}, recordings, budget, tmp_path
    )

    assert failed == []
    # All three results of the paid batch are kept, also the two that this run did not ask for.
    assert all(recordings.get(item.fingerprint(), 0) is not None for item in (one, two, three))
    # What the earlier batch did not hold went into a batch of its own, not into direct calls.
    [second] = api.created
    assert [item["custom_id"] for item in second["requests"]] == [four.fingerprint()]
    assert recordings.get(four.fingerprint(), 0) is not None
    assert budget.paid_calls == 4
    assert not (tmp_path / ".pending").exists()


async def test_a_batch_that_no_longer_exists_is_forgotten_and_its_requests_are_sent_again(
    tmp_path: Path,
) -> None:
    one = request("one")
    Pending("msgbatch_gone", {one.fingerprint(): one.model}).write(tmp_path / ".pending")
    recordings = Recordings(tmp_path / "judge.jsonl")
    api = FakeBatchAPI([])

    failed = await fill(api, {one.fingerprint(): one}, recordings, Budget(1.0), tmp_path)

    assert failed == []
    assert len(api.created) == 1
    assert recordings.get(one.fingerprint(), 0) is not None


async def test_a_batch_that_would_pass_the_cap_is_not_sent(tmp_path: Path) -> None:
    requests = {item.fingerprint(): item for item in (request(f"number {n}") for n in range(200))}
    recordings = Recordings(tmp_path / "judge.jsonl")
    estimate = estimate_usd(list(requests.values()), recordings)
    api = FakeBatchAPI([])
    budget = Budget(max_usd=estimate - 0.01)

    with pytest.raises(BudgetExceeded, match="200 requests is estimated at"):
        await fill(api, requests, recordings, budget, tmp_path)

    assert api.created == []
    assert budget.spent_usd == 0.0


def test_the_estimate_of_a_batch_uses_what_such_verdicts_cost_before(tmp_path: Path) -> None:
    recordings = Recordings(tmp_path / "judge.jsonl")
    fresh = [request("a short prompt")] * 3
    by_length = estimate_usd(fresh, recordings)
    message = succeeded("x", True)["result"]["message"]  # type: ignore[index]
    for number in range(10):
        recordings.put(f"k{number}", 0, _as_response(fresh[0], message))
    recorded_cost = cost_usd("claude-opus-5-5", recordings.responses()[0].usage, batch=True)

    by_history = estimate_usd(fresh, recordings)

    # Without history: 800 output tokens each at the batch price, plus the prompt.
    assert by_length == pytest.approx(3 * (800 * 20 / 1e6 + 7 * 4 / 1e6) / 2, rel=0.01)
    # With history: the mean recorded cost, and half as much again.
    assert by_history == pytest.approx(3 * 1.5 * recorded_cost)
