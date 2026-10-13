"""Recorded model responses: written once by a paid run, read back for nothing.

A response is stored under the fingerprint of its request and a repetition number, because the
evaluation sends the same request several times to see how much the answers vary.

Two things use the store. A paid run that is interrupted finds the responses it already has and
pays only for the rest. CI, which has no API key, replays a whole run: the agent builds each
prompt again from the current code and data, and a prompt whose fingerprint has no recording
means that the committed results no longer describe the code.
"""

import asyncio
import json
import os
from pathlib import Path

from triage.llm.base import LLM, LLMRequest, LLMResponse
from triage.llm.pricing import cost_usd


class RecordingMiss(LookupError):
    """A request has no recorded response."""


class BudgetExceeded(RuntimeError):
    """The next paid call would be made after the spending cap was reached."""


class Recordings:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._responses: dict[tuple[str, int], LLMResponse] = {}
        if path.exists():
            with path.open(encoding="utf-8") as lines:
                for line in lines:
                    row = json.loads(line)
                    self._responses[row["key"], row["rep"]] = LLMResponse.model_validate(
                        row["response"]
                    )

    def __len__(self) -> int:
        return len(self._responses)

    def get(self, key: str, rep: int) -> LLMResponse | None:
        return self._responses.get((key, rep))

    def put(self, key: str, rep: int, response: LLMResponse) -> None:
        """Keeps the response and appends it to the file at once, so a crash loses nothing."""
        self._responses[key, rep] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(self._line(key, rep, response) + "\n")

    def compact(self) -> None:
        """Rewrites the file in a fixed order, so that two runs of the same work diff cleanly.

        The new content goes to a second file that then replaces the first, so a process that
        dies half way leaves the old file whole.
        """
        lines = [
            self._line(key, rep, response)
            for (key, rep), response in sorted(self._responses.items())
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.path.with_name(self.path.name + ".tmp")
        scratch.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        os.replace(scratch, self.path)

    def responses(self) -> list[LLMResponse]:
        return list(self._responses.values())

    def prune(self, keep: set[tuple[str, int]]) -> int:
        """Drops every response that the given run did not use. Returns how many went."""
        dropped = [key for key in self._responses if key not in keep]
        for key in dropped:
            del self._responses[key]
        return len(dropped)

    @staticmethod
    def _line(key: str, rep: int, response: LLMResponse) -> str:
        return json.dumps(
            {"key": key, "rep": rep, "response": response.model_dump(mode="json")},
            ensure_ascii=False,
        )


class Budget:
    """A cap on what one run may spend on calls it has to pay for."""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self.paid_calls = 0

    def check(self) -> None:
        if self.spent_usd >= self.max_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} of a ${self.max_usd:.2f} cap; no further call is made"
            )

    def charge(self, response: LLMResponse) -> None:
        self.spent_usd += cost_usd(response.model, response.usage, batch=response.batch)
        self.paid_calls += 1


class ReplayLLM:
    """Answers from the recordings and never calls a model."""

    def __init__(self, recordings: Recordings, rep: int = 0) -> None:
        self._recordings = recordings
        self._rep = rep
        self.used: set[tuple[str, int]] = set()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.fingerprint()
        response = self._recordings.get(key, self._rep)
        if response is None:
            raise RecordingMiss(
                f"no recorded response for request {key[:12]} (repetition {self._rep})"
            )
        self.used.add((key, self._rep))
        return response


class RecordingLLM:
    """Answers from the recordings when it can, and otherwise pays for a call and records it."""

    def __init__(self, inner: LLM, recordings: Recordings, budget: Budget, rep: int = 0) -> None:
        self._inner = inner
        self._recordings = recordings
        self._budget = budget
        self._rep = rep
        self._in_flight: dict[str, asyncio.Future[LLMResponse]] = {}
        self.used: set[tuple[str, int]] = set()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.fingerprint()
        self.used.add((key, self._rep))
        recorded = self._recordings.get(key, self._rep)
        if recorded is not None:
            return recorded
        # Two callers that ask for the same request at the same moment share one paid call.
        # Without this each would pay, and only the answer recorded last would be replayed.
        call = self._in_flight.get(key)
        if call is None:
            call = asyncio.ensure_future(self._pay(key, request))
            self._in_flight[key] = call
            call.add_done_callback(lambda _: self._in_flight.pop(key, None))
        return await asyncio.shield(call)

    async def _pay(self, key: str, request: LLMRequest) -> LLMResponse:
        self._budget.check()
        response = await self._inner.complete(request)
        # Recorded before it is charged: once it is in the file, nothing can lose it.
        self._recordings.put(key, self._rep, response)
        self._budget.charge(response)
        return response


class ReplayAll:
    """Every recording file of a directory, read as one replay source."""

    def __init__(self, directory: Path) -> None:
        self._sources = [ReplayLLM(Recordings(path)) for path in sorted(directory.glob("*.jsonl"))]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        for source in self._sources:
            try:
                return await source.complete(request)
            except RecordingMiss:
                continue
        raise RecordingMiss(
            "replay mode answers only the recorded prompts of the evaluation, "
            "and this request is not one of them"
        )
