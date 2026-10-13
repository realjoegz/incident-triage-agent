"""Judge requests through the Batch API, which costs half and answers within the hour.

Nothing waits on a judge verdict, so the full evaluation sends them as one batch. The results
land in the same recordings as direct calls would, marked as batch so that their cost is
computed at the batch price.

A batch is paid for once it is accepted, whatever happens to the process that sent it. So the
id of a batch is written to a file with the fingerprints it holds before anything waits on it,
and a later run reads that file first: it collects every result of that batch, including those
that it did not ask for itself, before it considers sending another.
"""

import asyncio
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import anthropic

from triage.llm.base import LLMError, LLMRequest, Usage
from triage.llm.claude import message_params, to_response
from triage.llm.pricing import cost_usd
from triage.llm.recording import Budget, BudgetExceeded, Recordings

POLL_SECONDS = 20.0

# How an unsent batch is priced to hold it against the cap. With enough recorded responses of
# the same model, their mean cost with this margin on top. Otherwise a bound from the length of
# the prompt: English text runs to about four characters per token, so three overstates it.
ESTIMATE_MARGIN = 1.5
ESTIMATE_MIN_SAMPLES = 10
ESTIMATE_CHARS_PER_TOKEN = 3
ESTIMATE_OUTPUT_TOKENS = 800


@dataclass(frozen=True)
class Pending:
    batch_id: str
    # The model that each fingerprint in the batch was asked of.
    models: dict[str, str]

    @classmethod
    def read(cls, path: Path) -> "Pending | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(batch_id=data["batch_id"], models=data["models"])

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({"batch_id": self.batch_id, "models": self.models}, indent=1)
        path.write_text(content + "\n", encoding="utf-8")


def estimate_usd(requests: Collection[LLMRequest], recordings: Recordings) -> float:
    """What a batch of these requests is expected to cost at most, before it is sent.

    This number decides only whether the batch fits under the cap. No reported cost uses it.
    """
    total = 0.0
    for request in requests:
        recorded = [
            cost_usd(response.model, response.usage, batch=True)
            for response in recordings.responses()
            if response.model == request.model
        ]
        if len(recorded) >= ESTIMATE_MIN_SAMPLES:
            total += ESTIMATE_MARGIN * sum(recorded) / len(recorded)
        else:
            prompt_chars = len(request.system) + len(request.user)
            bound = Usage(
                input_tokens=prompt_chars // ESTIMATE_CHARS_PER_TOKEN,
                output_tokens=ESTIMATE_OUTPUT_TOKENS,
            )
            total += cost_usd(request.model, bound, batch=True)
    return total


async def fill_from_batch(
    client: anthropic.AsyncAnthropic,
    requests: Mapping[str, LLMRequest],
    recordings: Recordings,
    budget: Budget,
    pending_file: Path,
    log: Callable[[str], None] = print,
    poll_seconds: float = POLL_SECONDS,
) -> list[str]:
    """Records a response for every request that has none. Returns what got no response.

    `requests` is keyed by fingerprint, which is also the id of the request inside the batch.
    `client` must not retry on its own: a batch that was accepted while the reply was lost
    would be created, and billed, a second time.
    """
    failed: list[str] = []
    tried: set[str] = set()
    while True:
        pending = Pending.read(pending_file)
        if pending is None:
            missing = {
                key: request
                for key, request in sorted(requests.items())
                if recordings.get(key, 0) is None and key not in tried
            }
            if not missing:
                return failed
            estimate = estimate_usd(missing.values(), recordings)
            if budget.spent_usd + estimate > budget.max_usd:
                raise BudgetExceeded(
                    f"a judge batch of {len(missing)} requests is estimated at ${estimate:.2f}, "
                    f"which with the ${budget.spent_usd:.2f} already paid passes the "
                    f"${budget.max_usd:.2f} cap; no batch is sent"
                )
            batch = await client.messages.batches.create(
                requests=[
                    {"custom_id": key, "params": message_params(request)}  # type: ignore[typeddict-item]
                    for key, request in missing.items()
                ]
            )
            pending = Pending(batch.id, {key: request.model for key, request in missing.items()})
            pending.write(pending_file)
            log(
                f"judge: submitted batch {batch.id} with {len(missing)} requests, "
                f"estimated at ${estimate:.2f} at most"
            )
        else:
            log(f"judge: picking up batch {pending.batch_id}, which an earlier run submitted")

        outcomes = await _collect(client, pending, recordings, budget, log, poll_seconds)
        pending_file.unlink(missing_ok=True)
        if outcomes is None:
            # The batch is gone, so nothing was learnt about its requests. They are still
            # missing and go into the next batch.
            continue
        tried |= set(pending.models)
        failed += [
            f"{key[:12]} {outcomes.get(key, 'absent from the results')}"
            for key in sorted(pending.models)
            if key in requests and recordings.get(key, 0) is None
        ]


async def _collect(
    client: anthropic.AsyncAnthropic,
    pending: Pending,
    recordings: Recordings,
    budget: Budget,
    log: Callable[[str], None],
    poll_seconds: float,
) -> dict[str, str] | None:
    """Waits for the batch and records each of its results. Returns why the others failed.

    None means that the API does not know the batch.
    """
    while True:
        try:
            batch = await client.messages.batches.retrieve(pending.batch_id)
        except anthropic.NotFoundError:
            log(f"judge: batch {pending.batch_id} no longer exists; its requests are sent again")
            return None
        except (anthropic.APIConnectionError, anthropic.InternalServerError) as error:
            log(f"judge: could not ask for the state of the batch ({error}); asking again")
            await asyncio.sleep(poll_seconds)
            continue
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        log(f"judge: batch in progress, {counts.succeeded} done, {counts.processing} to go")
        await asyncio.sleep(poll_seconds)

    outcomes: dict[str, str] = {}
    async for item in await client.messages.batches.results(pending.batch_id):
        model = pending.models.get(item.custom_id)
        if model is None or recordings.get(item.custom_id, 0) is not None:
            continue
        if item.result.type != "succeeded":
            outcomes[item.custom_id] = item.result.type
            continue
        try:
            response = to_response(model, item.result.message, 0.0, 1, batch=True)
        except LLMError as error:
            outcomes[item.custom_id] = str(error)
            continue
        # Recorded before it is charged: once it is in the file, nothing can lose it.
        recordings.put(item.custom_id, 0, response)
        budget.charge(response)
    return outcomes
