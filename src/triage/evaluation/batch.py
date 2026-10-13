"""Judge requests through the Batch API, which costs half and answers within the hour.

Nothing waits on a judge verdict, so the full evaluation sends them as one batch. The results
land in the same recordings as direct calls would, marked as batch so that their cost is
computed at the batch price.
"""

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path

import anthropic

from triage.llm.base import LLMError, LLMRequest
from triage.llm.claude import message_params, to_response
from triage.llm.recording import Budget, Recordings

POLL_SECONDS = 20.0


async def fill_from_batch(
    client: anthropic.AsyncAnthropic,
    requests: Mapping[str, LLMRequest],
    recordings: Recordings,
    budget: Budget,
    pending_file: Path,
    log: Callable[[str], None] = print,
    poll_seconds: float = POLL_SECONDS,
) -> list[str]:
    """Records a response for every request that has none. Returns the ids that failed.

    `requests` is keyed by fingerprint, which is also the id of the request inside the batch.
    The id of a batch that was submitted is written to `pending_file` until its results are
    in, so that a run that dies while waiting picks the batch up again instead of paying twice.
    """
    missing = {key: request for key, request in requests.items() if recordings.get(key, 0) is None}
    if not missing:
        return []

    if pending_file.exists():
        batch_id = pending_file.read_text(encoding="utf-8").strip()
        log(f"judge: resuming batch {batch_id}")
    else:
        budget.check()
        batch = await client.messages.batches.create(
            requests=[
                {"custom_id": key, "params": message_params(request)}  # type: ignore[typeddict-item]
                for key, request in sorted(missing.items())
            ]
        )
        batch_id = batch.id
        pending_file.parent.mkdir(parents=True, exist_ok=True)
        pending_file.write_text(batch_id + "\n", encoding="utf-8")
        log(f"judge: submitted batch {batch_id} with {len(missing)} requests")

    while True:
        batch = await client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        log(f"judge: batch in progress, {counts.succeeded} done, {counts.processing} to go")
        await asyncio.sleep(poll_seconds)

    failed = []
    async for item in await client.messages.batches.results(batch_id):
        request = missing.get(item.custom_id)
        if request is None:
            continue
        if item.result.type != "succeeded":
            failed.append(f"{item.custom_id[:12]} {item.result.type}")
            continue
        try:
            response = to_response(request, item.result.message, 0.0, 1, batch=True)
        except LLMError as error:
            failed.append(f"{item.custom_id[:12]} {error}")
            continue
        budget.charge(response)
        recordings.put(item.custom_id, 0, response)
    pending_file.unlink(missing_ok=True)
    return failed
