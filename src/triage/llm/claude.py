"""The Claude API through the official SDK.

Three choices here differ from what an application would usually want, and all three are for
the sake of measurement.

The SDK's own retries are off and the loop below retries instead, so that the latency of a
response is that of the attempt that succeeded and the number of attempts is known.

There is no fallback model. The API can rerun a declined request on another model; a results
table whose row says Haiku must not contain answers from something else, so a refusal is
returned as a refusal and counted as one.

Nothing is cached. A cached prompt prefix would make the cost of a variant depend on what ran
just before it.
"""

import asyncio
import random
import time
from typing import Any

import anthropic

from triage.llm.base import LLMError, LLMRequest, LLMResponse, Usage

MAX_ATTEMPTS = 6
REQUEST_TIMEOUT_S = 120.0
BASE_DELAY_S = 1.0
MAX_DELAY_S = 30.0


def message_params(request: LLMRequest) -> dict[str, Any]:
    """The body of a Messages API call for a request. The Batch API takes the same body.

    The output schema goes in `output_config.format` and the caller validates the text itself,
    so that an answer cut off at max_tokens or declined comes back as a response with its stop
    reason instead of as a parsing exception.
    """
    return {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system": request.system,
        "messages": [{"role": "user", "content": request.user}],
        "output_config": {
            "effort": request.effort,
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(request.schema),
            },
        },
    }


class ClaudeLLM:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        # With no argument the SDK reads ANTHROPIC_API_KEY from the environment. It does not
        # need one to be constructed, so a process without credentials still starts.
        self._client = client or anthropic.AsyncAnthropic(max_retries=0, timeout=REQUEST_TIMEOUT_S)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                message = await self._client.messages.create(**message_params(request))
            except TypeError as error:
                # What the SDK raises before sending anything when it has no credentials.
                if "authentication method" not in str(error):
                    raise
                raise LLMError("no API credentials: set ANTHROPIC_API_KEY") from error
            except anthropic.RateLimitError as error:
                failure: Exception = error
                retry_after = _retry_after(error)
            except anthropic.APIStatusError as error:
                if error.status_code < 500:
                    raise LLMError(f"the API rejected the request: {error.message}") from error
                failure, retry_after = error, None
            except anthropic.APIConnectionError as error:
                failure, retry_after = error, None
            else:
                latency = time.perf_counter() - started
                return to_response(request, message, latency, attempt)
            if attempt == MAX_ATTEMPTS:
                raise LLMError(f"no response after {MAX_ATTEMPTS} attempts: {failure}") from failure
            backoff = min(BASE_DELAY_S * 2 ** (attempt - 1), MAX_DELAY_S) + random.uniform(0, 1)
            await asyncio.sleep(max(backoff, retry_after or 0.0))
        raise AssertionError("unreachable")


def _retry_after(error: anthropic.RateLimitError) -> float | None:
    value = error.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def to_response(
    request: LLMRequest,
    message: anthropic.types.Message,
    latency: float,
    attempts: int,
    *,
    batch: bool = False,
) -> LLMResponse:
    # An alias may resolve to a dated snapshot of the same model. Anything else served the
    # request in place of the model that was asked for, and its answer must not be scored.
    if not message.model.startswith(request.model):
        raise LLMError(f"asked for {request.model} and was served by {message.model}")
    text = next((block.text for block in message.content if block.type == "text"), None)
    return LLMResponse(
        text=text,
        stop_reason=message.stop_reason or "unknown",
        model=request.model,
        usage=Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
        ),
        latency_s=round(latency, 3),
        attempts=attempts,
        # Set on a message that is the response of a request, not on one inside a batch result.
        request_id=getattr(message, "_request_id", None),
        batch=batch,
    )
