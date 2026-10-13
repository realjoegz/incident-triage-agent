"""What the agent needs from a model: one structured answer to one prompt.

The interface is this narrow so that a provider is one class. `triage.llm.claude` is the only
real one; the agent, the evaluation and the tests depend on nothing in it.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

Effort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class LLMRequest:
    model: str
    effort: Effort
    max_tokens: int
    system: str
    user: str
    schema: type[BaseModel]

    def fingerprint(self) -> str:
        """A hash of everything that is sent. A recorded response is found by it.

        Any change to a prompt, to the retrieved evidence inside it, to the output schema or to
        a model setting gives a different hash, and the recording no longer matches.
        """
        payload = {
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "user": self.user,
            "schema": self.schema.model_json_schema(),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The JSON document the model returned, or None when it returned no text at all.
    text: str | None
    stop_reason: str
    # The model that served the request, as the response names it.
    model: str
    usage: Usage
    # Of the attempt that succeeded. Waits between attempts are not in it.
    latency_s: float
    attempts: int
    request_id: str | None = None
    batch: bool = False


class LLMError(Exception):
    """The request did not produce a response: the API refused it or stayed unreachable."""


class LLM(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
