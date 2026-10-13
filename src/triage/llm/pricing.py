"""List prices of the Claude API in US dollars per million tokens, as of 2026-10-06.

Cost is always computed from the token counts that a response reported and the model that
served it. Nothing in this repository estimates tokens from text length.
"""

from dataclasses import dataclass

from triage.llm.base import Usage

MILLION = 1_000_000

# A cache write costs 1.25 times the input price. Nothing here caches yet, but a response
# reports the fields, and a cost that ignored them would be wrong on the day something does.
CACHE_WRITE_MULTIPLIER = 1.25

# The Batch API halves every token price.
BATCH_MULTIPLIER = 0.5


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float


PRICES: dict[str, Price] = {
    # Up to 100,000 prompt tokens; a longer prompt is billed at a higher rate.
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20),
}

HAIKU_STANDARD_PROMPT_LIMIT = 100_000


def cost_usd(model: str, usage: Usage, *, batch: bool = False) -> float:
    price = PRICES.get(model)
    if price is None:
        raise KeyError(f"no price is recorded for the model {model!r}")
    prompt_tokens = (
        usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    )
    if model == "claude-haiku-5-5" and prompt_tokens > HAIKU_STANDARD_PROMPT_LIMIT:
        raise ValueError("a prompt above 100,000 tokens is billed at a rate that is not recorded")
    dollars = (
        usage.input_tokens * price.input
        + usage.cache_creation_input_tokens * price.input * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * price.cache_read
        + usage.output_tokens * price.output
    ) / MILLION
    return dollars * (BATCH_MULTIPLIER if batch else 1.0)
