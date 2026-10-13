"""The variants that the results table compares: one row each."""

from dataclasses import dataclass

from triage.agent import AgentConfig, AgentMode


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    mode: AgentMode
    model: str

    def config(self) -> AgentConfig:
        return AgentConfig(model=self.model, mode=self.mode)


HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"

# The first four differ only in retrieval. The last differs from the fourth only in the model.
VARIANTS: tuple[Variant, ...] = (
    Variant("none-haiku", "No retrieval, Haiku 5.5", "none", HAIKU),
    Variant("vector-haiku", "Vector, Haiku 5.5", "vector", HAIKU),
    Variant("fulltext-haiku", "Full text, Haiku 5.5", "fulltext", HAIKU),
    Variant("hybrid-haiku", "Hybrid, Haiku 5.5", "hybrid", HAIKU),
    Variant("hybrid-sonnet", "Hybrid, Sonnet 5.5", "hybrid", SONNET),
)

BY_NAME = {variant.name: variant for variant in VARIANTS}

# What the API serves by default, and the row that the others are read against.
DEFAULT = BY_NAME["hybrid-haiku"]
