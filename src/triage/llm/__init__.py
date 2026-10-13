"""The model behind an interface: the Claude API, recorded responses, or a test double."""

from triage.llm.base import LLM, LLMError, LLMRequest, LLMResponse, Usage

__all__ = ["LLM", "LLMError", "LLMRequest", "LLMResponse", "Usage"]
