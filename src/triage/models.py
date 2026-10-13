"""The data that crosses a boundary: the API, the model, the MCP tools and the data files."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from triage.taxonomy import Category

DocumentKind = Literal["incident", "runbook"]
Decision = Literal["triage", "insufficient_evidence"]
SearchMode = Literal["vector", "fulltext", "hybrid"]


class Incident(BaseModel):
    """What the agent is given. Any of the three fields may be empty, but not all of them."""

    model_config = ConfigDict(extra="forbid")

    alert: str = ""
    log_excerpt: str = ""
    ticket_text: str = ""


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: DocumentKind
    service: str
    title: str
    body: str

    @property
    def text(self) -> str:
        """The text that is embedded, indexed and quoted from."""
        return f"{self.title}\n{self.body}"


class Evidence(Document):
    """A document as a search returned it. `rank` starts at 1."""

    rank: int
    score: float


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="The id of an evidence document, exactly as given.")
    quote: str = Field(
        description="A passage copied word for word from that document that supports the cause."
    )


class TriageDraft(BaseModel):
    """What the model is asked to return. `Triage` is this after the citations were checked."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    cause_category: Category
    likely_cause: str = Field(description="One or two sentences. Empty when there is no cause.")
    citations: list[Citation]
    next_step: str = Field(description="The next diagnostic step, one sentence.")


class Triage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    cause_category: Category
    likely_cause: str
    citations: list[Citation]
    next_step: str
    evidence_ids: list[str] = Field(description="The documents the model was shown, in rank order.")
    abstained_because: str | None = Field(
        default=None,
        description="Set when the agent, not the model, turned the answer into an abstention.",
    )


class Label(BaseModel):
    """The answer key of one incident. Kept in a file the agent never reads."""

    model_config = ConfigDict(extra="forbid")

    id: str
    fault_id: str
    answerable: bool
    category: Category
    doc_ids: list[str]
    presentation: Literal["signature", "paraphrase", "downstream"]
    transparency: Literal["transparent", "opaque"] | None
    service: str
    background_fault: str | None = Field(
        default=None,
        description="The fault whose log line was mixed in as unrelated background, if any.",
    )


class LabelledIncident(BaseModel):
    id: str
    incident: Incident
    label: Label
