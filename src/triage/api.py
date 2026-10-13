"""The HTTP API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from mcp import Client
from pydantic import BaseModel, ConfigDict, Field

from triage.agent import Agent, AgentConfig, Outcome
from triage.llm import LLM, LLMError
from triage.llm.claude import ClaudeLLM
from triage.llm.pricing import cost_usd
from triage.llm.recording import RecordingMiss, ReplayAll
from triage.mcp_server import MAX_LIMIT, EvidenceError, EvidenceTools, build_server
from triage.models import Evidence, Incident, SearchMode, Triage
from triage.runtime import open_retriever
from triage.settings import Settings

RECORDINGS_DIR = "recordings"

# An incident is sent to a model in full. This bounds what one request can cost.
MAX_FIELD_CHARS = 20_000

Field20k = Annotated[str, Field(max_length=MAX_FIELD_CHARS)]


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert: Field20k = ""
    log_excerpt: Field20k = ""
    ticket_text: Field20k = ""


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Field20k
    mode: SearchMode = "hybrid"
    limit: int = Field(default=5, ge=1, le=MAX_LIMIT)


class EvidenceSummary(BaseModel):
    id: str
    kind: str
    service: str
    title: str
    rank: int
    score: float


class TriageResponse(BaseModel):
    triage: Triage
    evidence: list[EvidenceSummary]
    outcome: Outcome
    model: str
    model_calls: int
    cost_usd: float
    seconds: float


def build_llm(settings: Settings) -> LLM:
    if settings.llm == "replay":
        return ReplayAll(settings.eval_dir / RECORDINGS_DIR)
    return ClaudeLLM()


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # A text that the snapshot holds gets its committed vector, so a labelled incident is
        # searched here exactly as the evaluation searched it and its recorded answer is found
        # in replay mode. Any other text goes to the embedding model.
        async with (
            open_retriever(settings, snapshot_only=False) as retriever,
            Client(build_server(retriever)) as client,
        ):
            tools = EvidenceTools(client)
            app.state.tools = tools
            app.state.agent = Agent(
                llm or build_llm(settings),
                tools.search,
                AgentConfig(
                    model=settings.model,
                    effort=settings.effort,
                    mode=settings.search_mode,
                    top_k=settings.top_k,
                ),
            )
            yield

    app = FastAPI(title="incident-triage-agent", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, str]:
        # A search that reaches the database through the MCP tools, as a triage would.
        await request.app.state.tools.search("health", "fulltext", 1)
        return {"status": "ok", "llm": settings.llm, "model": settings.model}

    @app.post("/v1/search")
    async def search(request: Request, body: SearchRequest) -> list[Evidence]:
        tools: EvidenceTools = request.app.state.tools
        try:
            return await tools.search(body.query, body.mode, body.limit)
        except EvidenceError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/v1/triage")
    async def triage(request: Request, body: TriageRequest) -> TriageResponse:
        agent: Agent = request.app.state.agent
        try:
            run = await agent.run(Incident(**body.model_dump()))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (RecordingMiss, EvidenceError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except LLMError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return TriageResponse(
            triage=run.triage,
            evidence=[
                EvidenceSummary(**item.model_dump(exclude={"body"})) for item in run.evidence
            ],
            outcome=run.outcome,
            model=agent.config.model,
            model_calls=len(run.calls),
            cost_usd=round(
                sum(cost_usd(call.response.model, call.response.usage) for call in run.calls), 6
            ),
            seconds=run.total_s,
        )

    return app
