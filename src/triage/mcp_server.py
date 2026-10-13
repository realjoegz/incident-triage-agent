"""The evidence tools as an MCP server.

The agent reaches the corpus through these two tools and through nothing else. Inside this
process it connects to the server object directly; `triage mcp` serves the same object over
stdio, so that any MCP client can search the same evidence with the same tools.
"""

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import TypeAdapter

from triage.embeddings import SnapshotMiss
from triage.models import Document, Evidence, SearchMode
from triage.retrieval import Retriever
from triage.runtime import open_retriever
from triage.settings import Settings

MAX_LIMIT = 20

_EVIDENCE_LIST = TypeAdapter(list[Evidence])


def build_server(retriever: Retriever) -> MCPServer:
    server = MCPServer(
        "incident-evidence",
        instructions=(
            "Past incident reports and runbook sections. search_evidence ranks them against a "
            "description of an incident; get_document returns one of them by id."
        ),
    )

    @server.tool()
    async def search_evidence(
        query: str, mode: SearchMode = "hybrid", limit: int = 5
    ) -> list[Evidence]:
        """Search past incidents and runbook sections for those most like the query.

        The query is free text: an alert, a ticket, log lines, or all three. The best matches
        are returned whether or not any of them is about the same failure.
        """
        try:
            return await retriever.search(query, mode, max(1, min(limit, MAX_LIMIT)))
        except SnapshotMiss as error:
            # Expected when the server runs on the committed vectors alone: say so to the
            # caller instead of reporting a crash.
            raise ToolError(str(error)) from error

    @server.tool()
    async def get_document(doc_id: str) -> Document:
        """Return one past incident or runbook section by its id."""
        document = await retriever.get(doc_id)
        if document is None:
            raise ToolError(f"there is no document with the id {doc_id}")
        return document

    return server


class EvidenceError(RuntimeError):
    """A tool call came back as an error."""


class EvidenceTools:
    """The client side: what the agent calls."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def search(self, query: str, mode: SearchMode, limit: int) -> list[Evidence]:
        result = await self._client.call_tool(
            "search_evidence", {"query": query, "mode": mode, "limit": limit}
        )
        if result.is_error or result.structured_content is None:
            texts = [block.text for block in result.content if block.type == "text"]
            raise EvidenceError("; ".join(texts) or "search_evidence returned no result")
        return _EVIDENCE_LIST.validate_python(result.structured_content["result"])


async def serve_stdio(settings: Settings) -> None:
    """Serves the tools to an MCP client on the other end of stdin and stdout."""
    async with open_retriever(settings, snapshot_only=False) as retriever:
        await build_server(retriever).run_stdio_async()
