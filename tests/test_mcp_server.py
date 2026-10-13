import os
import sys

import pytest
from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from triage.mcp_server import MAX_LIMIT, EvidenceTools, build_server
from triage.models import LabelledIncident
from triage.retrieval import Retriever, incident_query
from triage.settings import Settings


@pytest.fixture
def server(retriever: Retriever) -> MCPServer:
    return build_server(retriever)


async def test_the_server_offers_the_two_evidence_tools(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

        assert set(tools) == {"search_evidence", "get_document"}
        assert tools["search_evidence"].input_schema["required"] == ["query"]
        assert set(tools["search_evidence"].input_schema["properties"]) == {
            "query",
            "mode",
            "limit",
        }
        assert tools["search_evidence"].description


@pytest.mark.parametrize("mode", ["vector", "fulltext", "hybrid"])
async def test_a_search_through_mcp_is_the_search_of_the_retriever(
    server: MCPServer, retriever: Retriever, labelled: list[LabelledIncident], mode: str
) -> None:
    async with Client(server) as client:
        query = incident_query(labelled[3].incident)

        through_mcp = await EvidenceTools(client).search(query, mode, 5)  # type: ignore[arg-type]

        assert through_mcp == await retriever.search(query, mode, 5)  # type: ignore[arg-type]
        assert len(through_mcp) == 5


async def test_the_default_search_is_hybrid_with_five_results(
    server: MCPServer, retriever: Retriever, labelled: list[LabelledIncident]
) -> None:
    async with Client(server) as client:
        query = incident_query(labelled[0].incident)

        result = await client.call_tool("search_evidence", {"query": query})

        assert result.structured_content is not None
        ids = [row["id"] for row in result.structured_content["result"]]
        assert ids == [hit.id for hit in await retriever.search(query, "hybrid", 5)]


async def test_a_limit_out_of_range_is_brought_into_range(
    server: MCPServer, labelled: list[LabelledIncident]
) -> None:
    async with Client(server) as client:
        tools = EvidenceTools(client)
        query = incident_query(labelled[0].incident)

        assert len(await tools.search(query, "vector", 10_000)) == MAX_LIMIT
        assert len(await tools.search(query, "vector", 0)) == 1


async def test_a_document_is_returned_by_id(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("get_document", {"doc_id": "RB-edge-proxy-00"})

        assert not result.is_error
        assert result.structured_content is not None
        assert result.structured_content["title"] == "edge-proxy: overview and escalation"


async def test_an_unknown_id_is_an_error_that_names_the_id(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("get_document", {"doc_id": "RB-nowhere-00"})

        assert result.is_error
        assert "there is no document with the id RB-nowhere-00" in str(result.content)


async def test_an_unknown_search_mode_is_rejected(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await client.call_tool("search_evidence", {"query": "x", "mode": "fuzzy"})

        assert result.is_error


async def test_the_same_tools_are_served_over_stdio(
    settings: Settings, loaded_database: str, labelled: list[LabelledIncident]
) -> None:
    """Starts `triage mcp` as a separate process, the way an MCP client would."""
    process = StdioServerParameters(
        command=sys.executable,
        args=["-m", "triage.cli", "mcp"],
        env={
            **os.environ,
            "TRIAGE_DATABASE_URL": loaded_database,
            "TRIAGE_DATA_DIR": str(settings.data_dir),
        },
    )
    query = incident_query(labelled[0].incident)

    async with Client(process) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
        hits = await EvidenceTools(client).search(query, "hybrid", 3)

    assert names == {"search_evidence", "get_document"}
    assert [hit.rank for hit in hits] == [1, 2, 3]
