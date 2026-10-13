import pytest

from triage.models import Incident, LabelledIncident, SearchMode
from triage.retrieval import CANDIDATES, RRF_K, Retriever, fuse, incident_query


def test_the_query_keeps_log_messages_once_and_drops_what_precedes_them() -> None:
    incident = Incident(
        alert="payments-gateway: decline rate above 8%",
        ticket_text="Cards are declined in waves.",
        log_excerpt=(
            "2026-10-01T09:50:35Z INFO payments-gateway-1 healthcheck ok\n"
            "2026-10-01T09:50:45Z ERROR payments-gateway-2 PGW-4012 declined code=R29\n"
            "2026-10-01T09:50:59Z ERROR payments-gateway-4 PGW-4012 declined code=R29\n"
            "a line without the usual prefix\n"
        ),
    )

    assert incident_query(incident).splitlines() == [
        "payments-gateway: decline rate above 8%",
        "Cards are declined in waves.",
        "healthcheck ok",
        "PGW-4012 declined code=R29",
        "a line without the usual prefix",
    ]


def test_an_incident_with_only_a_ticket_is_searched_with_the_ticket() -> None:
    assert incident_query(Incident(ticket_text="  Search finds nothing.  ")) == (
        "Search finds nothing."
    )


def test_fusion_adds_reciprocal_ranks_and_breaks_ties_by_id() -> None:
    fused = fuse([["b", "a", "c"], ["a", "b"]], limit=3)

    top = 1 / (RRF_K + 1) + 1 / (RRF_K + 2)
    assert fused == [("a", top), ("b", top), ("c", 1 / (RRF_K + 3))]


def test_fusion_keeps_a_document_that_only_one_ranking_has() -> None:
    assert [doc_id for doc_id, _ in fuse([["x"], ["y", "z"]], limit=10)] == ["x", "y", "z"]


@pytest.mark.parametrize("mode", ["vector", "fulltext", "hybrid"])
async def test_a_log_line_of_a_known_fault_finds_its_three_documents(
    live_retriever: Retriever, mode: SearchMode
) -> None:
    hits = await live_retriever.search(
        "PGW-4012 authorisation declined by upstream code=R29 route=acq-primary", mode, 3
    )

    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert {hit.service for hit in hits} == {"payments-gateway"}
    assert all("PGW-4012" in hit.body for hit in hits)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


@pytest.mark.parametrize("mode", ["vector", "fulltext", "hybrid"])
async def test_a_search_gives_the_same_ranking_every_time(
    retriever: Retriever, labelled: list[LabelledIncident], mode: SearchMode
) -> None:
    query = incident_query(labelled[0].incident)

    first = await retriever.search(query, mode, CANDIDATES)
    second = await retriever.search(query, mode, CANDIDATES)

    assert [hit.id for hit in first] == [hit.id for hit in second]
    assert len({hit.id for hit in first}) == len(first)


async def test_full_text_search_finds_a_ticket_worded_differently_from_the_document(
    live_retriever: Retriever,
) -> None:
    hits = await live_retriever.search(
        "nothing can be written to the ledger, disk space", "fulltext", 5
    )

    assert "LDG-1090" in hits[0].body


@pytest.mark.parametrize(
    "query",
    ["it's \\ broken ' | & ! ( ) : *", "''", "\\", "the and of", "!!!", "a:b & c | d <-> e"],
)
async def test_full_text_search_takes_any_text_as_data(
    live_retriever: Retriever, query: str
) -> None:
    # The query is turned into a tsquery inside PostgreSQL. None of these may be read as syntax.
    hits = await live_retriever.search(query, "fulltext", 5)

    assert isinstance(hits, list)


async def test_text_that_shares_no_word_with_the_corpus_has_no_full_text_hit(
    live_retriever: Retriever,
) -> None:
    assert await live_retriever.search("zzyzx qwfp", "fulltext", 5) == []
    # The vector search still ranks every document, and so does the fusion.
    assert len(await live_retriever.search("zzyzx qwfp", "hybrid", 5)) == 5


@pytest.mark.parametrize("mode", ["vector", "fulltext", "hybrid"])
async def test_an_empty_query_returns_nothing(retriever: Retriever, mode: SearchMode) -> None:
    assert await retriever.search("   ", mode, 5) == []


async def test_a_document_is_fetched_by_id(retriever: Retriever) -> None:
    document = await retriever.get("RB-edge-proxy-00")

    assert document is not None
    assert document.title == "edge-proxy: overview and escalation"
    assert await retriever.get("RB-nowhere-00") is None
