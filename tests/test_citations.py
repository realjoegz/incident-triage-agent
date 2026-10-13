from tests.fakes import DOC_A, DOC_B, QUOTE_A
from triage.agent.citations import MIN_QUOTE_CHARS, check
from triage.models import Citation


def problems(*citations: tuple[str, str]) -> list[str | None]:
    checked = check([Citation(doc_id=d, quote=q) for d, q in citations], [DOC_A, DOC_B])
    return [item.problem for item in checked]


def test_a_quote_copied_from_the_document_is_valid() -> None:
    assert problems(("INC-A", QUOTE_A)) == [None]


def test_the_title_of_a_document_can_be_quoted() -> None:
    assert problems(("INC-A", "Card authorisations declined in bursts")) == [None]


def test_whitespace_letter_case_and_curly_quotes_do_not_fail_a_quote() -> None:
    assert problems(
        ("INC-A", "meridian acquiring   limits our\nMERCHANT account"),
        (
            "RB-B",
            "Compare authorisations per second on "
            "\N{LEFT SINGLE QUOTATION MARK}Gateway traffic\N{RIGHT SINGLE QUOTATION MARK}",
        ),
    ) == [None, None]


def test_a_reworded_quote_is_not_valid() -> None:
    [problem] = problems(("INC-A", "Meridian limits the account to 300 authorisations"))

    assert problem == (
        'the quote "Meridian limits the account to 300 authorisations" does not appear in INC-A'
    )


def test_a_quote_from_another_document_is_not_valid_for_this_one() -> None:
    [problem] = problems(("RB-B", QUOTE_A))

    assert problem is not None
    assert problem.endswith("does not appear in RB-B")


def test_a_document_that_was_not_retrieved_cannot_be_cited() -> None:
    assert problems(("INC-Z", QUOTE_A)) == ["INC-Z is not one of the evidence documents"]


def test_a_quote_too_short_to_mean_anything_is_not_valid() -> None:
    short = "300 authorisations"
    assert len(short) < MIN_QUOTE_CHARS

    assert problems(("INC-A", short)) == [
        f"the quote from INC-A is shorter than {MIN_QUOTE_CHARS} characters"
    ]


def test_each_citation_is_judged_on_its_own() -> None:
    result = problems(("INC-A", QUOTE_A), ("INC-A", "this sentence is not in the document"))

    assert result[0] is None
    assert result[1] is not None
