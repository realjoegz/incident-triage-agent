"""The citation check, in code: a citation counts only if its quote is in its document.

This is the part of faithfulness that needs no model. It cannot tell whether a passage supports
the stated cause (the evaluation asks a judge that), but it does guarantee that every citation
in an answer points at a document the model was shown and at words that document contains.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from triage.models import Citation, Evidence

# Shorter than this and a "quote" can be a service name or a single common word.
MIN_QUOTE_CHARS = 20

_QUOTE_MARKS = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
    }
)


def normalise(text: str) -> str:
    """Whitespace, letter case and curly quotation marks are not differences worth failing on."""
    return " ".join(text.translate(_QUOTE_MARKS).split()).casefold()


@dataclass(frozen=True)
class Checked:
    citation: Citation
    problem: str | None

    @property
    def valid(self) -> bool:
        return self.problem is None


def check(citations: Sequence[Citation], evidence: Sequence[Evidence]) -> list[Checked]:
    documents = {item.id: normalise(item.text) for item in evidence}
    checked = []
    for citation in citations:
        quote = normalise(citation.quote)
        problem = None
        if citation.doc_id not in documents:
            problem = f"{citation.doc_id} is not one of the evidence documents"
        elif len(quote) < MIN_QUOTE_CHARS:
            problem = (
                f"the quote from {citation.doc_id} is shorter than {MIN_QUOTE_CHARS} characters"
            )
        elif quote not in documents[citation.doc_id]:
            problem = f'the quote "{citation.quote}" does not appear in {citation.doc_id}'
        checked.append(Checked(citation, problem))
    return checked
