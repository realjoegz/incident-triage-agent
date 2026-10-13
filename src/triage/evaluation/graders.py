"""The grading that needs no model: a row against the label of its incident.

An incident with evidence is answered correctly when the agent names the labelled category. An
incident without evidence is answered correctly when the agent says that the evidence is
insufficient. Both directions are graded, so neither "always answer" nor "always abstain" can
score well.
"""

from pydantic import BaseModel

from triage.evaluation.rows import Row
from triage.models import Label


class Grade(BaseModel):
    # False when the model gave nothing to grade: a refusal, a cut-off or an invalid output.
    scored: bool
    answerable: bool
    correct: bool
    abstained: bool
    # With evidence: the agent named another category than the labelled one.
    wrong_cause: bool
    # Without evidence: the agent named a cause anyway.
    false_attribution: bool
    # Of the three documents of the fault, the share among those retrieved. None when the
    # incident has no such documents or the variant does not retrieve.
    recall: float | None
    hit: bool | None
    reciprocal_rank: float | None
    # For an answer that names a cause on an incident with evidence: does any citation point
    # at one of the fault's documents? None when there is no such answer.
    cites_gold: bool | None
    # The labelled category, cited to a document of the fault. The category alone can be right
    # by way of another fault of the same category; this cannot. None without retrieval.
    correct_citing_the_fault: bool | None


def grade(row: Row, label: Label, *, retrieves: bool) -> Grade:
    named_a_cause = row.decision == "triage"
    gold = set(label.doc_ids)

    recall = hit = reciprocal_rank = None
    if retrieves and label.answerable:
        found = [doc_id for doc_id in row.retrieved if doc_id in gold]
        recall = len(found) / len(gold)
        hit = bool(found)
        first = next((rank for rank, doc_id in enumerate(row.retrieved, 1) if doc_id in gold), None)
        reciprocal_rank = 1.0 / first if first else 0.0

    cites_gold = None
    if retrieves and label.answerable and named_a_cause:
        cites_gold = any(citation.doc_id in gold for citation in row.citations)

    if label.answerable:
        correct = named_a_cause and row.category is label.category
        wrong_cause = named_a_cause and not correct
        false_attribution = False
    else:
        correct = not named_a_cause
        wrong_cause = False
        false_attribution = named_a_cause

    correct_citing_the_fault = None
    if retrieves and label.answerable:
        correct_citing_the_fault = correct and bool(cites_gold)

    return Grade(
        scored=row.outcome == "answered",
        answerable=label.answerable,
        correct=correct,
        abstained=not named_a_cause,
        wrong_cause=wrong_cause,
        false_attribution=false_attribution,
        recall=recall,
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        cites_gold=cites_gold,
        correct_citing_the_fault=correct_citing_the_fault,
    )
