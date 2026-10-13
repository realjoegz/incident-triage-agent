"""The harness, run end to end with stand-in models whose right score is known in advance.

Before any number of a real model is believed, the pipeline has to give a model that knows the
answer key the score it deserves, and a model that knows nothing the score it deserves.
"""

from pathlib import Path
from typing import Any

import pytest

from tests.fakes import RuleLLM, always_abstains, always_rate_limiting, oracle
from triage.agent import prompts
from triage.dataset.catalogue import Catalogue, load_catalogue
from triage.evaluation import runner
from triage.evaluation.variants import BY_NAME, VARIANTS
from triage.llm.base import LLMError, LLMRequest, LLMResponse
from triage.llm.pricing import cost_usd
from triage.llm.recording import Budget, BudgetExceeded
from triage.models import LabelledIncident
from triage.settings import Settings
from triage.taxonomy import Category

RETRIEVING = ("vector-haiku", "fulltext-haiku", "hybrid-haiku", "hybrid-sonnet")


@pytest.fixture
def catalogue(settings: Settings) -> Catalogue:
    return load_catalogue(settings.data_dir)


@pytest.fixture
def scratch(settings: Settings, loaded_database: str, tmp_path: Path) -> Settings:
    """The real data and database, with recordings and results in a directory of the test."""
    return settings.model_copy(update={"eval_dir": tmp_path})


async def run_with(scratch: Settings, llm: RuleLLM, **options: Any) -> runner.Evaluation:
    defaults: dict[str, Any] = {"reps": 1, "budget": Budget(max_usd=1000.0), "log": lambda _: None}
    return await runner.evaluate(scratch, live=True, paid_llm=llm, **{**defaults, **options})


def variants_of(evaluation: runner.Evaluation) -> dict[str, dict[str, Any]]:
    return evaluation.summary["variants"]  # type: ignore[return-value]


async def test_a_model_that_knows_the_answers_scores_exactly_what_retrieval_lets_it(
    scratch: Settings, labelled: list[LabelledIncident], catalogue: Catalogue
) -> None:
    evaluation = await run_with(scratch, RuleLLM(labelled, catalogue, oracle(catalogue)))

    results = variants_of(evaluation)
    retrieval: dict[str, dict[str, Any]] = evaluation.summary["retrieval"]  # type: ignore[assignment]
    assert evaluation.errors == []
    assert len(evaluation.rows) == len(VARIANTS) * 50

    # Without retrieval nothing has to be cited, so the oracle is right every time.
    assert results["none-haiku"]["root_cause_accuracy"] == 1.0
    assert results["none-haiku"]["retrieval_recall_at_5"] is None
    assert results["none-haiku"]["citation_faithfulness"] is None

    for name in RETRIEVING:
        of_variant = results[name]
        mode = BY_NAME[name].mode
        # With retrieval the oracle can only answer when a document of the fault was found,
        # so its accuracy is the hit rate of the search: two independent code paths agree.
        assert of_variant["root_cause_accuracy"] == retrieval[mode]["hit_at_5"]
        assert of_variant["retrieval_recall_at_5"] == retrieval[mode]["recall_at_5"]
        assert of_variant["retrieval_hit_at_5"] == retrieval[mode]["hit_at_5"]
        assert of_variant["wrong_cause_rate"] == 0.0
        assert of_variant["abstained_despite_evidence_rate"] == pytest.approx(
            1.0 - retrieval[mode]["hit_at_5"], abs=1e-4
        )
        assert of_variant["citation_faithfulness"] == 1.0
        assert of_variant["cites_a_document_of_the_fault"] == 1.0
        assert of_variant["draft_citation_validity"] == 1.0
        assert of_variant["repair_rate"] == 0.0
        assert of_variant["answers_judged"] == of_variant["answers_to_judge"] > 0

    for of_variant in results.values():
        assert of_variant["abstention_without_evidence"] == 1.0
        assert of_variant["incidents_with_evidence"] == 38
        assert of_variant["incidents_without_evidence"] == 12
        assert of_variant["unusable_answers"] == {"refused": 0, "truncated": 0, "invalid_output": 0}
        assert of_variant["latency_p50_s"] == of_variant["latency_p95_s"] == 1.0

    controls: dict[str, Any] = evaluation.summary["judge"]["controls"]  # type: ignore[index]
    assert controls == {
        "controls": 72,
        "as_expected": 72,
        "by_kind": {
            "other_cause": {"controls": 24, "as_expected": 24},
            "supported": {"controls": 24, "as_expected": 24},
            "symptom_only": {"controls": 24, "as_expected": 24},
        },
    }


async def test_cost_is_the_recorded_tokens_at_the_price_of_each_variants_model(
    scratch: Settings, labelled: list[LabelledIncident], catalogue: Catalogue
) -> None:
    evaluation = await run_with(
        scratch, RuleLLM(labelled, catalogue, oracle(catalogue)), judge="none"
    )

    usage = (await RuleLLM(labelled, catalogue, always_abstains).complete(_any_request())).usage
    results = variants_of(evaluation)
    assert results["hybrid-haiku"]["cost_usd_per_incident"] == pytest.approx(
        cost_usd("claude-haiku-5-5", usage), abs=1e-6
    )
    assert results["hybrid-sonnet"]["cost_usd_per_incident"] == pytest.approx(
        cost_usd("claude-sonnet-5-5", usage), abs=1e-6
    )
    assert results["hybrid-haiku"]["input_tokens_per_incident"] == 2000
    assert results["hybrid-haiku"]["citation_faithfulness"] is None


def _any_request() -> LLMRequest:
    from triage.evaluation.rows import Verdict

    return LLMRequest("claude-haiku-5-5", "low", 100, "", "<cited_passages>", Verdict)


async def test_a_model_that_always_abstains_is_never_right_about_a_cause(
    scratch: Settings, labelled: list[LabelledIncident], catalogue: Catalogue
) -> None:
    evaluation = await run_with(scratch, RuleLLM(labelled, catalogue, always_abstains))

    for of_variant in variants_of(evaluation).values():
        assert of_variant["root_cause_accuracy"] == 0.0
        assert of_variant["abstained_despite_evidence_rate"] == 1.0
        assert of_variant["abstention_without_evidence"] == 1.0
        assert of_variant["answers_to_judge"] == 0
        assert of_variant["citation_faithfulness"] is None


async def test_a_model_that_names_one_cause_for_everything_gets_past_only_where_nothing_is_checked(
    scratch: Settings, labelled: list[LabelledIncident], catalogue: Catalogue
) -> None:
    evaluation = await run_with(scratch, RuleLLM(labelled, catalogue, always_rate_limiting))

    results = variants_of(evaluation)
    share = sum(item.label.category is Category.RATE_LIMITING for item in labelled) / 38
    # Without retrieval there is no citation to check: it is right as often as its one answer
    # happens to be the label, and it never abstains.
    assert results["none-haiku"]["root_cause_accuracy"] == pytest.approx(share, abs=1e-4)
    assert results["none-haiku"]["abstention_without_evidence"] == 0.0
    # With retrieval its invented quote fails the check twice, and the agent abstains for it.
    for name in RETRIEVING:
        assert results[name]["root_cause_accuracy"] == 0.0
        assert results[name]["abstention_without_evidence"] == 1.0
        assert results[name]["draft_citation_validity"] == 0.0
        assert results[name]["repair_rate"] == 1.0
    reasons = {row.abstained_because for row in evaluation.rows if row.variant == "hybrid-haiku"}
    assert reasons == {"no citation could be verified"}


async def test_a_replay_rebuilds_a_recorded_run_byte_for_byte_and_pays_nothing(
    scratch: Settings, labelled: list[LabelledIncident], catalogue: Catalogue
) -> None:
    llm = RuleLLM(labelled, catalogue, oracle(catalogue))
    recorded = await run_with(scratch, llm, reps=runner.REPETITIONS)
    paid_calls = len(llm.requests)

    replayed = await runner.evaluate(scratch, live=False, log=lambda _: None)
    again = await run_with(scratch, llm, reps=runner.REPETITIONS)

    assert runner.rendered(replayed) == runner.rendered(recorded)
    assert runner.rendered(again) == runner.rendered(recorded)
    assert len(llm.requests) == paid_calls


async def test_a_changed_prompt_makes_the_recordings_stale(
    scratch: Settings,
    labelled: list[LabelledIncident],
    catalogue: Catalogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await run_with(
        scratch, RuleLLM(labelled, catalogue, oracle(catalogue)), reps=runner.REPETITIONS
    )
    monkeypatch.setattr(prompts, "SYSTEM_WITH_EVIDENCE", prompts.SYSTEM_WITH_EVIDENCE + " ")

    with pytest.raises(runner.StaleResults, match="vector-haiku, incident T") as raised:
        await runner.evaluate(scratch, live=False, log=lambda _: None)

    assert "has no recorded answer" in str(raised.value)


async def test_a_run_stops_at_its_cap_and_the_next_one_pays_only_for_the_rest(
    scratch: Settings, labelled: list[LabelledIncident], catalogue: Catalogue
) -> None:
    llm = RuleLLM(labelled, catalogue, oracle(catalogue))
    per_call = cost_usd("claude-haiku-5-5", (await llm.complete(_any_request())).usage)
    llm.requests.clear()
    only = (BY_NAME["none-haiku"],)

    with pytest.raises(BudgetExceeded):
        await run_with(scratch, llm, variants=only, judge="none", budget=Budget(per_call * 10))
    paid_before_the_cap = len(llm.requests)
    evaluation = await run_with(scratch, llm, variants=only, judge="none")

    assert 10 <= paid_before_the_cap < 10 + runner.CONCURRENCY
    assert len(evaluation.rows) == 50
    assert len(llm.requests) == 50


async def test_an_api_failure_is_an_error_row_and_not_a_wrong_answer(
    scratch: Settings, labelled: list[LabelledIncident], catalogue: Catalogue
) -> None:
    unlucky = prompts.incident_block(labelled[0].incident)

    class Flaky(RuleLLM):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            if request.user.startswith(unlucky):
                raise LLMError("no response after 6 attempts")
            return await super().complete(request)

    evaluation = await run_with(
        scratch,
        Flaky(labelled, catalogue, oracle(catalogue)),
        variants=(BY_NAME["none-haiku"],),
        judge="none",
    )

    [error] = evaluation.errors
    assert (error.variant, error.case_id, error.kind) == ("none-haiku", labelled[0].id, "api_error")
    assert labelled[0].id not in {row.case_id for row in evaluation.rows}
    assert len(evaluation.rows) == 49
    assert variants_of(evaluation)["none-haiku"]["root_cause_accuracy"] == 1.0


def test_a_sample_takes_turns_across_the_kinds_of_incident(
    labelled: list[LabelledIncident],
) -> None:
    sample = runner.sample_cases(labelled, 5)

    kinds = [(item.label.answerable, item.label.presentation) for item in sample]
    assert len(set(kinds)) == 5
    assert sum(1 for answerable, _ in kinds if not answerable) == 2
    assert runner.sample_cases(labelled, 5) == sample
    assert len(runner.sample_cases(labelled, 500)) == 50
