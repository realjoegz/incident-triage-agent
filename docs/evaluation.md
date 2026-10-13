# Evaluation

This document says what each number in the README means, how it is computed and how far it can be trusted. The numbers themselves are in [`eval/results/summary.json`](../eval/results/summary.json), and one row per answer is in [`eval/results/rows.jsonl`](../eval/results/rows.jsonl).

## 1. What is run

Five variants answer the same 50 incidents three times each, which is 750 answers.

| Variant | Retrieval | Model |
|---|---|---|
| `none-haiku` | none: the model sees the incident only | Claude Haiku 5.5 |
| `vector-haiku` | vector search | Claude Haiku 5.5 |
| `fulltext-haiku` | full-text search | Claude Haiku 5.5 |
| `hybrid-haiku` | both, fused | Claude Haiku 5.5 |
| `hybrid-sonnet` | both, fused | Claude Sonnet 5.5 |

The first four differ only in retrieval, and the last differs from the fourth only in the model. Every variant runs at effort `low` with a limit of 8,000 output tokens, and no sampling parameter is set. `hybrid-haiku` is what the API serves by default.

The 50 incidents are described in [data.md](data.md): 38 are made from faults that the corpus documents, and 12 from faults that it does not.

## 2. The metrics

Every rate is averaged within an incident first, over its three answers, and then across incidents. An incident counts once.

| Metric | Over | Definition | Graded by |
|---|---|---|---|
| Recall@5 | 38 incidents with evidence | Of the three documents of the fault, the share among the five retrieved | Code |
| Root-cause accuracy | 38 incidents with evidence | The agent named the labelled category. Abstaining counts as wrong | Code |
| Abstains without evidence | 12 incidents without evidence | The agent answered "insufficient evidence" | Code |
| Citation faithfulness | Answers that name a cause and cite | The cited documents state the mechanism that the answer gives as the cause | Judge model |
| Cost per incident | All 150 answers of a variant | Reported input and output tokens at list price | Code |
| Latency | Answers with a model call | Time of the model calls of one answer, added up | Recorded |

Accuracy and abstention are two halves of one question. A variant that always names a cause scores well on the first and zero on the second, and one that always abstains does the opposite. Neither can do well on both.

The summary file also holds what the README leaves out: the hit rate and the reciprocal rank of retrieval, how often an answer named a wrong cause as opposed to abstaining, how many first-draft citations passed the check, how many answers needed the repair call, and a 95% interval for accuracy and for abstention.

**Accuracy is a match on the category, and a category can be right for the wrong reason.** Three faults on three services are code regressions; an answer that pins an incident on the wrong one of them still names the labelled category. So the summary holds a stricter number beside it: the share of incidents where the category is right and at least one citation points at a document of the incident's own fault.

**Intervals resample faults.** The 50 incidents come from 30 faults, and two incidents of one fault share its log lines and its documents. A bootstrap draw therefore takes whole faults with all their incidents, which gives wider intervals than treating 50 incidents as independent would.

**Three outcomes are never scored.** A refusal, an answer cut off at the token limit and an answer that is not valid JSON are counted per variant under `unusable_answers` and left out of every rate. An attempt that got no response at all, after six tries or five minutes, is written to `errors.jsonl` and is not a row. The gate fails if either count is above zero.

**Cost** is computed from the token counts in each API response and the model that served it, at the prices in [`pricing.py`](../src/triage/llm/pricing.py). No reported cost is estimated from text length, nothing is cached, and the judge's cost is kept in its own field.

**Latency** is that of the attempt that succeeded. Waits between retries and the wait for a free slot among the four concurrent runs are not in it. Search time is measured separately, because a replay in CI cannot reproduce it.

## 3. The judge

Citation faithfulness is the one metric that needs a model. The citation check in the agent already guarantees that every quote is in its document; what code cannot tell is whether the document says what the answer claims.

The judge is Claude Opus 5.5, which is neither of the models being compared. For each answer that names a cause it gets the incident, the stated cause and the cited documents with the quoted passages, and returns one sentence of reasoning and a yes or no. It does not see the label, and its prompt says that whether the cause is the right one is not its question.

**The judge is tested on controls.** For each of the 24 known faults, six inputs are built whose right verdict is known by construction:

| Control | Right verdict | Construction |
|---|---|---|
| `incident_report` | supported | The fault's cause, cited to its own past incident |
| `runbook_section` | supported | The fault's cause, cited to its runbook section |
| `says_what_it_saw` | supported | The cause plus a true sentence about the incident |
| `other_fault` | not supported | The cause, cited to another fault of the same service |
| `no_mechanism` | not supported | The cause, cited to the overview of the service |
| `adds_a_fact` | not supported | The cause plus an invented second mechanism |

The gate fails if the judge gives the expected verdict on fewer than 95% of the 144.

**The controls earned their place before the full run.** On a five-incident sample, the first version of the judge compared the stated cause with the quoted sentences alone and rejected two of three controls that were built to be supported: a quote is a sentence and a cause takes two. The second version read the whole cited documents but was not shown the incident, so it marked an answer down for saying what the incident's own logs showed. The third version is the one described above.

## 4. Is the harness itself right?

Before any number of a real model is believed, the pipeline has to give a model whose score is known in advance the score it deserves. [`test_evaluation.py`](../tests/test_evaluation.py) runs the whole evaluation with three stand-in models:

- **One that knows the answer key** and cites a document of the fault whenever one was retrieved. Without retrieval it must score 100%. With retrieval it must score exactly the hit rate of the search, because it can only answer when the search found a document: two independent code paths have to agree to four decimals.
- **One that always abstains.** It must score 0% accuracy and 100% abstention.
- **One that names the same cause for everything** and cites words that no document contains. Without retrieval it must be right exactly as often as its one answer is the label. With retrieval the citation check must turn every answer into an abstention.

## 5. What CI checks without an API key

`triage eval gate` runs in CI on every push. It makes no model call.

1. **Retrieval is measured again** and compared with the baseline. Search needs no model, so a change to it shows here at once.
2. **The evaluation is replayed** from the recorded responses. The agent retrieves, builds its prompts and checks citations with the current code. A prompt with no recording fails the gate: the committed results no longer describe the code, and the paid evaluation has to be run again.
3. **The replay must compute the committed files**, byte for byte. This catches an edited number and a change to the grading.
4. **The results are compared with [`eval/baseline.json`](../eval/baseline.json).** A metric may fall below its baseline by a fixed tolerance and no further.
5. **The table in the README** must be the one rendered from the results.

A change that makes a paid run worse than the baseline can still be merged, by committing a new baseline with it. That is deliberate: the gate does not forbid a trade-off, it makes one visible in the diff.

[`test_gate.py`](../tests/test_gate.py) builds a passing state and breaks it eight ways, one at a time, and requires the gate to fail each time for the right reason.
