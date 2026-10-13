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

## 6. Results

The run of 2026-10-13: 750 answers, 301 of them judged, and 144 judge controls. It made 1,145 paid calls and cost $3.73, of which $1.84 went to the five variants and $1.89 to the judge. Every attempt got a response, and no answer was refused, cut off or invalid.

<!-- results:start -->
| Variant | Recall@5 | Root-cause accuracy | Abstains without evidence | Citation faithfulness | Cost per incident | Latency p50 / p95 |
|---|---|---|---|---|---|---|
| No retrieval, Haiku 5.5 | n/a | 43% | 83% | n/a | $0.0003 | 2.4 s / 3.7 s |
| Vector, Haiku 5.5 | 0.89 | 60% | 100% | 95% | $0.0007 | 3.5 s / 5.8 s |
| Full text, Haiku 5.5 | 0.79 | 58% | 100% | 91% | $0.0007 | 3.5 s / 6.1 s |
| Hybrid, Haiku 5.5 | 0.93 | 67% | 100% | 91% | $0.0007 | 3.5 s / 6.5 s |
| Hybrid, Sonnet 5.5 | 0.93 | 79% | 100% | 97% | $0.0103 | 2.8 s / 6.5 s |
<!-- results:end -->

<!-- detail:start -->
| Variant | Root-cause accuracy (95% interval) | Right and citing the fault | Wrong cause | Abstained despite evidence | Abstains without evidence (95% interval) | First-draft citations valid | Repaired |
|---|---|---|---|---|---|---|---|
| No retrieval, Haiku 5.5 | 43% (28 to 59) | n/a | 8% | 49% | 83% (67 to 100) | n/a | 0% |
| Vector, Haiku 5.5 | 60% (47 to 73) | 60% | 1% | 39% | 100% (100 to 100) | 100% | 0% |
| Full text, Haiku 5.5 | 58% (45 to 71) | 58% | 0% | 42% | 100% (100 to 100) | 99% | 1% |
| Hybrid, Haiku 5.5 | 67% (53 to 80) | 67% | 0% | 33% | 100% (100 to 100) | 100% | 0% |
| Hybrid, Sonnet 5.5 | 79% (67 to 89) | 79% | 0% | 21% | 100% (100 to 100) | 100% | 0% |
<!-- detail:end -->

Root-cause accuracy by the way an incident is shown ([data.md](data.md) describes the three):

<!-- presentation:start -->
| Variant | `signature` | `paraphrase` | `downstream` |
|---|---|---|---|
| No retrieval, Haiku 5.5 | 92% | 26% | 15% |
| Vector, Haiku 5.5 | 97% | 62% | 23% |
| Full text, Haiku 5.5 | 100% | 38% | 38% |
| Hybrid, Haiku 5.5 | 100% | 59% | 44% |
| Hybrid, Sonnet 5.5 | 100% | 79% | 59% |
<!-- presentation:end -->

**With retrieval, a named cause was right 300 times in 301.** Across the four variants with retrieval, 301 answers named a cause. 300 named the labelled category, and every one of the 301 cited a document of the incident's own fault. None of the 144 answers to an incident without evidence named a cause. The stricter accuracy of [section 2](#2-the-metrics) equals the plain one in all four variants.

**The errors are abstentions.** Haiku with hybrid retrieval said "insufficient evidence" for 33% of the incidents that the corpus documents, and Sonnet for 21%. The table by presentation shows where: every variant with retrieval is at 97% or above when the fault's own log line is in the excerpt, and Haiku with hybrid retrieval falls to 59% on a ticket in a user's words and to 44% when the incident is seen from a dependent service. The agent is built to prefer silence to a guess, and on this set it pays for that with a third of the answers it could have given.

**Retrieval is worth 24 points over none**, with a 95% interval of 11 to 39. Without retrieval the model named a cause 64 times: 9 were wrong, and 6 were for an incident that the corpus does not document. It still reached 92% on the `signature` incidents, where a log line such as `deadlock detected` gives the category away.

**This set cannot separate the three retrieval methods by accuracy.** Hybrid is 7 points above vector and 9 above full text, and both intervals start at zero. Recall separates them more clearly: 0.93, 0.89 and 0.79.

**Sonnet is 12 points above Haiku for 15 times the cost**, with an interval of 0 to 25. It was also faster at the median, 2.8 s against 3.5 s, because Haiku wrote twice as many output tokens at the same effort setting: 607 per incident against 295.

**The judge rejected 22 of 301 answers, and none for the wrong mechanism.** In all 22 its reason grants that the cited documents state the mechanism. What it rejects is an addition:

- 9 answers lean on a retrieved document that they did not cite, such as "and its repeat INC-2026-0308".
- 9 say something about the current incident that the incident does not show, such as "every edge proxy" when the logs name two hosts.
- 4 say that a table was edited "outside a code release". The phrase comes from the category definitions in the prompt, not from any document. Seven judged answers use it, and the judge rejected 4 of them and passed 3, so it is not consistent on this point.

**The judge gave the expected verdict on all 144 controls.**

**Citations almost never needed the repair.** Of 592 citations in first drafts, one failed the check, and one answer in 600 took the second model call.

**Search takes milliseconds.** Over 114 searches through the MCP tools, the median was 1.8 ms and the 95th percentile 4.6 ms, on the laptop that ran the evaluation.

## 7. How far to trust this

- **Small.** One incident is 2.6 points of accuracy, and the interval around 67% runs from 53% to 80%. A difference under about ten points is not a finding here.
- **Synthetic and templated.** The incidents and the documents are rendered from one catalogue by one author. About a third of the incidents carry the fault's own log line, and every variant, with or without retrieval, gets at least 92% of those right.
- **No held-out incidents.** The prompts were written by someone who could see all 50.
- **Abstaining is the easy half.** All four variants with retrieval abstained on all 12 incidents without evidence. Half of those 12 show a log code that no document contains, so 100% here says less than it seems to.
- **The judge is a model.** It agrees with 144 constructed controls, which are clear cases. On a borderline phrase it gave both verdicts.
- **Three answers to one incident are not three samples.** Single repetitions of the Haiku hybrid variant scored 71%, 63% and 66%. The table shows the mean, and the intervals resample faults.
- **Latency is one evening's.** The calls were made from one laptop at a concurrency of four.
- **The spending cap counts responses that arrived.** An attempt that timed out on the client may still have been billed.
