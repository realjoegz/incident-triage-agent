# The data

Everything in `data/` is synthetic. Larkhill is a fictional online shop, and none of its services, vendors, error codes, incidents or runbooks comes from a real system or a real postmortem. The failure modes are common ones: an expired certificate, a full WAL volume, a connection pool that ran dry.

## What there is

| File | Content |
|---|---|
| [`catalogue.yaml`](../data/catalogue.yaml) | The source: 8 services, 24 known faults and 6 faults that are documented nowhere |
| [`corpus.jsonl`](../data/corpus.jsonl) | 80 documents that the agent can retrieve |
| [`incidents.jsonl`](../data/incidents.jsonl) | 50 incidents: an alert, a ticket and a log excerpt each, any of which may be empty |
| [`labels.jsonl`](../data/labels.jsonl) | The answer key of the 50 incidents |
| [`embeddings.jsonl`](../data/embeddings.jsonl) | The vectors of the 80 documents and the 50 incidents |

The last four are generated. `triage data generate` renders the catalogue into the corpus, the incidents and the labels, and `triage data embed` computes the vectors.

**The corpus.** Each known fault has three documents: two past incident reports, dated 2025 and 2026, and one runbook section. Each service has one overview with its dependencies and its on-call team. That is 24 × 3 + 8 = 80 documents of 29 to 155 words.

**The incidents.** 38 are made from the 24 known faults; 14 faults appear twice, in two different presentations. 12 are made from the 6 undocumented faults, two each. Two of the undocumented faults are on services that have no document at all, and four are on known services but match none of that service's known faults.

An incident is shown in one of three ways:

| Presentation | Incidents | What the agent is given |
|---|---|---|
| `signature` | 18 | The alert and a log excerpt that contains the fault's own log line. No ticket |
| `paraphrase` | 19 | A ticket in the words of a user or of support, and log lines that show the effect without the line that names the fault. No alert |
| `downstream` | 13 | The alert, the ticket and the log lines of a service that depends on the faulty one. The fault's own log line is absent |

26 of the 50 excerpts also carry one warning from an unrelated service: a log line that belongs to another known fault and that also occurs when nothing is wrong, such as one declined card or one deadlock victim.

## Where the labels come from

A label is not read off the text of an incident. The text is rendered from a catalogue fault, and the fault's category is the label. No model assigned or checked a label, so no result here measures how well one model imitates another.

- An incident made from a known fault is labelled with that fault's category and its three documents. The right answer is to name the category.
- An incident made from an undocumented fault is labelled as having no evidence. The right answer is to say so.

The rendering is deterministic. Every choice that looks arbitrary, such as a date, a host name or which faults get a second incident, is taken from a hash of a fixed seed.

## What is checked

`triage data check` runs in CI and fails on any of these:

- A generated file is not what the catalogue renders.
- The counts are not 80 documents, 50 incidents and 12 incidents without evidence.
- An incident with evidence does not name three documents that exist, or one without evidence carries a category.
- Two incidents are identical.
- `incidents.jsonl` has a field beyond `id`, `alert`, `log_excerpt` and `ticket_text`.
- A category name or a catalogue id appears in a document or in an incident.
- The log code of an undocumented fault appears in the corpus.
- A `paraphrase` or `downstream` incident shows its own fault's log code.

The agent, the API and the MCP server read `incidents.jsonl` at most. Only the evaluation opens `labels.jsonl`.

## What changed, and when

The data was changed twice, both times before the full evaluation.

**Before any model call.** The first version of the incidents was too easy to tell the retrieval methods apart: vector, full-text and hybrid search each found between 95% and 97% of the documents they should. Two changes followed:

- The `downstream` presentation lost the faulty service's own log line.
- Half of the excerpts gained the unrelated warning.

After them the three methods found 90%, 76% and 90%.

**After a five-incident sample.** An audit of all 50 incidents against their labels, done before the full run, found one label that could be argued and three smaller faults in the text. Four things changed:

- **One cause.** The edge proxy that cannot resolve the checkout service had a deleted DNS record as its cause, which can also be read as a configuration change. The cause is now resolvers that serve a stale copy of the zone. None of the five sampled incidents was made from this fault.
- **Recurrences.** The second incident report of a fault repeated the lasting fix of the first without saying why the fault had come back. It now says that the fix was no longer in place.
- **Downstream alerts.** They claimed a call direction that the service overviews contradict. They no longer name a direction.
- **A tag.** Each fault carried a note of how guessable it is. No metric used it and the audit found it wrong for four faults, so it was removed.

With the reworded documents the three methods find 89%, 79% and 93%. The data has not changed since.

## Limits

- **The text is templated.** The two past incidents of a fault differ in date, severity and trigger and share the rest. Real postmortems of a recurring fault are written by different people in different words.
- **One author.** The incidents are rendered from the same catalogue as the documents, and the same person wrote the catalogue and the agent's prompt.
- **No held-out split.** All 50 incidents were visible while the prompts were written. After the five-incident sample, one sentence was added to the prompt of the no-retrieval variant so that both prompts carry the same caution about unrelated log lines. The prompt of the other four variants did not change after the sample.
- **Some incidents give their category away.** A log line that says `deadlock detected` needs no document. The no-retrieval variant measures how far that goes.
- **Abstaining is easier than 12 incidents suggest.** Six of the 12 show an alert and a log code that occur in no document, and four of the 12 are on a service that has no document at all.
- **Small.** With 38 incidents with evidence, one incident is 2.6 points of accuracy. With 12 without, one is 8.3 points.
- **A closed world.** Every incident is either one of 24 known faults or clearly none of them. A partial match, two faults at once and a known fault with a new cause do not occur.
