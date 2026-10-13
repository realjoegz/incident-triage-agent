# Documents

| Document | What it is |
|---|---|
| [design.md](design.md) | What the agent promises, the pipeline, retrieval, the citation check, the model interface, what makes a run repeatable, what was rejected and the limits |
| [evaluation.md](evaluation.md) | What is run, what each metric means and who grades it, the judge and its controls, how the harness itself is tested, what CI checks, the results and how far they can be trusted |
| [data.md](data.md) | The synthetic data set: what is in it, where the labels come from, what is checked, what was changed and when, and its limits |

Where to start:

- To judge the numbers, read sections 2, 3 and 6 of [evaluation.md](evaluation.md): the metrics, the judge and the results.
- To judge the design, read sections 1, 5 and 9 of [design.md](design.md): the promise, the citation check and the limits.
- To check a number, find it in [`eval/results/summary.json`](../eval/results/summary.json), then run `triage eval replay`, which computes it again from the recorded answers without an API key.
