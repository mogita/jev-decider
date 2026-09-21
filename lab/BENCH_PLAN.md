# Benchmark plan

What to measure on the next weights, why each test is on the list, and what was deliberately left off. Written after reading the Dohnuts 0.1.0 release, which is the closest public comparison to what this repo does.

## Why now

The numbers this project publishes today are its own: a merchant-disjoint Dutch holdout, a US test split, two public classification sets. They are honest and they are unshared, which means nobody can place them. A public benchmark other people also run is worth more than a better private one.

## The comparison

[Dohnuts 0.1.0](https://github.com/PsiACE/dohnuts), announced [2026-09-21](https://x.com/repsiace/status/2101927214820499627), is the same shape as this repo: a frozen small base, a rank-8 LoRA and a candidate scorer, one forward pass, a choice and a probability out, no text generated. It is 0.8B against this 4B, and it adds vision and several questions sharing one prefix, which this does not have.

Its published figures, for reference rather than for chasing:

| Measure | Dohnuts 0.1.0 | Reference |
| --- | --- | --- |
| Macro accuracy, 26 held-out groups, 180,031 decisions | 78.21% | one seed, no variance measured |
| JevBench v1.2.2, 231 public tasks | 65.80% | Jev 1.13.0 scores 86.58% on the same set |
| MASSIVE, 51 languages | 60.14% | Laya multilingual 36.61% |
| XNLI, 15 languages | 70.91% | Laya multilingual 73.84% |
| Latency, one text question | 15.07 ms | 0.8B, BF16, RX 7900 XTX, warm |

One detail that changes how our own numbers read: **AG News and dair-ai/emotion are in the Dohnuts training mixture.** The 86.2% and 47.2% in `RESULTS.md` are zero-shot on sets this model never saw, which is a different and stronger claim than a trained-on score. Say so whenever the two are put side by side.

## What is runnable here

The binding constraint is the architecture, not the compute. This model reads one letter out of one logit vector, so a benchmark is runnable exactly when its answers are a short enumerated list, and is not runnable when it needs free text, several turns, a tool, or an image.

| Test | Measures | Runnable | Why |
| --- | --- | --- | --- |
| JevBench v1.2.2, public 231 | Typed decision accuracy on bounded rubrics | **yes** | Every task is a choice, a boolean or a score, and none offers more than 6 labels against a cap of 26. No generation, no multi-turn, no tool use. |
| Per-decision-type temperature, 15-bin ECE | Calibration | **yes** | `calibrate.py` already fits one global temperature and computes the same metric. |
| 26-group macro, text half | Balanced accuracy over a broad public mixture | **adapted** | 12 of the 26 groups are text and fit under 26 labels. A direct extension of the `TASKS` dict in `schema_eval.py`. |
| XNLI | Cross-lingual entailment | **adapted** | Three labels, fits. Run a few languages directly rather than reproducing a 15-language protocol whose inputs cannot be verified. |
| Warm latency protocol | Speed | **adapted** | `serve_mlx.py --bench` already does this shape. Copy the method, never the number: 0.8B on AMD against 4B on Apple silicon is not a comparison. |

## What is not, and why

- **The five vision groups.** CLEVR, A-OKVQA, ScienceQA, VQAv2, ScreenQA. There is no vision encoder here, and adding one is a different project.
- **MASSIVE 60-intent and BANKING77.** Sixty and seventy-seven labels against twenty-six single-letter candidates. Getting past twenty-six needs multi-token candidates or hierarchical routing, which is an architecture change wearing a benchmark's clothes.
- **The official JevBench four-axis score.** One of its axes is cost per thousand decisions, and a self-hosted model has no tariff to quote. 303 of the tasks are undistributed. Dohnuts left the composite unmeasured for the same reasons.
- **The Laya comparison chart.** It needs a research branch pinned to a specific commit, fifty-one language sweeps, and input hashes its own documentation says are unavailable.
- **An informal "alternatives" ranking circulating in the replies.** No published method, no data.
- **S1bench.** Named once in a reply with no link. Ask what it is before spending an hour on it.

## Steps

All local on the M4 Max. No GPU spend.

Limits are off by default here. The serving path caps input length, category count and records per call to protect a single machine on a public URL, and none of that describes what the model can do. A benchmark run applies a limit only where something genuinely forces one, such as the twenty-six candidates a single letter can address.

1. **`lab/jevbench_eval.py`.** Read the public tiers, map each task's instructions, criteria and labels onto `MlxDecider.decide(state, template=..., criteria=...)`, serialize the structured states to JSON, and report accuracy per tier and per family alongside Brier and 15-bin ECE. `alternatives` already returns the full distribution, so nothing has to be adapted to read it. Two to three hours to write, about a minute to run.
2. **Run the benchmark unguarded.** Thirty-seven of the 111 hard tasks are longer than the 4,000 character input guard, the longest being 14,986. That guard exists to keep one laptop from being asked for more than it can give, and it says nothing about the model, which has the context for all of them. A benchmark measures the model, so the benchmark does not apply it, and no limit is applied anywhere unless something actually forces one. The guarded score is worth computing once as a secondary line, because the comparison figure counted over-long rows as wrong and a number measured differently is not a comparison; it belongs in a footnote describing the deployment, not in the headline describing the model. One flag, half an hour.
3. **Extend `schema_eval.py`** with BoolQ, XNLI, WikiQA, the UCI SMS spam set, ESCI and ShARC. All public on the Hub. ContractNLI waits until there is a chunking decision, since most of its rows are whole contracts. Keep the neutral prompt path, so the numbers describe the API as callers meet it. Two hours to add, roughly fifteen minutes per set to run.
4. **Fit temperature per decision type** rather than one scalar for everything, and carve a real calibration split out of train instead of reusing the split that selects the checkpoint, which `RESULTS.md` already flags as mildly optimistic. One to two hours.
5. **Align the latency protocol**: three warmups, twenty synchronized repetitions, p50 and interpolated p95, amortized milliseconds per decision at batch 1, 5 and 10. Half an hour.
6. **Run all of it against the current weights first**, so the PII-scrubbed checkpoint has a before and after on identical harnesses rather than a comparison against numbers measured a different way. About an hour of wall clock, most of it spent loading the model.
