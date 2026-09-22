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

## The benchmark actually being run

The task list above was written from a release announcement. Looking for the tasks themselves turned up something better and different: [jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) v0.1.1, a public dataset of 22 sources and 22,773 test records, each one a `(state, question, label)` triple in the wire format a decision model consumes, built by [Jevify](https://github.com/uspraveen/Jevify). It is not the 231-task JevBench the announcement named, and the two are not interchangeable: this one runs to 151 options where that one was described as capping at 6. What makes it the right target anyway is that it publishes the same columns for Jev 1.13.0 and for eighteen open checkpoints, so a number measured here lands on a board other people are already standing on.

Four of its sources carry the human vote shares behind each label, which is the part worth having. Accuracy against a single gold label cannot tell an overconfident model from a correct one on an item humans themselves split 60/40; total variation distance to the human distribution can.

The comparison points that matter, from its leaderboard:

| Model | Macro acc | Macro ECE | Choice | Score | Noul | TVD to human |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev 1.13.0, the commercial API | 0.733 | 0.113 | 0.770 | 0.503 | 0.881 | 0.432 |
| Qwen3.5-4B, LoRA plus residual heads | 0.747 | 0.110 | 0.770 | 0.529 | 0.903 | 0.347 |
| Qwen3.5-4B, frozen, prompt and logit readout only | 0.662 | 0.093 | 0.687 | 0.468 | 0.796 | 0.438 |

That last row is the honest neighbor for this repo. It is the same size and the same idea, scored with no training at all, and the question this run answers is what a LoRA tuned on one narrow schema does to general decision ability: the transaction adapter could plausibly be a small win, a wash, or real damage, and nobody has measured which.

## What is runnable here

The binding constraint is the architecture, not the compute. This model reads one letter out of one logit vector, so a source is runnable exactly when its answers are a short enumerated list, and is not runnable when it needs free text, several turns, a tool, or an image.

Seventeen of the 22 sources fit, 17,773 test records. All three primitives are covered: `choice` maps onto the candidate list directly, `score` offers the ordered levels as the candidates and reads the level index back, and `noul` is asked as the two-way yes-or-no choice, which is the only form available here. That last one costs something measurable and should be said out loud when the numbers are published: jev-bench's own probes find the same question better calibrated as a native Noul than as a 2-way Choice, BoolQ 0.028 against 0.054, so the Noul column here is measured on the worse of the two geometries.

| Source | K | Primitive |
| --- | ---: | --- |
| `mmlu`, `arc_challenge` | 4, 3 to 5 | choice |
| `mnli`, `chaosnli` | 3 | choice |
| `sst5`, `yelp5`, `helpsteer2_helpfulness`, `helpsteer2_verbosity` | 5 | score |
| `stsb` | 6 | score |
| `measuring_hate_speech` | 3 | score |
| `boolq`, `fever_evidence`, `paws`, `civil_comments`, `sms_spam`, `strategyqa_closed`, `strategyqa_grounded` | 2 | noul |

## What is not, and why

- **Five sources with more options than letters.** `go_emotions` at 28, `massive` at 60, `banking77` at 77, `ledgar` at 100, `clinc150` at 151. The candidates do not have to be letters, only distinct single tokens, so 26 is not the true ceiling; but the tuning taught this model uppercase letters, and swapping in an untrained candidate alphabet would measure the swap rather than the model. Worth trying as its own experiment, not as part of a comparison run.
- **The vision sources.** POPE, A-OKVQA, AI2D. There is no vision encoder here, and adding one is a different project.
- **The Laya comparison chart.** It needs a research branch pinned to a specific commit, fifty-one language sweeps, and input hashes its own documentation says are unavailable.
- **An informal "alternatives" ranking circulating in the replies.** No published method, no data.
- **S1bench.** Named once in a reply with no link. Ask what it is before spending an hour on it.

## Steps

All local on the M4 Max. No GPU spend.

Limits are off by default here. The serving path caps input length, category count and records per call to protect a single machine on a public URL, and none of that describes what the model can do. A benchmark run applies a limit only where something genuinely forces one, such as the twenty-six candidates a single letter can address.

1. **`lab/jevbench_eval.py`.** Done. Maps each source's instructions, criteria and labels onto `MlxDecider.decide_batch(states, template=..., criteria=...)`, groups rows that share a question so they ride one call, and reports the leaderboard's own columns: accuracy, top-label ECE, Brier, NLL, selective accuracy at 90% and 50% coverage, AURC, RPS and MAE on the ordinal sources, AUROC on the boolean ones, and total variation distance to the human distribution where one exists. Accuracy, ECE, Brier, NLL, RPS and MAE come from `jevify.evaluation.metrics` rather than a second implementation of the same formulas, so the numbers are comparable by construction and not by inspection.
2. **Run it unguarded.** Done, and it is not optional: `helpsteer2` states reach 8,261 characters against the 4,000 character service guard, and several sources carry option descriptions past the 80 character label guard. Those caps exist to keep one laptop answerable on a public URL, they describe the deployment and not the model, and the runner lifts them. The twenty-six letters stay.
3. **Retire the `schema_eval.py` extension.** Superseded: BoolQ, the UCI SMS spam set and the NLI sets it was going to add are all sources in jev-bench already, measured against published baselines instead of against nothing.
4. **Fit temperature per decision type** rather than one scalar for everything, and carve a real calibration split out of train instead of reusing the split that selects the checkpoint, which `RESULTS.md` already flags as mildly optimistic. jev-bench ships a validation split per source for exactly this, so the fit never touches test. One to two hours.
5. **Align the latency protocol**: three warmups, twenty synchronized repetitions, p50 and interpolated p95, amortized milliseconds per decision at batch 1, 5 and 10. Half an hour.
6. **Run the frozen base through the identical harness.** Qwen3-4B with no adapter, same prompts, same rows, same metrics. Without it there is no way to tell what the transaction tuning cost or bought, and the leaderboard's own Tier 0 row is a different base model. About an hour of wall clock.
