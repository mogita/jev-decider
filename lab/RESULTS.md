# Categorizing bank transactions in one forward pass

Assigns a budget category to a bank transaction, trained on a personal Lunch Money export. The point of interest is generalization to merchants never seen during training.

Two approaches were measured. The first fine-tuned Qwen3-0.6B with the decision heads from [NanoJev](https://github.com/TianyuCodings/NanoJev), a separate project; it scores each candidate as its own sequence, which is accurate enough but too slow to serve. The second, and the one that ships, is single-token constrained decoding on a larger base: one forward pass, every candidate read out of one logit vector. Both sets of numbers are below, because the comparison is the argument for the second.

## Task

Two questions per transaction, answered in one forward pass:

| Question | Type | Candidates |
| --- | --- | --- |
| `category` | choice | 23 budget categories |
| `is_recurring` | boolean | derived from the export's `recurring_id` |

`category_group` is ignored. It nests cleanly under `category`, so predicting the category
implies the group.

## Data

`lab/build_lunchmoney.py` converts a Lunch Money CSV export into the record format the trainers here read.

```bash
python lab/build_lunchmoney.py --csv <export>.csv --output-dir data/lunchmoney
```

The export used here covers 2026-01-01 to 2026-09-18: 680 rows, 34 categories, 6 accounts,
4 currencies. Dropping categories with fewer than 6 examples leaves **655 rows and 23
categories**.

Splits are **merchant-disjoint**, assigned by a SHA-256 hash of the normalized payee, so no
merchant appears on both sides of an evaluation. The trainer enforces this through
`metadata.source_group_id` and refuses a dataset where a source group crosses splits.

| Split | Rows | Questions |
| --- | ---: | ---: |
| train | 427 | 854 |
| dev | 87 | 174 |
| test | 141 | 282 |

No `ood` split. Every candidate holdout account shared merchants with the rest of the data,
which would have contaminated it while costing a third of the training rows. No `calibration`
split either, at this size those rows are better spent on dev.

Longest candidate path is 360 tokens, so `--max-length 512` is sufficient. One 23-candidate
question is up to 8,280 padded tokens, so `--max-microbatch-tokens` must stay at its 16384
default. The runbook's 6000 would reject a single question.

## Baselines on the test split

| Baseline | `category` accuracy |
| --- | ---: |
| Uniform over 23 categories | 4.3% |
| Always predict the majority class (groceries) | 21.3% |
| Exact-match merchant lookup table | 0%, zero coverage by construction |
| Exact-match lookup on a *random row* split | 82% |

That last row is the trap a merchant-disjoint split exists to avoid. A random row split lets
the same payee appear in train and test, and a lookup table then scores 82% without learning
anything.

## Results

All runs: Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`, objective
`gold_distribution`, loss `ce`, `--set-head attention`, `--gradient-checkpointing`, 12 head
steps, 250 full steps, batch 8 questions, microbatch 2.

| Run | Device | Precision | Seed | Accuracy | Top-3 | ECE after temp scaling |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `gpu_v2` | RTX 4090 | bf16 | 17 | 61.7% | 77.3% | 0.076 |
| `bf16_s18` | RTX 4090 | bf16 | 18 | 41.8% | 65.2% | 0.147 |
| `bf16_s19` | RTX 4090 | bf16 | 19 | 48.9% | 63.1% | 0.191 |
| `gpu_fp32_250` | RTX 4090 | fp32 | 17 | 42.6% | 63.1% | 0.139 |
| `fp32_s18` | RTX 4090 | fp32 | 18 | 51.1% | 67.4% | 0.141 |
| `lm_v2` | M4 Max, MPS | fp32 | 17 | 40.4% | 74.5% | 0.164 |

| Arm | n | Mean | SD |
| --- | ---: | ---: | ---: |
| bf16 CUDA | 3 | 50.8% | 10.1 |
| fp32 CUDA | 2 | 46.8% | 6.0 |

**The result that holds: roughly 48% accuracy on unseen merchants, against a 21.3% majority
baseline.** Every run beat the baseline by at least 19 points, across three seeds, two
precisions and two devices.

**Seed variance is 10 points standard deviation**, with a 20-point spread between the best and
worst seed of the same configuration. Any single run is therefore uninformative. Report a
median over at least three seeds.

Two negative findings, both initially misread as real effects before replication:

- **Precision does not matter.** The bf16 and fp32 arms overlap completely. A 19-point gap
  observed at seed 17 disappeared at seed 18.
- **MPS is numerically fine.** fp32 on CUDA scored 42.6% against MPS's 40.4%, well inside the
  noise. Apple silicon is roughly 9x slower here, not wrong.

### Loss function

An arm trained with `--loss brier`, a proper scoring rule, lost on every metric including its
own: 29.1% accuracy and a Brier score of 1.057, against cross-entropy's 42.6% and 0.867 at the
same step count. This matches the conclusion in `docs/RLCD_EXPERIMENT.md`, that the evidence
does not establish the calibrated objectives beating direct controls.

**Post-hoc temperature scaling is the cheap win for calibration.** Fitting a single scalar on
dev halved ECE on the seed-17 bf16 run, 0.169 to 0.076, at zero cost in accuracy, and beat an
hour of retraining with Brier. Note the temperature here is fitted on the same dev split used
for checkpoint selection, which makes it mildly optimistic. With more data, restore the
`calibration` split and fit there.

### `is_recurring`

Stuck at 87.9% in every run, which is exactly the always-false rate. At a 9% positive rate the
model correctly learned that the prior is enough. It is not a result, and the field can be
dropped.

## Checkpoint selection is unreliable at this data size

The trainer selects on minimum dev cross-entropy, hardcoded. On an 87-row dev split this
misfires often: three runs of the same configuration placed their dev minimum at steps 250, 50
and 250. When it lands early the run ships an undertrained model, and the Brier arm's first
attempt scored 12.1% for this reason alone.

Two identical fp32 runs on the same GPU also differed by 0.52 dev CE at step 250, from CUDA's
non-deterministic backward.

Workaround used here: `--steps 250 --eval-every 250`, which evaluates only at the end and
forces a fixed-step checkpoint. This makes runs comparable to each other.

## Running it

On CUDA:

```bash
# In a NanoJev checkout, against the data/lunchmoney built above.
python scripts/train_pipeline_decisions.py \
  --input data/lunchmoney --output-dir runs/<name> \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --objective gold_distribution --loss ce --set-head attention --gradient-checkpointing \
  --seed 17 --head-steps 12 --steps 250 \
  --batch-questions 8 --microbatch-questions 2 \
  --max-microbatch-tokens 16384 --max-length 512 \
  --eval-every 250 --precision bf16 \
  --backbone-lr 2e-5 --head-lr 2e-4 --head-warmup-lr 1e-3
```

On Apple silicon, substitute `--precision fp32`. That trainer's device gate resolves CUDA, then MPS, then CPU; bf16 autocast is CUDA-only. `--gradient-checkpointing` is not optional on MPS: without it the backward pass
allocates 63 GiB and the machine thrashes rather than failing.

Approximate cost of one 250-step run: 8 minutes on an RTX 4090 in bf16, 40 minutes on the same
GPU in fp32, 74 minutes on an M4 Max in fp32.

## Zero-shot scaling probe

`lab/zeroshot_probe.py` measures what a base model knows before any training: it shows the
transaction plus the 23 category names labelled A to W, and reads the next-token logits at
those letters. No head, no fine-tuning, same 141-row test split.

| Model | top-1 | top-3 |
| --- | ---: | ---: |
| Qwen3-0.6B | 6.4% | 12.1% |
| Qwen3-1.7B | 27.0% | 44.0% |
| Qwen3-4B | 48.2% | 61.7% |
| Qwen3-14B | 58.9% | 70.2% |

Three conclusions. The 0.6B knows nothing about these merchants, so its fine-tuned 47.8% was
almost entirely learned from 427 rows. An untrained 4B matches that fine-tuned 0.6B. An
untrained 14B beats it by 10 points. And the curve has not flattened at 14B.

Merchant world knowledge, not architecture or objective, was the binding constraint all along.

## QLoRA on Qwen3-14B

The 500 ms serving target rules out the NanoJev architecture: it scores each candidate as its own sequence, so 23 categories means 23 forward passes through a 14B, which is seconds. Single-token
constrained decoding needs **one** forward pass and reads all 23 candidates out of one logit
vector, so latency is flat in the number of categories.

`lab/qlora_train.py` trains that objective. Frozen 4-bit NF4 base, LoRA r=16 alpha=32 on
attention and MLP, 64.2M trainable parameters against a 14.8B base (0.43%), cross-entropy over
the 23 candidate letters at one position, 4 epochs, effective batch 16, cosine LR 1e-4, epoch
selected on dev top-1.

```bash
python lab/qlora_train.py --model Qwen/Qwen3-14B --data-dir data/lunchmoney \
  --output-dir runs/qlora14b_s17 --epochs 4 --batch-size 4 --accum 4 --lr 1e-4 \
  --rank 16 --alpha 32 --seed 17
```

| Seed | test top-1 | test top-3 |
| --- | ---: | ---: |
| 17 | 73.0% | 90.1% |
| 18 | 70.9% | 92.2% |
| 19 | 69.5% | 90.8% |
| **mean** | **71.2% ± 1.8** | **91.0% ± 1.1** |

25 minutes per seed on an RTX A6000. 4-bit quantization is free here: the frozen base scores
58.9% in NF4 against 58.2% in bf16, so serving costs ~10 GB rather than ~30 GB.

Note the seed standard deviation, **1.8 against the 0.6B's 8.0**. A LoRA nudging existing
knowledge is far more stable than a small model learning the task from nothing.

Calibration: ECE 0.199 raw, **0.093** at temperature 1.7 fitted on dev.

## Serving

`lab/serve_decision.py` exposes the model as structured state in, typed distribution out. One
forward pass, nothing generated, so the response is a probability distribution over the declared
enum and a type error is impossible.

```bash
python lab/serve_decision.py --adapter runs/qlora14b_s17/adapter --temperature 1.7 --port 8900
python lab/serve_decision.py --adapter runs/qlora14b_s17/adapter --bench 40   # latency only
```

Measured on an A6000 with a 396-token prompt through the real 4-bit serving path:
**248 ms p50, 280 ms p95**. The same path in bf16 was 137 ms, so 4-bit dequantization costs
roughly 110 ms per request while saving 20 GB.

Live examples, none of these merchants in the training split:

```
Albert Heijn        -> groceries        0.98   250 ms
Netflix             -> subscriptions    0.97   209 ms
NS Reizigers        -> public_transit   0.98   211 ms
SQ *KRUIDVAT 8812   -> healthcare       0.94   219 ms
```

## Summary across every approach

| Approach | top-1 | top-3 |
| --- | ---: | ---: |
| Majority baseline | 21.3% | |
| Merchant lookup table | 0% | |
| Fine-tuned 0.6B, NanoJev heads, 3 seeds | 47.8% ± 8.0 | ~68% |
| Zero-shot 14B, no training | 58.9% | 70.2% |
| QLoRA 14B, 3 seeds | **71.2% ± 1.8** | **91.0% ± 1.1** |

## What would actually help

Ranked by expected effect.

1. **More data.** 427 training rows across 23 classes remains the ceiling on every approach.
2. **Fix the label inconsistencies.** 8 rows where the same merchant and amount carry different
   categories across months. Separately, about 20 rows are irreducibly ambiguous (one Amazon
   charge cannot be distinguished from another), which caps any model near 97%.
3. **Always report a mean over three or more seeds.** Single runs misled this project twice.
4. **Drop `is_recurring`.** It never moved off the 87.9% always-false rate in any run.

---

# Session 2: staged data, class weighting, and ablations

Everything above used a 13-category schema and hash-assigned splits. Both changed here, so the
numbers below are not directly comparable to those above; the control run exists for that reason.

## Two bugs fixed first

**Splits ignored labels.** Assignment hashed the payee, which kept merchants disjoint but left
dev with 10% groceries against train's 41%, and 3 of 13 categories missing from dev entirely.
Every checkpoint selected on it was chosen from an unrepresentative sample: one run reported 58%
on dev and 86% on test. `stratify()` now keeps merchants whole while holding each category near
65/15/20.

**Plain accuracy flattered the model.** The same predictions score 74.0% plain and 59.7%
balanced. Balanced accuracy, the mean of per-class accuracy, is now the headline and the
selection metric. Per-row logits go to `predictions_test.jsonl` so any metric can be recomputed
without retraining.

## Data: 452 real rows became 73,048

| Stage | Source | Rows |
| --- | --- | ---: |
| 1 | US synthetic ([us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)), signed from the [debit]/[credit] prefix | 64,596 + 3,404 holdout |
| 2 | Synthetic Dutch: 1,074 real OSM merchants in the nine real bank formats | 8,000 |
| 3 | The real export, trained last | 452 |

Schema grew to 17 categories: donation dropped, Restaurants/Pets/Travel kept at 1-4 real rows
each because synthetic data now covers them, and Healthcare plus Education added as options with
no Dutch rows at all, defined entirely by the US data.

## Result

Qwen3-4B, LoRA r=16, class-weighted loss capped at 6x, decaying LR 1e-4 / 7e-5 / 3e-5.

| | EU balanced | US balanced | EU top-1 | EU top-3 |
| --- | ---: | ---: | ---: | ---: |
| zero-shot 4B | 50.1 | — | 64.1 | 79.7 |
| control (real rows only, class-weighted) | 62.2 ± 2.8 | 65.2 | 77.3 | 89.4 |
| **full recipe** | **71.3 ± 3.0** | **94.8** | 78.5 | 90.2 |
| 14B reference (13-cat schema) | 73.8 | — | 82.7 | 96.1 |

**+9.1 balanced on EU and +29.6 on US.** The two seed groups do not overlap: the worst full run
(69.2) beats the best control (64.1). The 4B now matches the 14B at a quarter the size.

Worth noting the control scores 65.2 on US transactions having trained only on 452 Dutch rows.
Whatever it learned transfers across continents, which is what made the US stage's value an open
question rather than an assumption.

## Ablations: what the model actually reads

`lab/ablate.py` blanks one field at a time and re-scores the same checkpoint. Balanced accuracy,
delta against the full prompt:

| Removed | ctrl EU | ctrl US | full EU | full US |
| --- | ---: | ---: | ---: | ---: |
| merchant text (both fields) | **-56.7** | **-59.4** | **-62.6** | **-71.3** |
| bank description | -11.5 | -1.2 | -8.1 | -3.8 |
| payee | -11.6 | -3.9 | -1.7 | -4.8 |
| amount | +2.1 | -2.4 | -9.4 | -3.2 |
| account | -2.1 | -0.3 | +0.0 | +0.2 |
| date | +0.7 | -1.2 | +0.0 | -1.8 |
| amount + date + account | +0.8 | -2.5 | **+1.5** | -2.9 |

**The merchant string is the entire mechanism.** Remove it and three of four columns land at 6 to
7% balanced against 5.9% chance. Account and date contribute nothing on any model or split.

This rules out every shortcut we could construct: not merchant memorisation (merchant-disjoint
splits, where a lookup table scores 0% against 82% on a random row split), not the class prior
(balanced accuracy), not Dutch string formats (94.8% on US data), not the amount, account or date.

**One caveat on the method.** On the full model's EU set, removing the amount alone costs 9.4
points while removing amount, date and account together *gains* 1.5. Information cannot behave
that way. Blanking one field leaves a prompt shape the model never trained on, so part of every
single-field drop is format shock rather than lost information. The model is sensitive to prompt
shape, not only content, and the per-field numbers overstate each field's importance.

**Practical:** `only_merchant_text` is the best configuration measured on the shipping model,
70.7 against 69.2, at roughly half the prompt length. It costs 2.9 points on US, so shortening
the prompt is a small explicit trade rather than a free win.

## Cost

$5.95 across the day: three pods, twenty-odd training runs, four zero-shot probe sweeps and
twenty-eight ablation passes.

## Session 3: caller-supplied schemas

The trained model knows one 17-way budget taxonomy. Serving a caller's own enum costs almost
nothing to allow, since the letters carry the decision and only the label mapping changes, but
allowing it says nothing about whether it works. Measured rather than assumed, on two public
sets neither model has trained on, 400 balanced rows each, same single-forward-pass path.

| Task | Classes | Model | Accuracy | ECE | Mean confidence |
| --- | ---: | --- | ---: | ---: | ---: |
| AG News | 4 | tuned | 84.2% | 0.115 | 0.951 |
| AG News | 4 | frozen base | 85.2% | 0.145 | 0.998 |
| Emotion | 6 | tuned | 46.0% | 0.314 | 0.771 |
| Emotion | 6 | frozen base | 43.9% | 0.541 | 0.980 |

**Fitting a budget taxonomy did not cost general ability.** Accuracy moved -1.0 points on AG
News and +2.0 on emotion, against margins of 3.6 and 4.9 points at these sample sizes. Neither
difference means anything.

**It did improve calibration, and that part transfers.** ECE fell on both tasks, because the
frozen base answers nearly everything at 98% confidence while the tuned model spreads its
probability. This is the one thing the tuning bought that shows up off distribution.

**The confidence is still not trustworthy off the trained schema.** On emotion the tuned model
claims 0.77 and is right 0.46 of the time. The 0.080 calibration error measured on budget
categories describes that schema only, which is why the response carries `"schema": "custom"`
and the page says so.

Emotion fails where the classes overlap: `love` 22.7%, `joy` 63.6%. A fixed label list offered
in one prompt collapses semantically adjacent classes, and no amount of serving code fixes that.

Harness: `lab/schema_eval.py`. Results: `lab/schema_*.json`.

### Does the caller need to supply the wording?

No. The 84.2% and 46.0% above came from prompts written per task. A two-field API cannot ask for
those, so the same sets were rerun with one neutral prompt that names nothing about the domain.

| Task | Task-specific prompt | Neutral prompt | Margin |
| --- | ---: | ---: | ---: |
| AG News | 84.2% | **86.2%** | 3.4 |
| Emotion | 46.0% | **47.2%** | 4.9 |

Both moved up, both inside the margin. Writing a prompt per task bought nothing measurable, so
the contract is `input` plus an optional `schema`, and the service picks the wording: the trained
wording when no schema is supplied, so the figures measured on that path still describe it, and
the neutral one otherwise.

# Session 4: a public benchmark

Every number above is this project's own: our holdout, our split, our harness. They are honest and they are unshared, so nobody can place them. [jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) v0.1.1 fixes that. It is 22 public sources, 22,773 test records, each a `(state, question, label)` triple in the wire format a decision model consumes, and it publishes the same columns for the commercial Jev 1.13.0 API and for eighteen open checkpoints. Four of its sources carry the human vote shares behind each label, which is the part worth having: accuracy against one gold label cannot separate an overconfident model from a correct one on an item the annotators themselves split 60/40.

Seventeen of the 22 fit here. The other five offer 28 to 151 options against the twenty-six a single letter can address. Every macro figure below is therefore recomputed over the same seventeen for every model, ours and theirs: the five we skip are the ones where every model on that board scores worst, so reading our seventeen against their twenty-two would flatter us by exactly the amount those five cost everyone else.

Harness: `lab/jevbench_eval.py`, table: `lab/jevbench_table.py`, plan and the sources left out: `lab/BENCH_PLAN.md`.

## Per source

`Ours` is the shipped adapter, `Base` is `Qwen3-4B` frozen through the identical harness, `Jev` is the commercial API as jev-bench published it. `after T` is the same run with one temperature fitted per source on that source's validation split.

| Source | Primitive | n | Ours acc | Base acc | Jev acc | Ours ECE | after T | T |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arc_challenge` | choice | 1000 | 0.878 | 0.871 | 0.979 | 0.066 | 0.020 | 1.6 |
| `boolq` | noul | 1000 | 0.868 | 0.849 | 0.917 | 0.098 | 0.057 | 3.2 |
| `chaosnli` | choice | 1599 | 0.662 | 0.592 | 0.615 | 0.238 | 0.238 | none |
| `civil_comments` | noul | 2000 | 0.606 | 0.637 | 0.729 | 0.268 | 0.116 | 4.0 |
| `fever_evidence` | noul | 1000 | 0.916 | 0.915 | 0.972 | 0.069 | 0.035 | 2.5 |
| `helpsteer2_helpfulness` | score | 1000 | 0.389 | 0.358 | 0.363 | 0.327 | 0.028 | 3.6 |
| `helpsteer2_verbosity` | score | 1000 | 0.590 | 0.596 | 0.341 | 0.171 | 0.060 | 1.7 |
| `measuring_hate_speech` | score | 1000 | 0.398 | 0.367 | 0.527 | 0.465 | 0.022 | 25.0 |
| `mmlu` | choice | 1000 | 0.664 | 0.659 | 0.923 | 0.196 | 0.057 | 2.7 |
| `mnli` | choice | 1000 | 0.818 | 0.818 | 0.883 | 0.123 | 0.047 | 2.3 |
| `paws` | noul | 1000 | 0.772 | 0.778 | 0.846 | 0.182 | 0.047 | 3.6 |
| `sms_spam` | noul | 800 | 0.835 | 0.588 | 0.965 | 0.061 | 0.043 | 1.2 |
| `sst5` | score | 1000 | 0.504 | 0.476 | 0.565 | 0.334 | 0.044 | 3.2 |
| `strategyqa_closed` | noul | 687 | 0.632 | 0.623 | 0.785 | 0.258 | 0.062 | 4.5 |
| `strategyqa_grounded` | noul | 687 | 0.831 | 0.811 | 0.956 | 0.130 | 0.042 | 3.5 |
| `stsb` | score | 1000 | 0.304 | 0.416 | 0.538 | 0.395 | 0.134 | 3.1 |
| `yelp5` | score | 1000 | 0.640 | 0.632 | 0.685 | 0.262 | 0.058 | 2.7 |

## Macro

| Model | acc | ECE | Brier | sel@90 | choice | score | noul | TVD to human |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **This model, temperature fitted** | 0.665 | **0.065** | 0.428 | 0.690 | 0.756 | 0.471 | 0.780 | 0.411 |
| This model, raw | 0.665 | 0.214 | 0.524 | 0.690 | 0.756 | 0.471 | 0.780 | 0.429 |
| Qwen3-4B frozen, temperature fitted | 0.646 | 0.076 | 0.484 | 0.667 | 0.735 | 0.474 | 0.743 | 0.454 |
| Qwen3-4B frozen, raw | 0.646 | 0.329 | 0.671 | 0.667 | 0.735 | 0.474 | 0.743 | 0.476 |
| Jev 1.13.0, the commercial API | 0.740 | 0.104 | 0.323 | 0.764 | 0.850 | 0.503 | 0.881 | 0.350 |
| Qwen3.5-4B, LoRA and residual heads | 0.747 | 0.097 | 0.324 | 0.771 | 0.799 | 0.529 | 0.903 | 0.286 |
| Qwen3.5-9B, frozen | 0.701 | 0.095 | 0.354 | 0.722 | 0.797 | 0.503 | 0.815 | 0.333 |
| Qwen3.5-4B, frozen | 0.670 | 0.087 | 0.373 | 0.691 | 0.753 | 0.468 | 0.796 | 0.349 |

## Tuning on one schema helped seventeen others

The frozen base and the shipped adapter differ only by the LoRA, and they went through the same rows, the same prompts and the same metrics. The adapter wins every macro column: accuracy 0.646 to 0.665, ECE 0.329 to 0.214 raw and 0.076 to 0.065 after fitting, Brier 0.671 to 0.524, distance to the human distributions 0.476 to 0.411.

That was not the expected result. A LoRA trained on 28,452 rows of one seventeen-way budget schema had no obvious reason to help a model answer an exam question or rate a movie review, and the plausible outcome was damage. What it appears to have taught is the shape of the task rather than its content: read a record, read a candidate list, put the mass on one letter.

The two ends of that:

- **`sms_spam` 0.588 to 0.835.** A twenty-five point gain on a source the model never saw, and the largest single move in either direction. It is also the source needing the least smoothing afterwards, temperature 1.2 against a typical 3, so the adapter arrived at both the answer and the confidence.
- **`stsb` 0.416 to 0.304.** The one real regression, eleven points. Six ordered levels of semantic similarity is the furthest thing here from picking a category, and the tuning cost the model something on it.

## Calibration, once the comparison is fair

The raw ECE of 0.214 is not comparable to the published baselines, and the direction of the unfairness is against us: jev-bench describes its Tier 0 rows as "prompt, logit readout, and a recipe fitted on the validation splits only", so every one of those numbers had been fitted before it was scored and ours had not. jev-bench ships a validation split per source for exactly this, so one temperature per source was fitted there by minimizing the same loss, and applied to test.

**0.065, which is the lowest number in the table, below Jev's 0.104 and below every open checkpoint on that board.** Accuracy is unchanged, as temperature scaling cannot move an argmax.

Three things keep that from being a bigger claim than it is. `chaosnli` ships no validation split, so it enters both macros raw at 0.238, which raises our fitted figure rather than lowering it. The baselines fitted a whole recipe and this fits one scalar, so they had more room, not less. And a temperature above 3 on most sources is a model saying its own confidences were nearly meaningless; `measuring_hate_speech` ran to the ceiling of the search at 25, which is the fit reporting that the best thing to do with those probabilities is flatten them almost to uniform.

## What these numbers are not

- **A comparison against the Qwen3.5 rows.** Those are a later base generation. The clean comparison here is this model against its own frozen base, which is the pair that differs by one thing.
- **A fair reading of the Noul column.** This model has one primitive, so every yes-or-no source was asked as a two-way choice. jev-bench's own probes find the same question better calibrated as a native Noul than as a 2-way Choice, BoolQ 0.028 against 0.054, so the Noul figures here sit on the worse of the two geometries.
- **Evidence about high-cardinality routing.** The five skipped sources are where the interesting failures live, `go_emotions` in particular, where Jev itself drops to 0.282. Nothing here says what this model would do with 77 intents.
- **More than one seed.**
