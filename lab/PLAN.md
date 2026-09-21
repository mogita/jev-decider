# Next session: raising the 4B

Baseline to beat: **59.7 ± 4.3 balanced** (Qwen3-4B LoRA, 3 seeds, `data/lunchmoney` stratified
splits). Ceiling reference: 14B QLoRA at **73.8 balanced**. Majority-class baseline is 7.7.

## Training recipe

Fresh base `Qwen/Qwen3-4B`, one LoRA adapter, three sequential stages, your real data last:

| Stage | Data | Purpose |
| --- | --- | --- |
| 1 | US 68k ([us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)) | task shape, US coverage for the demo |
| 2 | `data/synth/train.jsonl`, 6,000 rows | Dutch merchant knowledge, balanced across categories |
| 3 | `data/lunchmoney/train.jsonl`, 453 rows | your taxonomy and real formats |

Plus **class-weighted loss**, the one lever with a measured failure behind it: `shopping`,
`income`, `personal_care` and `rent_loan` score at or near zero top-1 while their top-3 is near
perfect, so the model ranks them correctly and never puts them first.

Not continuing from `c04lora_s17`: it has those failure modes baked in, and a fresh start keeps
the comparison against 59.7 honest.

## Guardrails

1. Evaluate on the existing test split. If balanced accuracy drops below 59.7, stage 1 hurt and
   the US data comes out. That is the concrete "does it break the weights" test.
2. Hold out a US slice and evaluate separately. That number is the US-visitor capability, which
   does not exist today at any price.
3. Three seeds minimum. Seed variance is 4.3 points, so a gain under ~5 is not detectable.

## Synthetic data, current state

`lab/osm_merchants.py` + `lab/synth_nl.py` are written and run. 6,000 rows, 665 distinct real
merchants from OpenStreetMap Noord-Holland (ODbL), templated into the nine bank formats found in
the real export (BEA Apple Pay, BEA Betaalpas, eCom Betaalpas, SEPA Incasso, SEPA Overboeking,
SEPA iDEAL, Wise Sent, Wise card, ABN AMRO fee). Merchants present in the test split are excluded
so the benchmark stays honest.

### Still to generate

Three categories have no merchant to draw on and need their own generators:

- **`internal_transfer`** — not defined by the counterparty name. Define it by **pairing**: a
  debit and a credit of the same or near-same amount within about a week, across two of the
  owner's accounts. Generate as pairs, not as single rows.
- **`income`** — salary, refunds, and unnamed inbound transfers. The reliable signal is the
  **sign**: it is the only consistently positive category (26 of 26 positive in the real export).
  Note internal_transfer is also 58% positive, so sign separates income from the eleven spending
  categories but not from transfers; the pairing rule above is what disambiguates.
- **`fees`** — the bank charging itself, no merchant. Format is
  `ABN AMRO Bank N.V. 7 x Betaalautomaat VV 1,05`.

### Dropped

- **`is_recurring`** — never moved off the 87.9% always-false rate in any run. Train `category`
  only.
- Merchant glossary in the prompt — invalid on merchant-disjoint splits: a glossary of training
  merchants cannot help on test merchants, and one that does help is reading the answer key.
- MoneyData / Mendeley `dnxtg6n4rv` — real UK data but one person's account, 44% of rows are
  `Savings`/`Cash`/`Amazon`, and UK strings are truncated to 18 characters, nothing like SEPA.

### Deferred

- Distillation from the 14B teacher with soft targets. Worth +3 to 8 by estimate, but the gap to
  the teacher is 14 points balanced and only 2 points top-3, so the 4B already ranks well and
  only mis-orders the top.
- `needs_review` as an output. Implement as a rule outside the model, not a trained class:
  `amount == 0` or `max(probability) < threshold`. There are zero zero-amount rows in the data,
  so this guards a hypothetical.

## Why 13 categories is the right choice, and what the demo actually lacks

The Choice type supports 2 to 255 candidates and single-token constrained decoding is **flat in
K**: 13 options cost the same 248 ms as 4, because they all live in one logit vector. Demoing at
13 exercises that property; demoing at 4 hides it.

The reason published Jev demos report higher numbers is not the option count, it is the kind of
task:

| | Rule task (Ed Huang's fraud triage) | Knowledge task (this) |
| --- | --- | --- |
| What decides the answer | a rule applied to stated facts | world knowledge about merchants |
| Is it in the prompt? | yes, the rule is in the description | no, nothing says Kruidvat is a drugstore |
| Learnable ceiling | ~100%, and `block_account` hit exactly that | capped by what the model knows |

A field that reaches exactly 100% is a deterministic function of stated inputs: the model learned
the rule. Nothing teaches a 0.6B what Vomar is, which is why the zero-shot scaling curve here is
6.4 -> 27.0 -> 48.2 -> 69.3 by model size alone. So this task will never produce a 100%, and that
is a property of the problem, not a mistake in setting it up.

**The demo's real gap is that it predicts one field.** Jev's contract is several typed fields in
one forward pass. Worth adding, using all three decision types:

- `direction` (money in / out) — Boolean, trivial, but shows parallel multi-field output
- `needs_review` — Boolean, from the confidence-threshold rule
- `amount_band` — Score, ordered levels with an expected value

Only the Choice field is hard; the rest demonstrate the contract.

**Lead with the right numbers.** Balanced 59.7 against a 7.7 random baseline is nearly 8x, and
top-3 of 94.0% reads well. "74% correct on merchants it has never seen, from 453 of my own
transactions, trained in 11 minutes" is a stronger claim than a boolean at 100%.
