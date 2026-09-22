---
base_model: Qwen/Qwen3-4B
library_name: peft
license: apache-2.0
pipeline_tag: text-classification
language:
  - en
  - nl
tags:
  - lora
  - constrained-decoding
  - transaction-categorization
  - calibration
---

# jev-decider, Qwen3-4B LoRA

A LoRA adapter that assigns a budget category to a bank transaction in **one forward pass**, and returns a probability distribution over the categories rather than a sentence.

Live demo: [mojev.mogita.rocks](https://mojev.mogita.rocks). Code, training scripts and evaluation: [github.com/mogita/jev-decider](https://github.com/mogita/jev-decider).

## How it answers

Nothing is generated. The categories are listed in the prompt as lettered options, the model runs a single forward pass, and the answer is read from the logits at the candidate letter tokens and softmaxed.

Two properties follow from that, and they are the reason this exists. An answer outside the list is impossible, because the only positions read are the letters offered. And response time does not grow with the number of categories, because they all live in one logit vector: seventeen categories cost exactly what two cost.

The cost is that this is not a chat model. It cannot explain itself, and asking it to generate text gets you the base model's behavior, not this adapter's.

## Using it

```python
import string, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto"),
    "mogita/jev-decider-qwen3-4b").eval()

CATEGORIES = ["🏥 Healthcare", "✈️ Travel", "🍽️ Restaurants", "🎓 Education",
              "🎟️ Entertainment", "🏦 Fees", "🐾 Pets", "💡 Utilities", "💵 Income",
              "🔄 Internal Transfer", "🔑 Rent, Loan", "🔔 Subscriptions",
              "🚗 Transportation", "🛍️ Shopping", "🛒 Groceries", "🛡️ Insurance",
              "🧴 Personal Care"]

PROMPT = """Below is a bank transaction.

{state}

Which budget category does it belong to?

{options}

Reply with the single letter only."""

def decide(record, categories=CATEGORIES):
    options = "\n".join(f"{c}. {label}" for c, label in zip(string.ascii_uppercase, categories))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(state=record, options=options)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    letters = [tokenizer.encode(c, add_special_tokens=False)[0]
               for c in string.ascii_uppercase[:len(categories)]]
    with torch.no_grad():
        logits = model(**ids).logits[0, -1, letters].float()
    probs = torch.softmax(logits, dim=-1)
    return categories[probs.argmax()], probs.max().item()

print(decide("Payee: Albert Heijn\nAmount: -37.82 EUR\nDate: 2026-03-12"))
# ('🛒 Groceries', 0.996)
```

**Bring your own categories.** Pass any 2 to 26 labels and it answers against those instead. That path rides the base model's general ability rather than anything taught here, and the headline numbers below do not cover it; the two measured examples are in the results table.

## Results

Measured on a **merchant-disjoint** Dutch holdout, so no merchant appears in both training and test, and on a public US test split.

| | |
| --- | ---: |
| Dutch holdout, top-1 | 83.6% |
| Dutch holdout, top-3 | 94.5% |
| Dutch holdout, balanced | 69.7% |
| US test, top-1 | 96.3% |
| US test, balanced | 96.0% |
| Frozen base, same Dutch holdout, top-1 | 64.1% |
| AG News, caller's own labels, never trained on | 87.0% |
| dair-ai/emotion, caller's own labels, never trained on | 46.3% |
| Latency, Apple M4 Max via MLX | 201 ms median |

**Calibration is the weak spot, and it is worse than the accuracy suggests.** Expected calibration error is **0.117** over 15 bins, untuned. The model is more confident than it should be, and fitting a temperature on the development split makes test calibration worse rather than better, which points at the split rather than the method. Treat the probability as a ranking signal, not as a number you can act on directly.

The Dutch holdout is 128 rows. Differences smaller than a few points there are noise.

## On a public benchmark

[jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) v0.1.1 scores decision models on 22 public sources and publishes the same columns for the commercial Jev 1.13.0 API and eighteen open checkpoints. Seventeen of those sources fit under the 26-candidate ceiling, 17,773 test records. Every macro figure below is recomputed over the same seventeen for every model, because the five that are skipped are where all of them score worst.

| Model | Macro acc | ECE | Brier | TVD to human |
| --- | ---: | ---: | ---: | ---: |
| **This adapter, one temperature fitted per source** | 0.665 | **0.065** | 0.428 | 0.411 |
| This adapter, untuned | 0.665 | 0.214 | 0.524 | 0.429 |
| `Qwen3-4B` frozen, same harness, temperature fitted | 0.646 | 0.076 | 0.484 | 0.454 |
| Jev 1.13.0, the commercial API | 0.740 | 0.104 | 0.323 | 0.350 |
| Qwen3.5-4B, LoRA and residual heads | 0.747 | 0.097 | 0.324 | 0.286 |
| Qwen3.5-4B, frozen | 0.670 | 0.087 | 0.373 | 0.349 |

**Tuning on one narrow schema did not cost general decision ability. It did not buy much either.** Against its own frozen base through the identical harness the adapter is ahead on every macro column, but the accuracy macro is one source: 11 of 17 improve, 4 regress, 2 are flat, and ten of the eleven gains are under 3.1 points. The eleventh is `sms_spam`, 0.588 to 0.835. Remove that single source and the accuracy gain falls from +1.9 points to **+0.45**, which is noise at these sizes. The worst regression is `stsb`, 0.416 to 0.304. The defensible claim is that a narrow adapter did no broad damage, which was not obvious beforehand; it is not evidence of general transfer.

The same decomposition applies to calibration and is less favorable. Untuned, the adapter is better calibrated than the base on 17 of 17 sources. Once both get a fitted temperature it is better on **7 of 17**, and the 0.065 against 0.076 rests on two sources; remove those two and the frozen base is better, 0.048 to 0.057. So: better calibrated than its base when neither is calibrated, about equal when both are.

Which ECE to quote depends on what you have. The published baselines were all fitted on validation splits before being scored, so **0.065 is the comparable figure** against that board. It also requires roughly 500 labeled rows per task to fit against. **If you have no labeled data for your schema, the number that describes what you get is 0.214.** Accuracy is identical either way, since temperature scaling cannot move an argmax.

Caveats that cut against these numbers rather than for them: the Qwen3.5 rows are a later base generation, so the clean comparison is against our own base and not against them; this model has one primitive, so every yes-or-no source was asked as a two-way choice, which jev-bench's own probes show is the worse geometry for calibration; `chaosnli` ships no validation split, so neither model could be fitted on it and its raw-versus-raw result sits inside a fitted table; this model is behind Jev on 14 of the 17 sources; and nothing here says anything about routing to 77 or 151 options, which is where the interesting failures live.

Harness and full per-source table: [`lab/jevbench_eval.py`](https://github.com/mogita/jev-decider) and `lab/RESULTS.md`.

## Training

Base `Qwen/Qwen3-4B`, frozen in bf16. LoRA rank 16, alpha 32, dropout 0.05, on all seven attention and MLP projections. 33.0M trainable parameters against a 4.06B base, 0.81%.

The loss is cross-entropy over the candidate letter logits at a single position, which is the same normalization used at inference, so nothing is optimized that is not also served.

Three stages, in order, with a decaying learning rate and the real data last:

| Stage | Data | Rows | Epochs | LR |
| --- | --- | ---: | ---: | ---: |
| 1 | [us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) | 20,000 | 1 | 1e-4 |
| 2 | Synthetic Dutch, included in this repo | 8,000 | 1 | 7e-5 |
| 3 | Real Dutch export, de-identified, not published | 452 | 6 | 3e-5 |

Class-weighted loss, capped at 6x, because four categories ranked correctly in the top three while almost never placing first. Checkpoint selected on development balanced accuracy. Seed 17. 106 minutes on one NVIDIA A40, about $0.92 of rented GPU.

## Data and privacy

Stage 3 is a personal bank export, and it is **not published**. Before training, every identifier in it was replaced with a shape-preserving stand-in: a twelve digit mandate number became a different twelve digits, account numbers and card digits and IBANs and payment references were regenerated in place, and personal names became generic names of the same shape. Merchant names were deliberately kept, since they are the signal. Verified across all 681 rows that no original value survives and that the set of value shapes is unchanged.

De-identified is not anonymous. Merchants, amounts and dates across nine months remain a behavioral record, which is why the rows stay private even in their cleaned form.

**On extraction.** The training objective computes cross-entropy at one position, over the letters, so no gradient ever reaches the transaction text and the model is never asked to reproduce a record. As a check rather than a substitute for that argument, 40 continuation probes cut real rows just before an identifier and looked for any original value the model supplied itself: none were recovered. The same probe recovers nothing from a model trained on the *unscrubbed* rows either, so treat the probe as consistent with the structural argument rather than as independent evidence. The decisive point for these weights is simpler: the values were never in their training data.

The synthetic Dutch set in this repo was generated from [OpenStreetMap](https://www.openstreetmap.org) merchant names with invented account details. It was checked against the private export and shares no IBAN, card number, account label, mandate or personal name with it.

## Limitations

- Seventeen categories, fixed at training time. Other label sets work but are not covered by the headline numbers.
- At most 26 categories per request, since each candidate is one letter.
- One field per call. It answers "which category", not several questions at once.
- Dutch and US transaction formats. Other locales are untested.
- One seed. No variance measured across seeds on this configuration.
- Overconfident, as above.

## License

Apache 2.0, matching the Qwen3-4B base. The adapter weights only; the base model is subject to its own license.
