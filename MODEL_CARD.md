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

LoRA adapter that picks a budget category for a bank transaction in **one forward pass**. Categories are listed as lettered options; the answer is read from the logits at those letter tokens and softmaxed. Nothing is generated.

- An answer outside the list is impossible.
- Latency is flat in the number of categories. Seventeen cost what two cost.
- Returns a distribution, not a sentence.
- Max 26 categories per call, one letter each.

Demo: [mojev.mogita.rocks](https://mojev.mogita.rocks). Code: [github.com/mogita/jev-decider](https://github.com/mogita/jev-decider).

## jev-bench, 17 of 22 sources, 17,773 records

Base is `Qwen3-4B`, no adapter, same rows and prompts. Verdict is McNemar's exact test on the rows where the two disagree, p < 0.05.

| Source | Base | This | Verdict |
|---|---|---|---|
| `sms_spam` | 58.8% | **83.5%** | better |
| `chaosnli` | 59.2% | **66.2%** | better |
| `measuring_hate_speech` | 36.7% | **39.8%** | better |
| `strategyqa_grounded` | 81.1% | **83.1%** | better |
| `boolq` | 84.9% | **86.8%** | better |
| `helpsteer2_helpfulness` | 35.8% | 38.9% | better, p=0.049 |
| `sst5` | 47.6% | 50.4% | better, p=0.035 |
| `mnli` | 81.8% | 81.8% | same |
| `fever_evidence` | 91.5% | 91.6% | same |
| `mmlu` | 65.9% | 66.4% | same |
| `helpsteer2_verbosity` | 59.6% | 59.0% | same |
| `strategyqa_closed` | 62.3% | 63.2% | same |
| `yelp5` | 63.2% | 64.0% | same |
| `paws` | 77.8% | 77.2% | same |
| `arc_challenge` | 87.1% | 87.8% | same |
| `civil_comments` | 63.7% | **60.6%** | worse |
| `stsb` | 41.6% | **30.4%** | worse |

5 better, 2 at p just under 0.05, 8 same, 2 worse.

| Macro | Base | This | Verdict |
|---|---|---|---|
| Accuracy | 64.6% | **66.5%** | better |
| Accuracy, minus `sms_spam` | 65.0% | 65.4% | same |
| ECE, uncalibrated | 32.9% | **21.4%** | better |
| ECE, temperature fitted | 7.6% | **6.5%** | better |
| ECE, fitted, minus `chaosnli` + `measuring_hate_speech` | **4.8%** | 5.7% | worse |
| TVD to human label distributions | 47.6% | **42.9%** | better |
| Brier (0 to 2, not a percentage) | 0.671 | **0.524** | better |

Per-source verdicts are the test. Macro verdicts are direction only, except `minus sms_spam`, which is +0.45 points and inside the noise.

| Model | Accuracy | ECE | Brier |
|---|---|---|---|
| **This adapter, temperature fitted** | 66.5% | **6.5%** | 0.428 |
| **This adapter, uncalibrated** | 66.5% | 21.4% | 0.524 |
| Jev 1.13.0 (commercial API) | 74.0% | 10.4% | **0.323** |
| Qwen3.5-4B, LoRA + residual heads | **74.7%** | 9.7% | 0.324 |
| Qwen3.5-9B, frozen | 70.1% | 9.5% | 0.354 |
| Qwen3.5-4B, frozen | 67.0% | 8.7% | 0.373 |

- Behind on accuracy on 14 of 17 sources. Best ECE in the table, worst Brier.
- The 6.5% needs ~500 labeled rows per task to fit a temperature. With no labeled data you get **21.4%**. Accuracy is identical either way.
- The 5 skipped sources need 28 to 151 options against the 26-letter ceiling.
- `chaosnli` ships no validation split, so neither model was fitted on it.
- The Qwen3.5 rows are a later base generation.

## Transaction categorization, what it was trained for

| Measure | Base | This |
|---|---|---|
| Dutch holdout, top-1 | 64.1% | **83.6%** |
| Dutch holdout, top-3 | — | 94.5% |
| Dutch holdout, balanced | — | 69.7% |
| US test, top-1 | — | 96.3% |
| US test, balanced | — | 96.0% |
| ECE, 15 bins, uncalibrated | — | 11.7% |
| AG News, caller's own labels | — | 87.0% |
| dair-ai/emotion, caller's own labels | — | 46.3% |

Merchant-disjoint: no merchant is in both training and test. The Dutch holdout is 128 rows, so a few points is noise.

Latency: 201 ms median, batch 1, Apple M4 Max via MLX.

## Usage

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

Pass any 2 to 26 labels for your own categories. That path rides the base model's ability, not anything taught here, and is not covered by the numbers above.

## Training

- LoRA r=16, alpha=32, dropout 0.05, all 7 attention and MLP projections, bf16 frozen base. 33.0M trainable of 4.06B, 0.81%.
- Loss is cross-entropy over the candidate letter logits at one position, the same normalization used at inference.
- Stage 1: 20,000 rows [us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2), 1 epoch, lr 1e-4.
- Stage 2: 8,000 synthetic Dutch rows, in this repo, 1 epoch, lr 7e-5.
- Stage 3: 452 real de-identified Dutch rows, 6 epochs, lr 3e-5.
- Class-weighted loss capped at 6x, checkpoint on dev balanced accuracy, seed 17. 106 min on one A40, $0.92.

## Data

- Stage 3 is a personal bank export and is **not published**. Every identifier was replaced with a shape-preserving stand-in before training: twelve digits became different twelve digits, names became generic names. Merchant names were kept, being the signal. Verified across all 681 rows that no original value survives.
- Loss touches one position, over the letters, so no gradient reaches the transaction text and the model is never asked to reproduce a record. 40 continuation probes recovered nothing. The same probe also recovers nothing from a model trained on the *unscrubbed* rows, so it is consistent with that argument rather than independent evidence. The values were not in the training data.
- The synthetic Dutch set here was built from [OpenStreetMap](https://www.openstreetmap.org/copyright) merchant names with invented account details. It shares no IBAN, card number, account label, mandate or name with the private export.
- `data/synth_train.jsonl` is a derivative database of OpenStreetMap and is licensed [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), © OpenStreetMap contributors. The adapter weights are Apache 2.0, matching the Qwen3-4B base.

## Limitations

- Overconfident without calibration: ECE 11.7% on its own holdout, 21.4% on jev-bench.
- 17 categories fixed at training time. Other label sets work but are not covered by the numbers above.
- At most 26 categories per request. One field per call.
- Not a chat model. Asking it to generate text gets the base model's behavior.
- Dutch and US transaction formats. Other locales untested.
- One seed. No variance measured.
- Untested above 26 options, which is where the interesting failures are.
