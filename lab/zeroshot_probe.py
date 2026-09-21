#!/usr/bin/env python3
"""Zero-shot category accuracy across model sizes, no training of any kind.

Shows the model a transaction plus the lettered category list and reads the next-token logits at those letters. Measures what the base model already knows about merchants, which is the ceiling a fine-tune starts from.
"""
import argparse
import collections
import json
import string
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """Below is a bank transaction.

{state}

Which budget category does it belong to?

{options}

Reply with the single letter only."""


def build(records, letters):
    criteria = records[0]["questions"]["category"]["criteria"]
    ids = list(criteria)
    options = "\n".join(f"{letters[i]}. {criteria[k]}" for i, k in enumerate(ids))
    items = []
    for row in records:
        items.append({
            "prompt": PROMPT.format(state=row["state"], options=options),
            "gold": ids.index(row["gold"]["category"]),
        })
    return items, ids


def letter_token_ids(tokenizer, letters):
    """One token per letter, or the run is measuring the wrong thing."""
    out = []
    for ch in letters:
        enc = tokenizer.encode(ch, add_special_tokens=False)
        if len(enc) != 1:
            raise SystemExit(f"letter {ch!r} is {len(enc)} tokens, not 1")
        out.append(enc[0])
    return out


@torch.no_grad()
def run(model_id, items, letters, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = letter_token_ids(tokenizer, letters)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    top1 = top3 = 0
    per_class = collections.defaultdict(lambda: [0, 0])
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for it in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        logits = model(**enc).logits[:, -1, :]
        scores = logits[:, ids].float()
        order = scores.argsort(dim=-1, descending=True)
        for row, it in zip(order, chunk):
            rank = row.tolist()
            hit = rank[0] == it["gold"]
            top1 += hit
            top3 += it["gold"] in rank[:3]
            per_class[it["gold"]][0] += hit
            per_class[it["gold"]][1] += 1
    del model
    torch.cuda.empty_cache()
    balanced = sum(h / n for h, n in per_class.values()) / len(per_class)
    return top1 / len(items), top3 / len(items), balanced


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/lunchmoney/test.jsonl")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default="lab/zeroshot_results.json")
    args = p.parse_args()

    records = [json.loads(l) for l in Path(args.split).read_text().splitlines() if l.strip()]
    letters = string.ascii_uppercase[:len(records[0]["questions"]["category"]["criteria"])]
    items, ids = build(records, letters)

    majority = max(set(i["gold"] for i in items), key=[i["gold"] for i in items].count)
    baseline = sum(i["gold"] == majority for i in items) / len(items)
    results = {"n": len(items), "categories": len(ids),
               "majority_baseline": baseline, "uniform": 1 / len(ids), "models": {}}
    print(f"n={len(items)} categories={len(ids)} majority={baseline:.1%} uniform={1/len(ids):.1%}\n")

    for model_id in args.models:
        acc, t3, bal = run(model_id, items, letters, args.batch_size)
        results["models"][model_id] = {"top1": acc, "top3": t3, "balanced": bal}
        print(f"{model_id:<24s} top1={acc:6.1%}  top3={t3:6.1%}  balanced={bal:6.1%}", flush=True)
        Path(args.output).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
