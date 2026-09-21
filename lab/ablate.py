#!/usr/bin/env python3
"""Ablation: blank one input field at a time and measure how far accuracy falls.

A model that scores well can be reading the merchant, or it can be riding a shortcut such as "anything near 2,300 EUR is rent". Removing a field and re-scoring the same checkpoint says which fields actually carry the signal. No training, inference only.

    python lab/ablate.py --adapter runs/s0_ctrl_17/adapter --split data/lunchmoney/test.jsonl
"""
import argparse
import collections
import json
import re
import string
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """Below is a bank transaction.

{state}

Which budget category does it belong to?

{options}

Reply with the single letter only."""

# Each ablation blanks the named lines, keeping the field label so the shape stays constant.
ABLATIONS = {
    "full": [],
    "no_description": ["Bank description"],
    "no_payee": ["Payee"],
    "no_merchant_text": ["Bank description", "Payee"],
    "no_amount": ["Amount"],
    "no_account": ["Account"],
    "no_date": ["Date"],
    "only_merchant_text": ["Amount", "Date", "Account"],
}


def blank(state, fields):
    out = []
    for line in state.split("\n"):
        label = line.split(":", 1)[0]
        out.append(f"{label}: (withheld)" if label in fields else line)
    return "\n".join(out)


@torch.no_grad()
def score(model, tokenizer, records, criteria, letters, letter_ids, fields, batch_size):
    ids = list(criteria)
    options = "\n".join(f"{letters[i]}. {criteria[k]}" for i, k in enumerate(ids))
    per_class = collections.defaultdict(lambda: [0, 0])
    top1 = top3 = 0
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(state=blank(r["state"], fields),
                                                       options=options)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        scores = model(**enc).logits[:, -1, :][:, letter_ids].float()
        order = scores.argsort(dim=-1, descending=True).tolist()
        for rank, r in zip(order, chunk):
            gold = ids.index(r["gold"]["category"])
            hit = rank[0] == gold
            top1 += hit
            top3 += gold in rank[:3]
            per_class[gold][0] += hit
            per_class[gold][1] += 1
    balanced = sum(h / n for h, n in per_class.values()) / len(per_class)
    return {"top1": top1 / len(records), "top3": top3 / len(records), "balanced": balanced}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--split", default="data/lunchmoney/test.jsonl")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default="lab/ablation.json")
    args = p.parse_args()

    records = [json.loads(l) for l in Path(args.split).read_text().splitlines() if l.strip()]
    criteria = records[0]["questions"]["category"]["criteria"]
    letters = string.ascii_uppercase[:len(criteria)]

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in letters]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    results = {}
    for name, fields in ABLATIONS.items():
        results[name] = score(model, tokenizer, records, criteria, letters,
                              letter_ids, set(fields), args.batch_size)
        r = results[name]
        drop = results["full"]["balanced"] - r["balanced"]
        print(f"{name:20s} top1={r['top1']:6.1%} balanced={r['balanced']:6.1%} "
              f"top3={r['top3']:6.1%}  balanced drop={drop:+.3f}", flush=True)
        Path(args.output).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
