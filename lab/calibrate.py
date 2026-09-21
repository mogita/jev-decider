#!/usr/bin/env python3
"""Fit a temperature on dev and report expected calibration error on test.

Accuracy says whether the top choice is right. Calibration says whether the number next to it means anything: of the answers claiming 80% confidence, do 80% turn out correct? A model can be accurate and badly calibrated, and a confidence shown to a user is a claim that has to hold.

    python lab/calibrate.py --model models/decider-4b-bf16
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from serve_mlx import MlxDecider


def logits(decider, records):
    """Raw scores at the candidate letters, plus the gold index, one row at a time."""
    from serve_decision import render
    rows = []
    for record in records:
        tokens = decider.tokenizer.apply_chat_template(
            [{"role": "user", "content": render(None, record["state"], decider.options)}],
            add_generation_prompt=True, enable_thinking=False)
        scores = decider.model(mx.array([tokens]))[0, -1][decider.letter_ids]
        rows.append((scores.astype(mx.float32).tolist(),
                     decider.ids.index(record["gold"]["category"])))
    return rows


def softmax(values, temperature):
    top = max(v / temperature for v in values)
    exp = [math.exp(v / temperature - top) for v in values]
    total = sum(exp)
    return [e / total for e in exp]


def nll(rows, temperature):
    return -sum(math.log(max(softmax(s, temperature)[g], 1e-12)) for s, g in rows) / len(rows)


def ece(rows, temperature, bins=15):
    """Expected calibration error: average gap between confidence and accuracy, per bin."""
    buckets = [[0, 0.0, 0] for _ in range(bins)]
    for scores, gold in rows:
        probs = softmax(scores, temperature)
        best = max(range(len(probs)), key=lambda i: probs[i])
        index = min(int(probs[best] * bins), bins - 1)
        buckets[index][0] += best == gold
        buckets[index][1] += probs[best]
        buckets[index][2] += 1
    total = sum(b[2] for b in buckets)
    return sum(abs(hits / n - conf / n) * n / total for hits, conf, n in buckets if n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/decider-4b-bf16")
    p.add_argument("--data-dir", default="data/lunchmoney")
    args = p.parse_args()

    def load(split):
        return [json.loads(l) for l in
                Path(args.data_dir, f"{split}.jsonl").read_text().splitlines() if l.strip()]

    dev, test = load("dev"), load("test")
    decider = MlxDecider(args.model, dev[0]["questions"]["category"]["criteria"], 1.0)
    dev_rows, test_rows = logits(decider, dev), logits(decider, test)

    grid = [0.5 + 0.05 * i for i in range(71)]
    best = min(grid, key=lambda t: nll(dev_rows, t))
    accuracy = sum(max(range(len(s)), key=lambda i: s[i]) == g for s, g in test_rows) / len(test_rows)
    print(json.dumps({
        "dev_rows": len(dev_rows), "test_rows": len(test_rows),
        "test_top1": round(accuracy, 4),
        "temperature": round(best, 2),
        "test_ece_raw": round(ece(test_rows, 1.0), 4),
        "test_ece_scaled": round(ece(test_rows, best), 4),
        "dev_nll_raw": round(nll(dev_rows, 1.0), 4),
        "dev_nll_scaled": round(nll(dev_rows, best), 4),
    }, indent=1))


if __name__ == "__main__":
    main()
