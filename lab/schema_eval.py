#!/usr/bin/env python3
"""Measure the decider on schemas it was never trained on.

The tuning taught one 17-way budget taxonomy. Serving a caller's own enum is cheap to allow, but allowing it says nothing about whether it works, and a demo that offers a knob nobody measured is a demo that lies quietly. This runs public classification sets through the same single-forward-pass path with the caller's labels, and reports accuracy and calibration.

Comparing the tuned model against the frozen base answers the question that matters: did fitting a budget taxonomy cost the general ability the base model already had?

    python lab/schema_eval.py --model models/decider-4b-bf16 --task ag_news --limit 300
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

TASKS = {
    "ag_news": {
        "path": "fancyzhx/ag_news", "split": "test", "text": "text", "label": "label",
        "criteria": {"world": "World news", "sports": "Sports",
                     "business": "Business", "scitech": "Science and technology"},
        "question": "Which section of the newspaper does this story belong to?",
        "noun": "news story",
    },
    "emotion": {
        "path": "dair-ai/emotion", "split": "test", "text": "text", "label": "label",
        "criteria": {"sadness": "Sadness", "joy": "Joy", "love": "Love",
                     "anger": "Anger", "fear": "Fear", "surprise": "Surprise"},
        "question": "Which emotion does the writer express?",
        "noun": "message",
    },
}

TEMPLATE = """Below is a {noun}.

{{state}}

{question}

{{options}}

Reply with the single letter only."""

# What a two-field API has to use, since the caller supplies no wording of its own.
GENERIC = """Below is a record.

{state}

Which of the following does it belong to?

{options}

Reply with the single letter only."""


def ece(rows, bins=15):
    buckets = [[0, 0.0, 0] for _ in range(bins)]
    for confidence, correct in rows:
        index = min(int(confidence * bins), bins - 1)
        buckets[index][0] += correct
        buckets[index][1] += confidence
        buckets[index][2] += 1
    total = len(rows)
    return sum(abs(hits / n - conf / n) * n / total for hits, conf, n in buckets if n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/decider-4b-bf16")
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--generic", action="store_true",
                   help="Use the neutral prompt a two-field API must use, not a task-specific one")
    p.add_argument("--data-dir", default="data/lunchmoney")
    p.add_argument("--output")
    args = p.parse_args()

    from datasets import load_dataset
    from serve_mlx import MlxDecider

    task = TASKS[args.task]
    data = load_dataset(task["path"], split=task["split"])
    names = data.features[task["label"]].names
    if len(names) != len(task["criteria"]):
        raise SystemExit(f"{args.task} has {len(names)} labels, criteria has {len(task['criteria'])}")
    ids = list(task["criteria"])
    template = (GENERIC if args.generic
                else TEMPLATE.format(noun=task["noun"], question=task["question"]))

    # Balanced subsample, so a skewed test set cannot flatter or punish the score.
    per_class = max(1, args.limit // len(names))
    picked, seen = [], collections.Counter()
    for row in data:
        label = row[task["label"]]
        if seen[label] < per_class:
            picked.append(row)
            seen[label] += 1
        if len(picked) >= per_class * len(names):
            break

    reference = [json.loads(l) for l in
                 Path(args.data_dir, "test.jsonl").read_text().splitlines() if l.strip()]
    decider = MlxDecider(args.model, reference[0]["questions"]["category"]["criteria"], 1.0)

    calibration, per_label = [], collections.defaultdict(lambda: [0, 0])
    for row in picked:
        result = decider.decide(row[task["text"]][:2000], 1, template, task["criteria"])
        gold = ids[row[task["label"]]]
        correct = result["decision"] == gold
        calibration.append((result["confidence"], correct))
        per_label[gold][0] += correct
        per_label[gold][1] += 1

    report = {
        "model": args.model, "task": args.task, "rows": len(picked),
        "prompt": "generic" if args.generic else "task-specific",
        "classes": len(names),
        "top1": round(sum(c for _, c in calibration) / len(calibration), 4),
        "balanced": round(sum(h / n for h, n in per_label.values()) / len(per_label), 4),
        "ece": round(ece(calibration), 4),
        "mean_confidence": round(sum(c for c, _ in calibration) / len(calibration), 4),
        "per_label": {k: round(v[0] / v[1], 3) for k, v in sorted(per_label.items())},
    }
    print(json.dumps(report, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
