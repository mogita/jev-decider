#!/usr/bin/env python3
"""Score this model on jev-bench, the public System One benchmark, so its numbers can be placed next to other people's.

jev-bench ships 22 sources as (state, question, label) triples in the wire format a decision model consumes, and publishes the same columns for Jev 1.13.0 and for eighteen open checkpoints. Seventeen of the sources fit here; the other five ask for 28 to 151 options against the twenty-six a single letter can address.

Nothing about the deployment applies to a benchmark. The service caps a record at 4,000 characters and a label at 80 to keep one laptop answerable on a public URL, and a limit that exists to protect a machine says nothing about the model, so those caps are lifted here. The twenty-six letters are the one limit that is not a choice.

    python lab/jevbench_eval.py --model models/decider-4b-clean-bf16 --out runs/jevbench/clean
    python lab/jevbench_eval.py --model models/decider-4b-clean-bf16 --out /tmp/smoke --limit 20

Predictions land one file per source, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import math
import string
import time
from pathlib import Path

import mlx.core as mx

import serve_decision
from jevify.evaluation.metrics import (accuracy, brier_score, expected_calibration_error, log_loss,
                                       mean_absolute_error, ranked_probability_score)

# The benchmark supplies the wording of every question, so the template carries only the frame the
# model was tuned on and lets the source speak for itself.
TEMPLATE = """Below is a record.

{state}

{instructions}

{options}

Reply with the single letter only."""

LETTERS = len(string.ascii_uppercase)


def options_for(question):
    """The candidates as {label key: what the model reads}, in the order the benchmark declares them.

    A choice names its own keys, a score is a ladder addressed by level index, and a noul is the
    yes-or-no pair, which this model can only ask as a two-way choice. Where a key carries meaning
    of its own it stays in front of its description, since that is the wording jev-bench measured.
    """
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        if criteria:
            return {"1": criteria["true"], "0": criteria["false"]}
        return {"1": "Yes", "0": "No"}
    if kind == "score":
        return {str(i): text for i, text in enumerate(criteria)}
    return {key: (value if len(key) == 1 else f"{key}: {value}") for key, value in criteria.items()}


def state_text(raw):
    """States arrive as JSON. A bare string is the record itself; a structure is worth indenting, since the model reads it rather than parsing it."""
    value = json.loads(raw)
    return value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)


def selective_accuracy(correct, confidence, coverage):
    """Accuracy over the most confident slice, which is what a confidence-gated router actually gets."""
    order = sorted(range(len(correct)), key=lambda i: -confidence[i])
    keep = max(1, round(len(order) * coverage))
    return sum(correct[i] for i in order[:keep]) / keep


def aurc(correct, confidence):
    """Area under the risk-coverage curve: the mean error rate over every prefix of the confidence ranking."""
    order = sorted(range(len(correct)), key=lambda i: -confidence[i])
    errors, total = 0, 0.0
    for taken, i in enumerate(order, start=1):
        errors += not correct[i]
        total += errors / taken
    return total / len(order)


def auroc(labels, scores):
    """Rank based, with ties sharing their average rank, so a model that returns the same number twice is not rewarded for it."""
    positives = sum(labels)
    if not positives or positives == len(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    positive_rank = sum(r for r, label in zip(ranks, labels) if label)
    return (positive_rank - positives * (positives + 1) / 2) / (positives * (len(labels) - positives))


def total_variation(model, human):
    return 0.5 * sum(abs(model.get(key, 0.0) - value) for key, value in human.items())


def human_distribution(soft, keys, kind):
    """The vote shares behind a label, put in the same space as the model's distribution."""
    value = json.loads(soft)
    if kind == "noul":
        return {"1": value, "0": 1.0 - value}
    if isinstance(value, list):
        return {str(i): p for i, p in enumerate(value)}
    return value


def score_source(rows, predictions, kind):
    probs = [p["probs"] for p in predictions]
    labels = [row["label"] for row in rows]
    selected = [p["decision"] for p in predictions]
    correct = [s == label for s, label in zip(selected, labels)]
    confidence = [max(p.values()) for p in probs]
    # Fifteen bins, because that is what the published baselines used: their reliability tables are
    # cut at 7/15 and 8/15, and an ECE compared across different bin counts is not a comparison.
    ece, _ = expected_calibration_error(probs, labels, bins=15)
    ece10, _ = expected_calibration_error(probs, labels, bins=10)
    out = {"n": len(rows), "primitive": kind,
           "acc": accuracy(selected, labels), "ece": ece, "ece10": ece10,
           "brier": brier_score(probs, labels), "nll": log_loss(probs, labels),
           "sel@90": selective_accuracy(correct, confidence, 0.9),
           "sel@50": selective_accuracy(correct, confidence, 0.5),
           "aurc": aurc(correct, confidence)}
    if kind == "score":
        levels = sorted(probs[0], key=int)
        ordered = [[p[level] for level in levels] for p in probs]
        indices = [int(label) for label in labels]
        out["rps"] = ranked_probability_score(ordered, indices)
        out["mae"] = mean_absolute_error(
            [sum(i * p for i, p in enumerate(row)) for row in ordered], indices)
    if kind == "noul":
        out["auroc"] = auroc([int(label) for label in labels], [p["1"] for p in probs])
    soft = [row["soft_label"] for row in rows]
    if all(soft):
        out["tvd"] = sum(total_variation(p, human_distribution(s, list(p), kind))
                         for p, s in zip(probs, soft)) / len(rows)
    return out


def rescale(row, temperature):
    """Temperature scaling, done on the stored distribution. A softmax is invertible up to a constant the next softmax removes, so the log of a saved probability stands in for the logit it came from and no forward pass has to be repeated."""
    keys = list(row)
    scaled = [math.log(max(row[key], 1e-12)) / temperature for key in keys]
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    return {key: weight / total for key, weight in zip(keys, weights)}


def fit_temperature(probs, labels, low=0.2, high=25.0, steps=80):
    """One scalar per source, chosen on the validation split, by ternary search on the loss it is meant to fix."""
    def loss(temperature):
        return -sum(math.log(max(rescale(row, temperature).get(label, 0.0), 1e-12))
                    for row, label in zip(probs, labels)) / len(labels)
    for _ in range(steps):
        third = (high - low) / 3
        if loss(low + third) < loss(high - third):
            high -= third
        else:
            low += third
    return (low + high) / 2


def lines(path):
    """str.splitlines also breaks on the vertical tab, the line separator and the next-line character, all of which appear inside these records, and a line cut in the middle of a JSON string does not parse."""
    return [line for line in path.read_text().split("\n") if line.strip()]


def load(path, limit):
    rows = [json.loads(line) for line in lines(path)]
    return rows[:limit] if limit else rows


def run_source(decider, rows, group, chunk, report):
    """Rows sharing a question and a candidate set ride one call; the rest is bookkeeping.

    Only rows with identical options can be scored together, because one call reads one logit
    vector per row against one set of candidates. Sources that vary their options per row, such as
    the exam questions, therefore fall back to groups of one, which costs the padding saving and
    nothing else.
    """
    batches = {}
    for index, row in enumerate(rows):
        question = json.loads(row["question"])
        options = options_for(question)
        batches.setdefault((question["instructions"], tuple(options.items())), []).append(index)
    out = [None] * len(rows)
    done = 0
    for (instructions, pairs), members in batches.items():
        options = dict(pairs)
        template = TEMPLATE.replace("{instructions}", instructions)
        for start in range(0, len(members), group):
            slice_ = members[start:start + group]
            results = decider.decide_batch([state_text(rows[i]["state"]) for i in slice_],
                                           template=template, criteria=options, chunk=chunk)
            for i, result in zip(slice_, results):
                out[i] = {"id": rows[i]["id"], "decision": result["decision"],
                          "probs": {a["category"]: a["p"] for a in result["alternatives"]}}
            done += len(slice_)
            report(done)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/decider-4b-clean-bf16")
    p.add_argument("--data", default="data/jevbench")
    p.add_argument("--out", required=True, help="directory for predictions and the report")
    p.add_argument("--split", default="test")
    p.add_argument("--sources", help="comma separated subset of the runnable sources")
    p.add_argument("--limit", type=int, default=0, help="first N rows of each source, for a smoke test")
    p.add_argument("--group", type=int, default=64, help="rows handed to the model per call")
    p.add_argument("--chunk", type=int, default=8, help="rows per forward pass inside a call")
    p.add_argument("--cache-gb", type=float, default=4.0,
                   help="ceiling on the MLX buffer cache; 0 leaves it unbounded")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--calibrate", help="a run over the validation split; one temperature is fitted "
                                       "per source there and applied here, which is what the "
                                       "published Tier 0 baselines did before they were scored")
    args = p.parse_args()

    # MLX keeps freed buffers in a cache keyed by size, and every batch here is padded to the longest
    # row in it, so a run over sources whose records range from one line to several thousand tokens
    # keeps minting shapes it never reuses. Left alone the cache reached 36 GB on a 48 GB machine and
    # pushed it into swap; the service never sees this because its records are all one shape.
    if args.cache_gb:
        mx.set_cache_limit(int(args.cache_gb * 2 ** 30))

    # The service guards protect one laptop on a public URL. A benchmark measures the model, so the
    # only cap left standing is the one the architecture imposes.
    serve_decision.MAX_STATE = 10 ** 7
    serve_decision.MAX_TEMPLATE = 10 ** 7
    serve_decision.MAX_LABEL = 10 ** 4

    manifest = json.loads(Path(args.data, "manifest.json").read_text())["sources"]
    runnable, skipped = [], {}
    for name, source in manifest.items():
        k = source["k"]
        if isinstance(k, int) and k > LETTERS:
            skipped[name] = k
        else:
            runnable.append(name)
    if args.sources:
        runnable = [name for name in args.sources.split(",") if name in runnable]

    # Loaded on the first source that still needs scoring, so re-reading a finished run to change a
    # metric does not spend eight gigabytes and half a minute faulting in weights it never uses.
    loaded = []

    def decider():
        if not loaded:
            from serve_mlx import MlxDecider
            loaded.append(MlxDecider(args.model, {"a": "a", "b": "b"}, args.temperature))
        return loaded[0]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "split": args.split, "limit": args.limit,
              "benchmark": json.loads(Path(args.data, "manifest.json").read_text())["version"],
              "skipped": skipped, "sources": {}}
    for name in runnable:
        path = manifest[name]["files"].get(args.split)
        if path is None:                      # ChaosNLI ships test only, all 1,599 rows of it
            continue
        rows = load(Path(args.data, path), args.limit)
        kind = json.loads(rows[0]["question"])["type"]
        cache = out / f"{name}.jsonl"
        started = time.perf_counter()
        if cache.exists():
            predictions = [json.loads(line) for line in lines(cache)]
        if not cache.exists() or len(predictions) != len(rows):
            def show(done, name=name, total=len(rows), started=started):
                rate = done / max(1e-9, time.perf_counter() - started)
                held = (mx.get_active_memory() + mx.get_cache_memory()) / 2 ** 30
                print(f"\r{name} {done}/{total} {rate:.1f} rows/s {held:.1f} GB",
                      end="", flush=True)
            predictions = run_source(decider(), rows, args.group, args.chunk, show)
            cache.write_text("".join(json.dumps(r) + "\n" for r in predictions))
            print()
        temperature = None
        if args.calibrate:
            held = Path(args.calibrate, f"{name}.jsonl")
            if held.exists():
                fit_rows = load(Path(args.data, manifest[name]["files"]["validation"]), args.limit)
                fit = [json.loads(line) for line in lines(held)]
                temperature = fit_temperature([p["probs"] for p in fit],
                                              [row["label"] for row in fit_rows])
                predictions = [{**p, "probs": rescale(p["probs"], temperature)} for p in predictions]
        report["sources"][name] = score_source(rows, predictions, kind)
        report["sources"][name]["temperature"] = temperature
        report["sources"][name]["seconds"] = round(time.perf_counter() - started, 1)
        print(name, json.dumps({k: round(v, 4) for k, v in report["sources"][name].items()
                                if isinstance(v, float)}))

    families = {}
    for name, result in report["sources"].items():
        families.setdefault(result["primitive"], []).append(result["acc"])
    report["macro"] = {
        "acc": sum(r["acc"] for r in report["sources"].values()) / len(report["sources"]),
        "ece": sum(r["ece"] for r in report["sources"].values()) / len(report["sources"]),
        "brier": sum(r["brier"] for r in report["sources"].values()) / len(report["sources"]),
        "sel@90": sum(r["sel@90"] for r in report["sources"].values()) / len(report["sources"]),
        **{f"{kind} acc": sum(v) / len(v) for kind, v in families.items()}}
    tvd = [r["tvd"] for r in report["sources"].values() if "tvd" in r]
    if tvd:
        report["macro"]["tvd"] = sum(tvd) / len(tvd)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["macro"], indent=1))


if __name__ == "__main__":
    main()
