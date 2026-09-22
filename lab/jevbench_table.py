#!/usr/bin/env python3
"""Put a run from lab/jevbench_eval.py next to the baselines jev-bench publishes, on the same sources.

The published macro numbers average 22 sources. This model can answer 17 of them, and the five it cannot are the ones with 28 to 151 options, where every model on that board scores well below its own average. Reading our 17-source macro against their 22-source macro would therefore flatter us by exactly the amount those five hurt everyone else, so every row here is recomputed over the same 17.

    python lab/jevbench_table.py runs/jevbench/clean runs/jevbench/base
"""
import argparse
import json
from pathlib import Path

# The published files name their columns differently; this is the same quantity under both names.
PUBLISHED = {"acc": "accuracy", "ece": "ece", "brier": "brier", "nll": "nll",
             "sel@90": "selective_acc_at_90", "sel@50": "selective_acc_at_50", "aurc": "aurc",
             "rps": "rps", "mae": "mae", "auroc": "auroc", "tvd": "tvd_to_human"}

BASELINES = {"jev-1.13.0": "Jev 1.13.0, the commercial API",
             "qwen35-4b-t2": "Qwen3.5-4B, LoRA and residual heads",
             "qwen35-9b": "Qwen3.5-9B, frozen",
             "qwen35-4b": "Qwen3.5-4B, frozen"}


def published(directory, sources, primitives):
    raw = json.loads(Path(directory, "test_metrics.json").read_text())
    out = {}
    for name in sources:
        row = raw.get(name)
        if row is None:
            return None
        out[name] = {ours: row.get(theirs) for ours, theirs in PUBLISHED.items()}
        out[name]["primitive"] = primitives[name]
    return out


def macro(sources):
    families = {}
    for result in sources.values():
        families.setdefault(result["primitive"], []).append(result["acc"])
    out = {key: sum(r[key] for r in sources.values()) / len(sources)
           for key in ("acc", "ece", "brier", "sel@90")}
    out.update({kind: sum(v) / len(v) for kind, v in families.items()})
    tvd = [r["tvd"] for r in sources.values() if r.get("tvd") is not None]
    out["tvd"] = sum(tvd) / len(tvd) if tvd else None
    return out


def cell(value):
    return "" if value is None else f"{value:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="directories written by lab/jevbench_eval.py")
    p.add_argument("--results", default="data/jevbench/results")
    p.add_argument("--label", nargs="*", default=[], help="a name per run, in the same order")
    args = p.parse_args()

    ours = []
    for index, run in enumerate(args.runs):
        report = json.loads(Path(run, "report.json").read_text())
        name = args.label[index] if index < len(args.label) else report["model"]
        ours.append((name, report["sources"]))
    shared = sorted(set.intersection(*(set(sources) for _, sources in ours)))
    primitives = {name: ours[0][1][name]["primitive"] for name in shared}

    table = [(name, {k: v for k, v in sources.items() if k in shared}) for name, sources in ours]
    for directory, label in BASELINES.items():
        rows = published(Path(args.results, directory), shared, primitives)
        if rows:
            table.append((label, rows))

    print(f"## Per source, {len(shared)} of 22\n")
    header = " | ".join(name for name, _ in table)
    print(f"| Source | K | {header} |")
    print("| --- | --- | " + " | ".join("---:" for _ in table) + " |")
    for name in shared:
        cells = " | ".join(cell(sources[name]["acc"]) for _, sources in table)
        print(f"| `{name}` | {primitives[name]} | {cells} |")
    print("\n## Macro over those sources\n")
    summaries = [(name, macro(sources)) for name, sources in table]
    keys = ["acc", "ece", "brier", "sel@90", "choice", "score", "noul", "tvd"]
    print("| Model | " + " | ".join(keys) + " |")
    print("| --- | " + " | ".join("---:" for _ in keys) + " |")
    for name, summary in summaries:
        print(f"| {name} | " + " | ".join(cell(summary.get(key)) for key in keys) + " |")


if __name__ == "__main__":
    main()
