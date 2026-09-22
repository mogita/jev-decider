#!/usr/bin/env python3
"""Put a run from lab/jevbench_eval.py next to the baselines jev-bench publishes, on the same sources.

The published macro numbers average 22 sources. This model can answer 17 of them, and the five it cannot are the ones with 28 to 151 options, where every model on that board scores well below its own average. Reading our 17-source macro against their 22-source macro would therefore flatter us by exactly the amount those five hurt everyone else, so every row here is recomputed over the same 17.

    python lab/jevbench_table.py runs/jevbench/clean runs/jevbench/base
    python lab/jevbench_table.py runs/jevbench/clean runs/jevbench/base --verdict
"""
import argparse
import json
import math
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


def predictions(run, name):
    path = Path(run, f"{name}.jsonl")
    return {row["id"]: row["decision"]
            for row in (json.loads(line) for line in path.read_text().split("\n") if line.strip())}


def mcnemar(gold, ours, theirs):
    """The exact paired test, which is the one these runs call for.

    Both models answered the same rows, so the rows they agree on carry no information about which
    is better and an unpaired test throws that away. Only the disagreements count: if the adapter
    changed nothing real, each flip should fall either way like a fair coin.
    """
    gained = sum(1 for i in gold if ours[i] == gold[i] and theirs[i] != gold[i])
    lost = sum(1 for i in gold if ours[i] != gold[i] and theirs[i] == gold[i])
    flips = gained + lost
    if not flips:
        return gained, lost, 1.0
    smaller = min(gained, lost)
    tail = sum(math.comb(flips, i) for i in range(smaller + 1)) / 2 ** flips
    return gained, lost, min(1.0, 2 * tail)


def verdicts(mine, base, data, manifest, sources, alpha=0.05):
    print("| Field | Base | This | Verdict | Flips won / lost | p |")
    print("| --- | ---: | ---: | --- | ---: | ---: |")
    for name in sources:
        rows = [json.loads(line) for line
                in Path(data, manifest[name]["files"]["test"]).read_text().split("\n")
                if line.strip()]
        gold = {row["id"]: row["label"] for row in rows}
        ours, theirs = predictions(mine, name), predictions(base, name)
        gained, lost, p = mcnemar(gold, ours, theirs)
        call = "same" if p >= alpha else ("better" if gained > lost else "worse")
        accuracy = lambda table: sum(table[i] == gold[i] for i in gold) / len(gold)
        print(f"| `{name}` | {accuracy(theirs):.3f} | {accuracy(ours):.3f} | {call} "
              f"| {gained} / {lost} | {p:.2g} |")


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
    p.add_argument("--data", default="data/jevbench")
    p.add_argument("--verdict", action="store_true",
                   help="with exactly two runs, call each source better, worse or the same by the "
                        "paired test, instead of leaving a reader to eyeball the difference")
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

    if args.verdict:
        if len(args.runs) != 2:
            raise SystemExit("--verdict compares exactly two runs: the one under test, then the base")
        manifest = json.loads(Path(args.data, "manifest.json").read_text())["sources"]
        print(f"## Verdict per source, {len(shared)} of 22\n")
        verdicts(args.runs[0], args.runs[1], args.data, manifest, shared)
        print()

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
