#!/usr/bin/env python3
"""Convert a Lunch Money CSV export into the decision records the trainers here read.

One record per transaction, two questions: category (choice) and is_recurring (boolean). Splits are merchant-disjoint so a payee never appears on both sides of an evaluation.
"""
import argparse
import csv
import collections
import datetime
import json
import re
from pathlib import Path

SPLITS = ("train", "dev", "calibration", "test", "ood")
TARGETS = {"train": 0.65, "dev": 0.15, "test": 0.20}


def normalize_merchant(text):
    text = re.sub(r"\d", "", text.upper())
    text = re.sub(r"[^A-Z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:40] or "UNKNOWN"


def slug(category):
    value = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
    return value or "other"


def stratify(rows, ood_merchants):
    """Assign whole merchants to splits so each category keeps its share in every split.

    A plain hash of the payee keeps merchants disjoint but ignores labels, which produced a dev split with 10% groceries against train's 41%. Merchants stay whole here; only which split they land in is chosen, greedily, worst-represented category first.
    """
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["merchant"]].append(row)

    wanted = {}
    for name, share in TARGETS.items():
        for category, total in collections.Counter(
                r["category"] for r in rows if r["merchant"] not in ood_merchants).items():
            wanted[(name, category)] = total * share

    have = collections.defaultdict(float)
    assignment = {m: "ood" for m in ood_merchants}
    # Largest merchants first: they constrain the balance most, so place them while there is room.
    for merchant, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if merchant in ood_merchants:
            continue
        counts = collections.Counter(r["category"] for r in members)
        # Deficit = how far below quota a split is for the categories this merchant carries.
        best = max(TARGETS, key=lambda s: (
            sum(min(n, max(0.0, wanted[(s, c)] - have[(s, c)])) for c, n in counts.items()),
            -sum(have[(s, c)] for c in counts),
            s))
        assignment[merchant] = best
        for category, n in counts.items():
            have[(best, category)] += n
    return assignment


def state_text(row, max_description):
    day = datetime.date.fromisoformat(row["date"]).strftime("%A")
    return (f"Bank description: {row['original_name'][:max_description]}\n"
            f"Payee: {row['payee'][:max_description]}\n"
            f"Amount: {row['amount']} {row['currency'].upper()}\n"
            f"Date: {row['date']} ({day})\n"
            f"Account: {row['account_name']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output-dir", required=True)
    # Tiny categories score near zero on real data alone, but synthetic data now covers them, so keeping them costs little and the taxonomy stays the user's own.
    p.add_argument("--min-examples", type=int, default=1, help="Drop categories rarer than this")
    p.add_argument("--drop-category", action="append", default=[],
                   help="Drop this category outright, repeatable")
    # A category that exists in the account but has no transactions yet still belongs in the option list; other stages (US data) supply its training examples.
    p.add_argument("--extra-category", action="append", default=[],
                   help="Add a category with no rows in this export, repeatable")
    # 655 rows is too few to also fund an ood slice, and every candidate account shares merchants with the rest. The merchant-disjoint test split carries generalization.
    p.add_argument("--ood-account", default="", help="Hold this account out as the ood split")
    p.add_argument("--max-description", type=int, default=300)
    args = p.parse_args()

    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8-sig")))
    counts = collections.Counter(r["category"] for r in rows)
    drop = {c for c in counts if any(d.lower() in c.lower() for d in args.drop_category)}
    kept = sorted(c for c, n in counts.items() if n >= args.min_examples and c not in drop)
    dropped = {c: n for c, n in counts.items() if c not in kept}
    rows = [r for r in rows if r["category"] in kept]
    kept = sorted(set(kept) | set(args.extra_category))
    criteria = {slug(c): c for c in kept}
    if len(criteria) != len(kept):
        raise SystemExit("Category slugs collide; adjust slug()")

    # A merchant seen on the ood account is held out entirely, so no source group crosses splits.
    for row in rows:
        row["merchant"] = normalize_merchant(row["payee"])
    ood_merchants = {r["merchant"] for r in rows if r["account_name"] == args.ood_account}

    assignment = stratify(rows, ood_merchants)

    out = collections.defaultdict(list)
    for row in rows:
        split = assignment[row["merchant"]]
        uid = f"txn_{row['transaction_id']}"
        out[split].append({
            "id": uid,
            "state_id": uid,
            "family_id": "lunchmoney_categorize",
            "split": split,
            "state": state_text(row, args.max_description),
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Assign this transaction to exactly one budget category.",
                    "criteria": criteria,
                },
                "is_recurring": {
                    "type": "boolean",
                    "instructions": "This merchant charges this account on a regular repeating schedule.",
                },
            },
            "gold": {"category": slug(row["category"]), "is_recurring": bool(row["recurring_id"])},
            "metadata": {"source_group_id": f"merchant:{row['merchant']}"},
        })

    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out[split])
        (directory / f"{split}.jsonl").write_text(text, encoding="utf-8")

    print(json.dumps({
        "categories": len(kept),
        "dropped_categories": dropped,
        "rows_by_split": {s: len(out[s]) for s in SPLITS},
        "train_questions": len(out["train"]) * 2,
        "max_state_chars": max(len(r["state"]) for v in out.values() for r in v),
        "recurring_positive_rate": round(
            sum(r["gold"]["is_recurring"] for v in out.values() for r in v) / len(rows), 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
