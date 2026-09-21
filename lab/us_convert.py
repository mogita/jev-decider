#!/usr/bin/env python3
"""Convert the US synthetic bank dataset into our training record format.

Source: DoDataThings/us-bank-transaction-categories-v2, 68k rows, description + category only. It carries no amount, so one is generated per row with the sign taken from the [debit]/[credit] prefix. That prefix reproduces the real sign semantics closely: Income is 100% credit, Transfer 36% credit, every spending category 2-7% credit (refunds).

This is stage 1 of training and teaches the task shape, not the format. Dutch formats come from the synthetic stage and the real export, which trains last.

    python lab/us_convert.py --csv transactions-synthetic.csv --output data/us/train.jsonl
"""
import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

# US taxonomy -> ours. Mortgage and Rent collapse; Donation has no US source.
CATEGORY_MAP = {
    "Groceries": "groceries", "Shopping": "shopping", "Personal Care": "personal_care",
    "Subscription": "subscriptions", "Utilities": "utilities", "Insurance": "insurance",
    "Entertainment": "entertainment", "Transportation": "transportation", "Fees": "fees",
    "Income": "income", "Transfer": "internal_transfer", "Mortgage": "rent_loan",
    "Rent": "rent_loan", "Restaurants": "restaurants", "Travel": "travel",
    "Education": "education", "Healthcare": "healthcare",
}

AMOUNTS = {
    "groceries": (4, 190), "shopping": (5, 600), "personal_care": (4, 140),
    "transportation": (2, 160), "entertainment": (6, 120), "utilities": (15, 340),
    "insurance": (25, 500), "subscriptions": (3, 60), "rent_loan": (600, 3600),
    "income": (25, 9000), "internal_transfer": (20, 6000), "fees": (1, 40),
    "restaurants": (6, 180), "travel": (30, 1800), "education": (15, 2000),
    "healthcare": (8, 900),
}

US_ACCOUNTS = ["Chase Checking 4412", "Capital One 360 8827", "Bank of America 1903",
               "Wells Fargo Everyday 5510", "Ally Interest Checking 7734"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PROMPT_STATE = ("Bank description: {desc}\nPayee: {payee}\nAmount: {amount} USD\n"
                "Date: 2026-{m:02d}-{d:02d} ({day})\nAccount: {account}")


def payee_from(description):
    """Best-effort merchant string: the leading words before codes and numbers take over."""
    head = re.split(r"\s{2,}|\bPPD ID:|\bWEB ID:|\bID:\s", description)[0]
    head = re.sub(r"\b\d{3,}\b.*$", "", head).strip(" -:*#")
    return (head or description)[:60]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--reference", default="data/lunchmoney/train.jsonl",
                   help="supplies the question block so prompts match the other stages exactly")
    p.add_argument("--output", required=True)
    p.add_argument("--holdout", help="write a held-out US slice here for separate evaluation")
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    random.seed(args.seed)

    template = json.loads(Path(args.reference).read_text().splitlines()[0])
    questions = template["questions"]
    allowed = set(questions["category"]["criteria"])

    rows, skipped = [], collections.Counter()
    for source in csv.DictReader(Path(args.csv).open(encoding="utf-8-sig")):
        raw = source["description"].strip()
        category = CATEGORY_MAP.get(source["category"])
        if category is None or category not in allowed:
            skipped[source["category"]] += 1
            continue
        credit = raw.lower().startswith("[credit]")
        desc = re.sub(r"^\[(debit|credit)\]\s*", "", raw, flags=re.I)
        low, high = AMOUNTS[category]
        amount = round(random.uniform(low, high), 2)
        uid = f"us_{len(rows):06d}"
        rows.append({
            "id": uid, "state_id": uid, "family_id": "us_bank", "split": "train",
            "state": PROMPT_STATE.format(desc=desc, payee=payee_from(desc),
                                         amount=amount if credit else -amount,
                                         m=random.randint(1, 9), d=random.randint(1, 28),
                                         day=random.choice(DAYS),
                                         account=random.choice(US_ACCOUNTS)),
            "questions": questions,
            "gold": {"category": category},
            "metadata": {"source_group_id": f"us:{payee_from(desc).upper()[:30]}"},
        })

    random.shuffle(rows)
    holdout = []
    if args.holdout:
        # Split by merchant so the US holdout is merchant-disjoint like every other eval here.
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r["metadata"]["source_group_id"]].append(r)
        keys = list(groups)
        random.shuffle(keys)
        target = len(rows) * args.holdout_frac
        held = set()
        for k in keys:
            if sum(len(groups[h]) for h in held) >= target:
                break
            held.add(k)
        holdout = [r for k in held for r in groups[k]]
        for r in holdout:
            r["split"] = "test"
        rows = [r for r in rows if r["metadata"]["source_group_id"] not in held]

    for path, data in ((args.output, rows), (args.holdout, holdout)):
        if path and data:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data),
                           encoding="utf-8")

    print(json.dumps({
        "train_rows": len(rows), "holdout_rows": len(holdout),
        "skipped_categories": dict(skipped),
        "per_category": dict(collections.Counter(r["gold"]["category"] for r in rows).most_common()),
    }, indent=1))


if __name__ == "__main__":
    main()
