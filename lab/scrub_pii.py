#!/usr/bin/env python3
"""Replace the identifiers in a real bank export with consistent stand-ins.

Weights trained on 452 real rows can memorize what is in them, and an adapter anyone can download
has none of the protection the serving path has, where the only thing that leaves the process is a
single letter. So the identifiers come out before training, not after.

Every distinct value maps to one stand-in and always the same one. The mapping has to be a
relabeling rather than noise: an account that signals a transfer has to keep signaling it, and the
splits are merchant-disjoint by payee, which is signal and is left alone.

The hard part is the `Naam:` field, which holds a merchant on one line and a person on the next.
Replacing a merchant there would delete the very thing the model reads, so the default is to keep
and the burden is on showing a value is a person: it carries dotted initials, or a Dutch
tussenvoegsel, or it is the account holder, who is found rather than configured, by reading the
names that appear on the transfers between the owner's own accounts.

    python lab/scrub_pii.py --in data/lunchmoney --out data/lunchmoney_clean

This de-identifies, it does not anonymize. Merchants, amounts and dates remain a real spending
history. The point is only that a model cannot give back an account number or a name.
"""
import argparse
import json
import random
import re
from pathlib import Path

SPLITS = ("train", "dev", "calibration", "test", "ood")

# Dutch bank statement fields, in the shapes the export writes them.
IBAN = re.compile(r"\b([A-Z]{2}\d{2})([A-Z]{4})([A-Z0-9]{4,20})\b")
CARD = re.compile(r"(KAARTNUMMER:\s*)(\*+)(\d+)", re.I)
ACCOUNT = re.compile(r"^(Account:\s*)(.+)$", re.M)
NAAM = re.compile(r"(Naam:\s*)(.+?)(?=\s+(?:Omschrijving|Machtiging|Kenmerk|IBAN|BIC):|$)", re.I)
REFS = (re.compile(r"(Kenmerk:\s*)([A-Za-z0-9]{4,})"),
        re.compile(r"(Machtiging:\s*)([A-Za-z0-9-]{4,})"),
        re.compile(r"(Incassant:\s*)([A-Za-z0-9]{4,})"),
        re.compile(r"(NR:)([A-Z0-9]{4,})"))

COMPANY = re.compile(r"\b(B\.?\s?V|N\.?\s?V|V\.?O\.?F|Ltd|Inc|LLC|GmbH|S\.?A|PLC|Holding|Stichting|"
                     r"Gemeente|Ministerie|Belastingdienst|Bank|Verzekering|Group|Services|"
                     r"International|Nederland|Europe|Energie|Zorg|Media|Telecom)\b", re.I)
DOTTED = re.compile(r"\b[A-Z]\.")
TUSSENVOEGSEL = re.compile(r"(^|\s)(de|van|van der|van den|den|der|ter|te)(\s)", re.I)

SURNAMES = ["de Vries", "Jansen", "Bakker", "Visser", "Smit", "Meijer", "Bos", "Vos"]
INITIALS = ["A.", "B.", "C.", "J.", "K.", "M.", "P.", "T."]
ACCOUNT_KINDS = ["Betaalrekening", "Spaarrekening", "Beleggingsrekening"]


class Standins:
    """One stand-in per distinct value, drawn once and reused everywhere it appears."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.seen = {}

    def get(self, kind, value, make):
        key = (kind, value)
        if key not in self.seen:
            self.seen[key] = make()
        return self.seen[key]

    def iban(self, country, bank, tail):
        # The bank code stays, so an ABN account still reads as one: it names the institution
        # rather than the holder, and it is part of the shape the model sees.
        return self.get("iban", country + bank + tail,
                        lambda: f"{country}{bank}0{self.rng.randrange(10 ** 9):09d}")

    def card(self, digits):
        return self.get("card", digits,
                        lambda: f"{self.rng.randrange(10 ** len(digits)):0{len(digits)}d}")

    def account(self, value):
        return self.get("account", value,
                        lambda: (f"ABN AMRO {self.rng.choice(ACCOUNT_KINDS)} "
                                 f"{self.rng.randrange(1000, 10000)}"))

    def person(self, value):
        return self.get("person", value,
                        lambda: f"{self.rng.choice(INITIALS)} {self.rng.choice(SURNAMES)}")

    def ref(self, value):
        pool = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"
        return self.get("ref", value,
                        lambda: "".join(self.rng.choice(pool) for _ in value))


def account_holder(records):
    """The names on transfers between the owner's own accounts are the owner, in every spelling the bank writes them."""
    names = set()
    for record in records:
        if record.get("gold", {}).get("category") == "internal_transfer":
            names.update(value.strip() for _, value in NAAM.findall(record["state"]))
    return names


def is_person(value, holder, merchants):
    if value in holder:
        return True
    if COMPANY.search(value) or value.lower() in merchants:
        return False
    return bool(DOTTED.search(value) or TUSSENVOEGSEL.search(value))


def scrub(state, standins, holder, merchants):
    state = IBAN.sub(lambda m: standins.iban(*m.groups()), state)
    state = CARD.sub(lambda m: m.group(1) + m.group(2) + standins.card(m.group(3)), state)
    state = ACCOUNT.sub(lambda m: m.group(1) + standins.account(m.group(2).strip()), state)
    state = NAAM.sub(
        lambda m: m.group(1) + (standins.person(m.group(2).strip())
                                if is_person(m.group(2).strip(), holder, merchants)
                                else m.group(2)), state)
    for pattern in REFS:
        state = pattern.sub(lambda m: m.group(1) + standins.ref(m.group(2)), state)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="source", default="data/lunchmoney")
    p.add_argument("--out", dest="target", default="data/lunchmoney_clean")
    p.add_argument("--merchants", default="lab/merchants_nl.json")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    splits = {}
    for split in SPLITS:
        source = Path(args.source, f"{split}.jsonl")
        if source.exists():
            splits[split] = [json.loads(line) for line in source.read_text().splitlines()
                             if line.strip()]

    merchants = {name.lower() for group in json.loads(Path(args.merchants).read_text()).values()
                 for name in group}
    holder = account_holder([r for rows in splits.values() for r in rows])

    # One Standins across every split, so an account keeps the same stand-in in train and in test.
    # A per-split mapping would leak the split boundary into the text.
    standins = Standins(args.seed)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, rows in splits.items():
        rewritten = 0
        lines = []
        for record in rows:
            clean = scrub(record["state"], standins, holder, merchants)
            rewritten += clean != record["state"]
            record["state"] = clean
            lines.append(json.dumps(record, ensure_ascii=False))
        Path(target, f"{split}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
        report[split] = {"rows": len(rows), "rewritten": rewritten}

    counts = {}
    for kind, _ in standins.seen:
        counts[kind] = counts.get(kind, 0) + 1
    # The mapping is the only place the original values still exist, so it is written beside the
    # data, which is ignored by git, and never printed.
    Path(target, "mapping.json").write_text(json.dumps(
        {f"{kind}\t{value}": standin for (kind, value), standin in standins.seen.items()},
        ensure_ascii=False, indent=1))
    print(json.dumps({"splits": report, "distinct_values_replaced": counts,
                      "holder_spellings": len(holder),
                      "mapping": str(target / "mapping.json")}, indent=1))


if __name__ == "__main__":
    main()
