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
import string
from pathlib import Path

SPLITS = ("train", "dev", "calibration", "test", "ood")

# Dutch bank statement fields, in the shapes the export writes them.
IBAN = re.compile(r"\b([A-Z]{2}\d{2})([A-Z]{4})([A-Z0-9]{4,20})\b")
CARD = re.compile(r"(KAARTNUMMER:\s*)(\*+)(\d+)", re.I)
ACCOUNT = re.compile(r"^(Account:\s*)(.+)$", re.M)
NAAM = re.compile(r"(Naam:\s*)(.+?)(?=\s+(?:Omschrijving|Machtiging|Kenmerk|IBAN|BIC):|$)", re.I)
# The dash has to be inside the class, not a boundary: a reference written 12-34-5678 would
# otherwise match only its first pair, fail the length test, and pass through untouched.
REFS = (re.compile(r"(Kenmerk:\s*)([A-Za-z0-9-]{4,})"),
        re.compile(r"(Machtiging:\s*)([A-Za-z0-9-]{4,})"),
        re.compile(r"(Incassant:\s*)([A-Za-z0-9-]{4,})"),
        re.compile(r"(NR:)([A-Z0-9-]{4,})"))

# Words the bank writes in a reference field to say there is no reference. They identify nobody,
# and scrambling them would turn a meaningful absence into a plausible looking number.
MARKERS = {"NOTPROVIDED", "TERMBNET"}

COMPANY = re.compile(r"\b(B\.?\s?V|N\.?\s?V|V\.?O\.?F|Ltd|Inc|LLC|GmbH|S\.?A|PLC|Holding|Stichting|"
                     r"Gemeente|Ministerie|Belastingdienst|Bank|Verzekering|Group|Services|"
                     r"International|Nederland|Europe|Energie|Zorg|Media|Telecom)\b", re.I)
DOTTED = re.compile(r"\b[A-Z]\.")
TUSSENVOEGSEL = re.compile(r"(^|\s)(de|van|van der|van den|den|der|ter|te)(\s)", re.I)

FIRST_NAMES = ["Jan", "Piet", "Sanne", "Emma", "Luuk", "Noor", "Daan", "Fleur"]
SURNAMES = ["Jansen", "Bakker", "Visser", "Smit", "Meijer", "Vos", "Dekker", "Hendriks"]


class Standins:
    """One stand-in per distinct value, drawn once and reused everywhere it appears.

    Every replacement keeps the shape of what it replaces. A twelve digit mandate number becomes
    twelve digits, a dashed reference keeps its dashes, and a name stays a name, because a value
    that no longer looks like its own field is a value the model has to learn around.
    """

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.seen = {}

    def get(self, kind, value, make):
        key = (kind, value)
        if key not in self.seen:
            self.seen[key] = make()
        return self.seen[key]

    def like(self, value):
        """Same characters class for class: digits stay digits, letters keep their case, and punctuation is left where it is."""
        out = []
        for ch in value:
            if ch.isdigit():
                out.append(self.rng.choice(string.digits))
            elif ch.isupper():
                out.append(self.rng.choice(string.ascii_uppercase))
            elif ch.islower():
                out.append(self.rng.choice(string.ascii_lowercase))
            else:
                out.append(ch)
        return "".join(out)

    def iban(self, country, bank, tail):
        # The country and the bank code stay, because they name the institution rather than the
        # holder and are part of the shape the model reads. The check digits go with the account
        # number they are derived from.
        return self.get("iban", country + bank + tail,
                        lambda: country[:2] + self.like(country[2:]) + bank + self.like(tail))

    def card(self, digits):
        return self.get("card", digits, lambda: self.like(digits))

    def account(self, value):
        """An account label is an institution, sometimes a currency, then the holder's initials and number. The first two stay, the last two go."""
        def make():
            parts = []
            for token in value.split():
                identifying = token.isdigit() or (token.isalpha() and len(token) <= 2)
                parts.append(self.like(token) if identifying else token)
            return " ".join(parts)
        return self.get("account", value, make)

    def person(self, value):
        """A generic name of the same shape: an initial stays an initial, a word stays a word, and a shouted name stays shouted."""
        def make():
            parts = []
            for index, token in enumerate(value.split()):
                if re.fullmatch(r"[A-Za-z]\.?", token):
                    parts.append(self.rng.choice(string.ascii_uppercase)
                                 + ("." if token.endswith(".") else ""))
                else:
                    word = self.rng.choice(FIRST_NAMES if index == 0 else SURNAMES)
                    parts.append(word.upper() if token.isupper() else word)
            return " ".join(parts)
        return self.get("person", value, make)

    def ref(self, value):
        if value.upper() in MARKERS:
            return value
        return self.get("ref", value, lambda: self.like(value))


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
    p.add_argument("--mapping", help="where the original to stand-in map is written; "
                                     "defaults beside the source, never inside the output")
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
    mapping = Path(args.mapping or Path(args.source).parent / "pii_mapping.json")
    mapping.write_text(json.dumps(
        {f"{kind}\t{value}": standin for (kind, value), standin in standins.seen.items()},
        ensure_ascii=False, indent=1))
    print(json.dumps({"splits": report, "distinct_values_replaced": counts,
                      "holder_spellings": len(holder), "mapping": str(mapping)}, indent=1))


if __name__ == "__main__":
    main()
