#!/usr/bin/env python3
"""Generate synthetic Dutch bank transactions from real OSM merchants.

Merchant names are real (OpenStreetMap, ODbL) and the formats are lifted from an actual ABN AMRO / Wise export, so the only invented parts are the reference codes, amounts and dates. Labels are correct by construction: a merchant is drawn from the list for the category we are generating.

Merchants appearing in the evaluation split are excluded upstream by osm_merchants.py.

    python lab/synth_nl.py --merchants lab/merchants_nl.json --output data/synth/train.jsonl -n 6000
"""
import argparse
import json
import random
import string
from pathlib import Path

CITIES = ["AMSTERDAM", "HAARLEM", "ZAANDAM", "AMSTELVEEN", "HOOFDDORP", "ALKMAAR",
          "PURMEREND", "HILVERSUM", "UTRECHT", "DEN HAAG", "ROTTERDAM", "LEIDEN"]
# Account labels, card digits and the transfer owner are stand-ins. The shapes are what the model reads; the real values are personal data and belong only in the private export.
ACCOUNTS = ["ABN AMRO Betaalrekening 1234", "ABN AMRO Spaarrekening 5678", "Wise EUR 9012"]
CARDS = ["**1234", "**5678"]
OWNERS = ["A. de Vries", "A DE VRIES", "A.J. de Vries", "de Vries A.J."]

# Non-shop categories have no OSM equivalent; these are well-known Dutch brands a model reproduces reliably, unlike the retail long tail.
BRANDS = {
    "utilities": ["Waternet", "Vattenfall", "Eneco", "Essent", "Greenchoice", "Budget Energie",
                  "KPN", "Ziggo", "Odido", "Simyo", "Vodafone", "Tele2", "PWN", "Dunea"],
    "insurance": ["Zilveren Kruis", "VGZ", "CZ", "Menzis", "Achmea", "Centraal Beheer", "ASR",
                  "Nationale-Nederlanden", "Univé", "OHRA", "FBTO", "Interpolis", "Aegon"],
    "subscriptions": ["Netflix", "Spotify", "Disney Plus", "Videoland", "Adobe", "Microsoft",
                      "Google", "Apple", "Dropbox", "GitHub", "Strava", "Audible", "NPO Plus"],
    "donation": ["Stichting Zeehondencentrum Pieterburen", "Greenpeace Nederland", "KWF Kankerbestrijding",
                 "Artsen zonder Grenzen", "Natuurmonumenten", "Amnesty International", "Rode Kruis",
                 "Stichting AAP", "Hartstichting", "Oxfam Novib"],
    "rent_loan": ["Stichting kwaliteitsgeld", "Vesteda", "Bouwinvest", "Woonstad", "Ymere",
                  "De Key", "Stadgenoot", "Rochdale"],
    "transportation": ["NS Groep IZ NS Reizigers", "GVB Amsterdam", "OVpay", "Connexxion", "Q-Park",
                       "Interparking", "NS Stations Retailbedrijf"],
    # Online platforms plus the Dutch adult-education institutions that run language and hobby courses. Healthcare is deliberately absent: the US data defines that category.
    "education": ["Coursera", "Udemy", "edX", "Babbel", "Duolingo", "Pluralsight", "Skillshare",
                  "MasterClass", "Volksuniversiteit Amsterdam", "Taalcentrum VU", "Regina Coeli",
                  "NHA Opleidingen", "LOI", "NTI", "Open Universiteit", "Codam", "Winc Academy"],
    "restaurants": ["Loetje", "La Place", "De Beren", "Vapiano", "Happy Italy", "Bagels & Beans",
                    "New York Pizza", "FEBO", "Kwalitaria", "Bram Ladage", "Van der Valk"],
    "pets": ["Pets Place", "Dierenspeciaalzaak", "Welkoop", "Jumper", "Dierenkliniek Amsterdam",
             "Medipet", "Zooplus"],
    "travel": ["Booking.com", "Transavia", "KLM", "NS International", "FlixBus", "Eurostar",
               "Van der Valk Hotel", "Center Parcs", "Landal GreenParks", "Sunweb"],
}

REF = lambda n=6: "".join(random.choices(string.ascii_uppercase + string.digits, k=n))
IBAN = lambda: f"NL{random.randint(10,99)}{random.choice(['INGB','ABNA','RABO','TRIO','SNSB','DEUT'])}0{random.randint(100000000,999999999)}"


def date_str():
    return f"{random.randint(1,28):02d}.{random.randint(1,12):02d}.26/{random.randint(0,23):02d}:{random.randint(0,59):02d}"


def card(merchant, category):
    """BEA/eCom card formats — how in-person and online card spending arrives."""
    city = random.choice(CITIES)
    store = f" FIL{random.randint(1000,9999)}" if random.random() < 0.3 else f" {random.randint(1,9999)}"
    if random.random() < 0.15:
        return (f"eCom, Betaalpas {merchant.upper()} NR:{REF(8)}, {date_str()} {city}, "
                f"Land: NLD KAARTNUMMER: {random.choice(CARDS)}")
    prefix = "BEA, Apple Pay" if random.random() < 0.75 else "BEA, Betaalpas"
    return (f"{prefix} {merchant.upper()}{store} NR:{REF()}, {date_str()} {city} "
            f"KAARTNUMMER: {random.choice(CARDS)}")


def incasso(merchant, category):
    """SEPA direct debit — how utilities, insurance and subscriptions arrive."""
    return (f"SEPA Incasso algemeen doorlopend Incassant: NL{random.randint(10,99)}ZZZ"
            f"{random.randint(100000000000,999999999999)} Naam: {merchant} "
            f"Machtiging: MD-{random.randint(1000000,9999999)} Omschrijving: Klant: "
            f"{random.randint(1000000000,9999999999)}, Factuur: {random.randint(1000000000,9999999999)} "
            f"IBAN: {IBAN()} Kenmerk: DBet{random.randint(100000000,999999999)}")


def overboeking(merchant, category):
    """SEPA credit transfer — rent, salary, transfers between accounts."""
    note = {"rent_loan": f"Rent {random.choice(['januari','februari','maart','september','oktober'])}",
            "income": "Salaris", "internal_transfer": "Overboeking"}.get(category, "Betaling")
    return (f"SEPA Overboeking IBAN: {IBAN()} BIC: {random.choice(['INGBNL2A','ABNANL2A','RABONL2U'])} "
            f"Naam: {merchant} Omschrijving: {note} Kenmerk: NOTPROVIDED")


def ideal(merchant, category):
    """SEPA iDEAL — the Dutch online payment rail, used for donations and one-off buys."""
    return (f"SEPA iDEAL/Wero IBAN: {IBAN()} BIC: DEUTNL2A Naam: {merchant} via Stichting Mollie "
            f"Payments Omschrijving: {REF(16).lower()} {random.randint(10**15, 10**16-1)}")



# Three categories have no merchant. They are defined by structure, not by who was paid.

def fee_row():
    """The bank charging its own customer. No counterparty at all."""
    kind = random.choice(["Betaalautomaat VV", "Maandelijkse kosten", "Kosten geldopname",
                          "Betaalpas vervanging", "Spoedoverboeking"])
    n = random.randint(1, 12)
    amount = round(random.uniform(0.35, 14.0), 2)
    return f"ABN AMRO Bank N.V. {n} x {kind} {str(amount).replace('.', ',')}", -amount


def income_row():
    """Salary, refunds and unnamed inbound transfers. The reliable signal is the positive sign."""
    kind = random.random()
    if kind < 0.45:
        name = random.choice(["Payroll Services B.V.", "Randstad Nederland", "Stichting Salaris",
                              "Adyen N.V.", "Booking.com B.V.", "Werkgever B.V."])
        note, amount = "Salaris", round(random.uniform(1800, 9000), 2)
    elif kind < 0.75:
        name = random.choice(["Belastingdienst", "Zorginstituut Nederland", "Gemeente Amsterdam",
                              "Bol.com", "Coolblue", "Zalando"])
        note, amount = random.choice(["Teruggaaf", "Refund", "Restitutie"]), round(random.uniform(5, 400), 2)
    else:
        name = random.choice(["AAB INZ TIKKIE", "ING Bank", "Rabobank", "SNS Bank"])
        note, amount = f"Tikkie ID {random.randint(10**11, 10**12-1)}", round(random.uniform(5, 250), 2)
    desc = (f"SEPA Overboeking IBAN: {IBAN()} BIC: {random.choice(['INGBNL2A','ABNANL2A','RABONL2U'])} "
            f"Naam: {name} Omschrijving: {note} Kenmerk: NOTPROVIDED")
    return desc, amount, name


def transfer_pair():
    """An internal transfer is a pair, not a row: the same amount leaves one owned account and arrives in another within a few days. Generated as two linked rows so the model can learn that a matching counter-leg is what distinguishes a transfer from income."""
    amount = round(random.uniform(20, 9000), 2)
    owner = random.choice(OWNERS)
    out_acct, in_acct = random.sample(ACCOUNTS, 2)
    if random.random() < 0.35:
        out = f"Sent money to {owner} (fee: {round(random.uniform(0.2, 22), 2)} EUR)"
        inn = f"Received money from {owner} with reference {REF(8)}"
    else:
        out = (f"SEPA Overboeking IBAN: {IBAN()} BIC: ABNANL2A Naam: {owner} "
               f"Omschrijving: Overboeking eigen rekening Kenmerk: NOTPROVIDED")
        inn = (f"SEPA Overboeking IBAN: {IBAN()} BIC: ABNANL2A Naam: {owner} "
               f"Omschrijving: Overboeking eigen rekening Kenmerk: NOTPROVIDED")
    # A few cents of drift and a few days apart, as a real pair looks.
    return [(out, -amount, owner, out_acct), (inn, round(amount * random.uniform(0.997, 1.0), 2), owner, in_acct)]


FORMATS = {
    "groceries": [(card, 0.95), (ideal, 0.05)],
    "shopping": [(card, 0.7), (ideal, 0.3)],
    "personal_care": [(card, 1.0)],
    "transportation": [(card, 0.8), (incasso, 0.2)],
    "entertainment": [(card, 0.6), (ideal, 0.4)],
    "utilities": [(incasso, 0.85), (overboeking, 0.15)],
    "insurance": [(incasso, 1.0)],
    "subscriptions": [(incasso, 0.4), (card, 0.6)],
    "donation": [(ideal, 0.7), (incasso, 0.3)],
    "rent_loan": [(overboeking, 1.0)],
    "income": [(overboeking, 1.0)],
    "internal_transfer": [(overboeking, 1.0)],
    "education": [(ideal, 0.5), (card, 0.3), (incasso, 0.2)],
    "restaurants": [(card, 1.0)],
    "pets": [(card, 0.85), (ideal, 0.15)],
    "travel": [(ideal, 0.5), (card, 0.5)],
}

AMOUNTS = {
    "groceries": (2, 90), "shopping": (3, 320), "personal_care": (2, 70),
    "transportation": (1.5, 95), "entertainment": (6, 60), "utilities": (9, 180),
    "insurance": (11, 220), "subscriptions": (2, 40), "donation": (5, 60),
    "rent_loan": (700, 2800), "income": (400, 9000), "internal_transfer": (20, 9000),
    "education": (12, 900), "restaurants": (8, 120), "pets": (4, 140), "travel": (25, 1200),
}
POSITIVE = {"income", "internal_transfer"}

PROMPT_STATE = ("Bank description: {desc}\nPayee: {payee}\nAmount: {amount} EUR\n"
                "Date: 2026-{m:02d}-{d:02d} ({day})\nAccount: {account}")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--merchants", required=True, help="from osm_merchants.py")
    p.add_argument("--reference", default="data/lunchmoney/train.jsonl",
                   help="supplies the question/criteria block so prompts match exactly")
    p.add_argument("--output", required=True)
    p.add_argument("-n", type=int, default=6000)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    random.seed(args.seed)

    osm = json.loads(Path(args.merchants).read_text())
    pool = {cat: list(BRANDS.get(cat, [])) for cat in FORMATS}
    weights = {}
    for cat in FORMATS:
        names = list(osm.get(cat, {}).items()) + [(b, 3) for b in BRANDS.get(cat, [])]
        if not names:
            continue
        pool[cat] = [n for n, _ in names]
        weights[cat] = [w for _, w in names]

    template = json.loads(Path(args.reference).read_text().splitlines()[0])
    questions = template["questions"]
    categories = list(questions["category"]["criteria"])
    usable = [c for c in categories if pool.get(c)]

    # fees/income/internal_transfer have no merchant, so they get structural generators.
    usable = usable + [c for c in ("fees", "income", "internal_transfer") if c in categories]

    rows = []
    def emit(cat, desc, amount, payee, account=None, group=None):
        uid = f"synth_{len(rows):06d}"
        rows.append({
            "id": uid, "state_id": uid, "family_id": "lunchmoney_synth", "split": "train",
            "state": PROMPT_STATE.format(desc=desc, payee=payee, amount=amount,
                                         m=random.randint(1, 9), d=random.randint(1, 28),
                                         day=random.choice(DAYS),
                                         account=account or random.choice(ACCOUNTS)),
            "questions": questions,
            "gold": {"category": cat},
            "metadata": {"source_group_id": group or f"synth:{payee}"},
        })

    i = 0
    while len(rows) < args.n:
        cat = usable[i % len(usable)]
        i += 1
        if cat == "fees":
            desc, amount = fee_row()
            emit(cat, desc, amount, "ABN AMRO Bank N.V.", group="synth:abn_fee")
        elif cat == "income":
            desc, amount, name = income_row()
            emit(cat, desc, amount, name)
        elif cat == "internal_transfer":
            # Both legs share a source_group_id so the pair never straddles a split.
            group = f"synth:transfer_{len(rows):06d}"
            for desc, amount, owner, acct in transfer_pair():
                emit(cat, desc, amount, owner, account=acct, group=group)
        else:
            merchant = random.choices(pool[cat], weights=weights[cat], k=1)[0]
            fmts, probs = zip(*FORMATS[cat])
            desc = random.choices(fmts, weights=probs, k=1)[0](merchant, cat)
            low, high = AMOUNTS[cat]
            amount = round(random.uniform(low, high), 2)
            emit(cat, desc, amount if cat in POSITIVE else -amount, merchant)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    import collections
    print(json.dumps({"rows": len(rows), "categories": len(usable),
                      "distinct_merchants": len({r["metadata"]["source_group_id"] for r in rows}),
                      "per_category": dict(collections.Counter(r["gold"]["category"] for r in rows))},
                     indent=1))


if __name__ == "__main__":
    main()
