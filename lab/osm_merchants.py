#!/usr/bin/env python3
"""Extract real Dutch merchant names per budget category from an OSM extract.

OSM shop/amenity tags are already a taxonomy, so the mapping below is mechanical. Branch counts come along as a frequency weight: Albert Heijn has 218 branches and Ekoplaza 22, so synthetic data built from this mirrors how often each merchant is actually encountered instead of sampling merchants uniformly.

    osmium tags-filter nh.osm.pbf -o brands.osm.pbf "nwr/shop" \
        "nwr/amenity=pharmacy,fuel,cinema,restaurant,cafe,fast_food,bank" "nwr/office=insurance"
    osmium export brands.osm.pbf -f json -o brands.json
    python lab/osm_merchants.py --input brands.json --output lab/merchants_nl.json
"""
import argparse
import collections
import json
import re
from pathlib import Path

# OSM tag -> your budget category. Tags absent here are dropped rather than guessed.
TAG_CATEGORY = {
    "groceries": ["supermarket", "convenience", "greengrocer", "butcher", "bakery", "cheese",
                  "deli", "seafood", "alcohol", "wine", "confectionery", "pastry", "farm",
                  "frozen_food", "health_food", "spices", "coffee", "tea", "dairy", "chocolate"],
    "personal_care": ["chemist", "pharmacy", "hairdresser", "beauty", "cosmetics", "perfumery",
                      "massage", "optician", "nail_salon", "herbalist", "hearing_aids"],
    "shopping": ["clothes", "shoes", "jewelry", "gift", "furniture", "interior_decoration",
                 "doityourself", "houseware", "electronics", "mobile_phone", "computer",
                 "department_store", "variety_store", "mall", "toys", "sports", "books",
                 "stationery", "kitchen", "garden_centre", "hardware", "second_hand",
                 "bag", "fabric", "watches", "antiques", "florist", "pet", "newsagent",
                 "bicycle", "outdoor", "musical_instrument", "photo", "paint", "trade"],
    "transportation": ["fuel", "car_repair", "car_parts", "tyres", "charging_station"],
    "entertainment": ["cinema", "theatre", "nightclub", "casino", "video_games", "music",
                      "art", "games", "video"],
    "insurance": ["insurance"],
}
TAG_TO_CATEGORY = {tag: cat for cat, tags in TAG_CATEGORY.items() for tag in tags}

# Names that are addresses, categories or noise rather than a merchant a bank would print.
NOISE = re.compile(r"^(yes|shop|winkel|onbekend|unknown|\d+)$", re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="GeoJSON from `osmium export`")
    p.add_argument("--output", required=True)
    p.add_argument("--min-branches", type=int, default=1)
    p.add_argument("--exclude", help="JSONL split whose merchants must not appear (test set)")
    args = p.parse_args()

    excluded = set()
    if args.exclude:
        for line in Path(args.exclude).read_text().splitlines():
            if line.strip():
                group = json.loads(line)["metadata"]["source_group_id"]
                excluded.add(group.split("merchant:", 1)[-1])

    counts = collections.defaultdict(collections.Counter)
    for feature in json.load(Path(args.input).open())["features"]:
        tags = feature.get("properties", {})
        tag = tags.get("shop") or tags.get("amenity") or tags.get("office")
        name = (tags.get("brand") or tags.get("name") or "").strip()
        category = TAG_TO_CATEGORY.get(tag)
        if not category or not name or NOISE.match(name) or len(name) > 40:
            continue
        counts[category][name] += 1

    # A merchant already in the eval split would leak the answer into training.
    normalized = lambda s: re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", " ", s.upper())).strip()[:40]
    out, dropped = {}, 0
    for category, names in counts.items():
        kept = {}
        for name, n in names.items():
            if n < args.min_branches:
                continue
            if normalized(name) in excluded:
                dropped += 1
                continue
            kept[name] = n
        out[category] = dict(sorted(kept.items(), key=lambda kv: -kv[1]))

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "categories": len(out),
        "merchants": sum(len(v) for v in out.values()),
        "branches": sum(sum(v.values()) for v in out.values()),
        "excluded_as_test_merchants": dropped,
        "per_category": {k: len(v) for k, v in sorted(out.items(), key=lambda kv: -len(kv[1]))},
    }, indent=1))


if __name__ == "__main__":
    main()
