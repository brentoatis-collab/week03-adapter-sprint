#!/usr/bin/env python3
"""Deterministic generator for the CivicDesk DPO preference dataset (seed 42).

Every pair = one resident complaint + a CHOSEN ticket + a REJECTED ticket.
Chosen is the canonical contract-correct ticket (same rendering code as the frozen
instruction dataset). Rejected differs ONLY in address handling:

  A fabricated_address               no exact address; rejected invents address_or_null
  B fabricated_address_and_location  no exact address; rejected invents an address and
                                     also puts it in `location` ("resolves" the location)
  C over_null          (control)     exact address given; rejected returns null
  D altered_address    (control)     exact address given; rejected changes the house number

Category, urgency and summary are always identical between chosen and rejected.

Shortcut control: null appears in exactly as many rejected responses (type C) as chosen
responses (types A+B), so "prefer null" cannot be learned from the response alone.
Fabricated addresses reuse the resident-address distributions (house-number ranges,
suffix forms incl. no suffix) so they are not identifiable by surface form.

Leakage control: complaints are rendered only from TRAIN scenario families of the frozen
instruction split, never eval families, and must not duplicate any instruction complaint.

Usage:
    python scripts/generate_preferences.py            # writes data/preference_train.jsonl
    python scripts/generate_preferences.py --out /tmp/p.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import generate_instruction_data as gen

REPO_ROOT = gen.REPO_ROOT
PAIR_TYPES = ("fabricated_address", "fabricated_address_and_location", "over_null", "altered_address")
NO_ADDRESS_PAIR_TYPES = PAIR_TYPES[:2]
STREET_BY_NAME = {s[0].lower(): s for s in gen.STREETS}


def house_number(rng: random.Random) -> int:
    """Same distribution as resident addresses in the instruction generator."""
    r = rng.random()
    return rng.randint(1, 99) if r < 0.2 else (rng.randint(100, 999) if r < 0.7 else rng.randint(1000, 9999))


def street_from_text(text: str) -> tuple[str, str, str] | None:
    """Find a known street in normalized location text (e.g. 'Oak Ave' -> Oak tuple)."""
    low = text.lower()
    for name in sorted(STREET_BY_NAME, key=len, reverse=True):
        if re.search(rf"(^|\s){re.escape(name)}(\s|$)", low):
            return STREET_BY_NAME[name]
    return None


def fabricate_address(rng: random.Random, loc_type: str, location: str, complaint: str) -> str:
    """A plausible but unsupported house-number + street, derived the way a model would
    hallucinate it from the supplied location (a street it mentioned, a number inside a
    block range) or invented outright when nothing usable was supplied."""
    for _ in range(50):
        street = None
        number = house_number(rng)
        if loc_type == "intersection":
            parts = location.replace(" (intersection)", "").split(" and ")
            street = street_from_text(rng.choice(parts))
        elif loc_type == "block_range":
            m = re.match(r"(\d+) block of (.+)", location)
            if m:
                number = int(m.group(1)) + rng.randint(1, 99)
                street = street_from_text(m.group(2))
            else:
                street = street_from_text(location.split(" between ")[0])
        elif loc_type == "street_only":
            street = street_from_text(location.replace(" (no house number given)", ""))
        elif loc_type == "vague" and "(resident unsure)" in location:
            street = street_from_text(rng.choice(location.replace(" (resident unsure)", "").split(" or ")))
        if street is None:
            street = rng.choice(gen.STREETS)
        _, norm = gen.street_forms(rng, street)
        addr = f"{number} {norm}"
        if addr.lower() not in complaint.lower():
            return addr
    raise RuntimeError(f"could not fabricate an unsupported address for {location!r}")


def alter_house_number(rng: random.Random, address: str) -> str:
    """Change the house number but keep its digit count (length-neutral)."""
    num, rest = address.split(" ", 1)
    for _ in range(50):
        digits = list(num)
        i = rng.randrange(len(digits))
        choices = [d for d in "0123456789" if d != digits[i] and not (i == 0 and d == "0")]
        digits[i] = rng.choice(choices)
        new = "".join(digits)
        if new != num:
            return f"{new} {rest}"
    raise RuntimeError(f"could not alter {address!r}")


def dumps(ticket: dict) -> str:
    assert tuple(ticket) == gen.TICKET_KEYS
    return json.dumps(ticket, ensure_ascii=False)


def build_preferences(seed: int, cfg: dict) -> list[dict]:
    data_cfg = cfg["data"]
    mix = data_cfg["preference_mix"]
    n_total = data_cfg["n_preference_examples"]
    if sum(mix[t] for t in PAIR_TYPES) != n_total:
        raise ValueError(f"preference_mix sums to {sum(mix[t] for t in PAIR_TYPES)}, expected {n_total}")

    # Frozen split -> training families only (the instruction build is pure and deterministic).
    inst_train, inst_eval, split_info = gen.build_dataset(data_cfg["seed"], data_cfg)
    train_fams = split_info["train_families"]
    wide_train = [(c, i) for c in gen.CATEGORIES for i in train_fams[c] if not gen.SCENARIOS[c][i].point]
    point_train = [(c, i) for c in gen.CATEGORIES for i in train_fams[c] if gen.SCENARIOS[c][i].point]
    existing = {r["complaint"] for r in inst_train + inst_eval}

    rng = random.Random(seed)

    # Pair-type slots.
    types = [t for t in PAIR_TYPES for _ in range(mix[t])]
    rng.shuffle(types)

    # No-address conditions: equal count per condition across A+B, spread over both types.
    n_noaddr = sum(mix[t] for t in NO_ADDRESS_PAIR_TYPES)
    per_cond, rem = divmod(n_noaddr, len(gen.NO_ADDRESS_TYPES))
    if rem:
        raise ValueError(f"{n_noaddr} no-address pairs not divisible by {len(gen.NO_ADDRESS_TYPES)} conditions")
    conds = [c for c in gen.NO_ADDRESS_TYPES for _ in range(per_cond)]
    rng.shuffle(conds)
    # street_only: half street-wide (actionable), half point (clarification)
    n_so = conds.count("street_only")
    so_wide = [True] * (n_so // 2) + [False] * (n_so - n_so // 2)
    rng.shuffle(so_wide)

    # Balanced category cycle for non-street-only slots.
    cat_cycle = []
    while len(cat_cycle) < n_total:
        block = list(gen.CATEGORIES)
        rng.shuffle(block)
        cat_cycle += block
    styles, weights = zip(*gen.STYLES)

    records, seen = [], set()
    cond_iter, so_iter = iter(conds), iter(so_wide)
    for n, ptype in enumerate(types):
        loc_type = next(cond_iter) if ptype in NO_ADDRESS_PAIR_TYPES else "exact_address"
        if loc_type == "street_only":
            cat, s_idx = rng.choice(wide_train if next(so_iter) else point_train)
        else:
            cat = cat_cycle[n]
            s_idx = rng.choice(train_fams[cat])
        slot = {"category": cat, "scenario_index": s_idx, "location_type": loc_type,
                "style": rng.choices(styles, weights)[0], "secondary": None}
        if rng.random() < data_cfg["multi_issue_rate"]:
            other = rng.choice([c for c in gen.CATEGORIES if c != cat])
            slot["secondary"] = {"category": other, "scenario_index": rng.choice(train_fams[other])}

        for _ in range(50):
            rec = gen.render_example(rng, dict(slot))
            if rec["complaint"] not in existing and rec["complaint"] not in seen:
                break
        else:
            raise RuntimeError("could not render a unique complaint")
        seen.add(rec["complaint"])

        chosen = json.loads(rec["response"])
        rejected = dict(chosen)
        if ptype == "fabricated_address":
            fab = fabricate_address(rng, loc_type, chosen["location"], rec["complaint"])
            rejected["address_or_null"] = fab
            error_value = fab
        elif ptype == "fabricated_address_and_location":
            fab = fabricate_address(rng, loc_type, chosen["location"], rec["complaint"])
            rejected["address_or_null"] = fab
            rejected["location"] = fab
            error_value = fab
        elif ptype == "over_null":
            rejected["address_or_null"] = None
            error_value = None
        else:  # altered_address
            alt = alter_house_number(rng, chosen["address_or_null"])
            rejected["address_or_null"] = alt
            rejected["location"] = alt
            error_value = alt

        records.append({
            "pair_type": ptype,
            "family_id": rec["family_id"],
            "secondary_family_id": rec["secondary_family_id"],
            "category": rec["category"],
            "urgency": rec["urgency"],
            "location_type": loc_type,
            "has_exact_address": rec["has_exact_address"],
            "needs_clarification": rec["needs_clarification"],
            "style": rec["style"],
            "rejected_value": error_value,
            "prompt": rec["complaint"],
            "chosen": dumps(chosen),
            "rejected": dumps(rejected),
        })

    rng.shuffle(records)
    for i, r in enumerate(records, 1):
        r["id"] = f"cd-pref-{i:04d}"
    return records


FIELD_ORDER = ["id", "pair_type", "family_id", "secondary_family_id", "category", "urgency", "location_type",
               "has_exact_address", "needs_clarification", "style", "rejected_value", "prompt", "chosen", "rejected"]


def serialize(records: list[dict]) -> str:
    return "".join(json.dumps({k: r[k] for k in FIELD_ORDER}, ensure_ascii=False) + "\n" for r in records)


def main() -> None:
    cfg = json.loads(gen.CONFIG_PATH.read_text())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=cfg["data"]["preference_seed"])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / cfg["data"]["preference_train_path"])
    args = ap.parse_args()
    recs = build_preferences(args.seed, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialize(recs))
    print(f"seed={args.seed}  wrote {len(recs)} preference pairs -> {args.out}")


if __name__ == "__main__":
    main()
