#!/usr/bin/env python3
"""FROZEN scoring rules for the three-way held-out evaluation (BASE vs SFT vs DPO).

These rules were written before any model output of the formal evaluation was produced.
evaluate_models.py records this file's SHA-256 in every result manifest, so any later
change to the rules is detectable. Scoring is pure, deterministic and programmatic; raw
generations are never edited. Cases the behavior specification does not settle are FLAGGED
(per-example `flags`), never silently resolved.

Rules (see README 9 for the prose version)
------------------------------------------
Parsing (no repair):
  strict     the whole generated text (whitespace-stripped) is exactly one JSON object
  extracted  otherwise, the first decodable JSON object found by scanning for '{'
             (e.g. inside a Markdown fence); reported separately, never counted as compliant
  none       no JSON object at all -> content fields unscorable

Contract compliance = strict parse AND keys exactly (category, urgency, location, summary,
address_or_null) in that order AND category/urgency in their enums AND location, summary
non-empty strings AND address_or_null is a string or null.

Address outcome (exact-address complaints, gold address present):
  retained_exact          normalized(pred) == normalized(gold)       <- primary retention
  retained_suffix_variant same house number + street name, different/absent/added suffix
                          (a spec violation: suffixes must not be completed or changed; flagged)
  over_null               pred is null
  wrong_address           address-shaped, but a different house number or street
  non_address_value       non-null value that is not house-number + street
  unscorable              no parsed object / missing key / invalid type
Address outcome (no-exact-address complaints):
  correct_null            pred is null
  fabricated_address      address-shaped value (house number + street) that is NOT a verbatim
                          (normalized) span of the complaint                 <- fabrication
  non_address_value       any other non-null value, incl. complaint text copied into the field
                          (e.g. "2500 block of Madison"): a spec violation, flagged, but nothing
                          was invented, so it is not counted as fabrication
  unscorable
normalized(): lowercase, periods removed, whitespace collapsed.
address-shaped: ^\\d{1,6}[a-z]?\\s+<alphanumeric>  (house number followed by a word)

Other fields (only when a parsed object has the key):
  category_correct / urgency_correct   exact string equality with gold
  location_exact                       normalized equality with the gold location string
  location_anchor                      the grounded content is present (per location_type):
      exact_address   gold address contained;  intersection  both street names contained
      landmark        landmark name contained; block_range   gold core (w/o parenthetical) contained
      street_only     street contained;       vague           not address-shaped and != "unspecified"
      missing         == "unspecified"
  clarification_correct   ("location clarification required" in summary) == needs_clarification
  summary_unsupported_address   an address-shaped span in the summary not present in the complaint
  summary_exact            normalized equality (diagnostic only; README binding rule 5)
  core_correct   contract_compliant AND category AND urgency AND address outcome in
                 {retained_exact, correct_null} AND clarification_correct
  full_exact_match   all five fields equal gold (location/summary normalized)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import generate_instruction_data as gen

SCORING_RULES_VERSION = "2026-09-28.v1"
STATES = ("base", "sft", "dpo")
CLAR = "location clarification required"
SUFFIXES = {"st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard", "dr", "drive", "ln", "lane",
            "way", "ct", "court", "pl", "place"}
ADDRESS_RE = re.compile(r"^\d{1,6}[a-z]?\s+[a-z0-9]", re.I)
ADDRESS_SPAN_RE = re.compile(r"\b\d{1,6}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b")
RELATION_PREFIXES = sorted({norm for _, norm in gen.LANDMARK_RELATIONS}, key=len, reverse=True)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace(".", "")).strip()


def strip_paren(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()


# ---------------------------------------------------------------------------
def parse_output(text: str) -> tuple[dict | None, str, str]:
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, "strict", ""
        err = f"top-level JSON is {type(obj).__name__}"
    except json.JSONDecodeError as e:
        err = f"strict parse failed: {e.msg}"
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            obj, _ = dec.raw_decode(t[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, "extracted", err
    return None, "none", err


def street_core(addr: str) -> tuple[str, list[str]]:
    toks = norm(addr).split()
    number, rest = toks[0], toks[1:]
    if rest and rest[-1] in SUFFIXES:
        rest = rest[:-1]
    return number, rest


def address_outcome(pred, gold: str | None, key_present: bool, complaint: str = "") -> str:
    if not key_present or not (pred is None or isinstance(pred, str)):
        return "unscorable"
    if gold is None:
        if pred is None:
            return "correct_null"
        if ADDRESS_RE.match(pred.strip()) and norm(pred) not in norm(complaint):
            return "fabricated_address"
        return "non_address_value"
    if pred is None:
        return "over_null"
    if norm(pred) == norm(gold):
        return "retained_exact"
    if ADDRESS_RE.match(pred.strip()):
        return "retained_suffix_variant" if street_core(pred) == street_core(gold) else "wrong_address"
    return "non_address_value"


def location_anchor(pred: str, gold_loc: str, loc_type: str) -> bool:
    p = norm(pred)
    g = norm(strip_paren(gold_loc))
    if loc_type == "missing":
        return p == norm(gen.LOCATION_UNSPECIFIED)
    if loc_type == "vague":
        return bool(p) and not ADDRESS_RE.match(p) and p != norm(gen.LOCATION_UNSPECIFIED)
    if loc_type == "intersection":
        return all(part.strip() in p for part in g.split(" and "))
    if loc_type == "landmark":
        for rel in RELATION_PREFIXES:
            if g.startswith(rel + " "):
                g = g[len(rel) + 1:]
                break
        return g in p
    return g in p   # exact_address, block_range, street_only


def score_example(rec: dict, raw_text: str, secondary_category: str | None = None) -> dict:
    gold = json.loads(rec["response"])
    obj, mode, perr = parse_output(raw_text)
    flags: list[str] = []
    s: dict = {"id": rec["id"], "parse_mode": mode, "parse_error": perr}

    keys = list(obj) if obj is not None else []
    s["key_order_exact"] = tuple(keys) == gen.TICKET_KEYS
    s["missing_keys"] = [k for k in gen.TICKET_KEYS if k not in keys]
    s["extra_keys"] = [k for k in keys if k not in gen.TICKET_KEYS]
    g = (lambda k: obj.get(k) if obj is not None else None)
    s["category_valid"] = g("category") in gen.CATEGORIES
    s["urgency_valid"] = g("urgency") in gen.URGENCY_LEVELS
    loc_ok = isinstance(g("location"), str) and bool(g("location").strip())
    sum_ok = isinstance(g("summary"), str) and bool(g("summary").strip())
    addr_type_ok = obj is not None and "address_or_null" in obj and (g("address_or_null") is None or isinstance(g("address_or_null"), str))
    s["contract_compliant"] = (mode == "strict" and s["key_order_exact"] and s["category_valid"] and s["urgency_valid"]
                               and loc_ok and sum_ok and addr_type_ok)
    if mode == "extracted":
        flags.append("json_extracted_not_strict")

    s["category_correct"] = g("category") == gold["category"]
    s["urgency_correct"] = g("urgency") == gold["urgency"]
    if (not s["category_correct"] and rec.get("multi_issue") and secondary_category
            and g("category") == secondary_category):
        flags.append("category_is_other_mentioned_issue")

    s["address_outcome"] = address_outcome(g("address_or_null"), gold["address_or_null"],
                                           obj is not None and "address_or_null" in obj, rec["complaint"])
    s["pred_address"] = g("address_or_null")
    if s["address_outcome"] == "retained_suffix_variant":
        flags.append("address_suffix_variant")
    if s["address_outcome"] == "non_address_value":
        flags.append("address_non_address_value")

    if loc_ok:
        s["location_exact"] = norm(g("location")) == norm(gold["location"])
        s["location_anchor"] = location_anchor(g("location"), gold["location"], rec["location_type"])
        if s["location_anchor"] and not s["location_exact"]:
            flags.append("location_anchor_only")
    else:
        s["location_exact"] = s["location_anchor"] = False

    if sum_ok:
        summ = g("summary")
        s["clarification_correct"] = (CLAR in summ.lower()) == rec["needs_clarification"]
        spans = [m.group(0) for m in ADDRESS_SPAN_RE.finditer(summ)]
        unsupported = [sp for sp in spans if norm(sp) not in norm(rec["complaint"])]
        s["summary_unsupported_address"] = bool(unsupported)
        if unsupported:
            flags.append("summary_unsupported_address:" + "|".join(unsupported))
        s["summary_exact"] = norm(summ) == norm(gold["summary"])
    else:
        s["clarification_correct"] = s["summary_unsupported_address"] = s["summary_exact"] = False

    s["core_correct"] = bool(s["contract_compliant"] and s["category_correct"] and s["urgency_correct"]
                             and s["address_outcome"] in ("retained_exact", "correct_null") and s["clarification_correct"])
    def field_eq(p, q):
        return norm(p) == norm(q) if isinstance(p, str) and isinstance(q, str) else p == q
    s["full_exact_match"] = bool(obj is not None and not s["missing_keys"]
                                 and all(field_eq(obj[k], gold[k]) for k in gen.TICKET_KEYS))
    s["flags"] = flags
    return s


# ---------------------------------------------------------------------------
def rate(num: int, den: int) -> dict:
    return {"n": num, "d": den, "rate": round(num / den, 4) if den else None}


def aggregate(records: list[dict], scored: dict[str, list[dict]], majority_urgency: dict[str, str]) -> dict:
    """scored[state] is aligned with records (same order)."""
    exact = [i for i, r in enumerate(records) if r["has_exact_address"]]
    noaddr = [i for i, r in enumerate(records) if not r["has_exact_address"]]
    out = {"population": {"total": len(records), "exact_address": len(exact), "no_address": len(noaddr),
                          "no_address_by_condition": dict(Counter(records[i]["location_type"] for i in noaddr))}}
    base_urg = sum(majority_urgency.get(r["category"]) == r["urgency"] for r in records)
    out["majority_urgency_per_category_baseline"] = rate(base_urg, len(records))
    for st, sc in scored.items():
        a: dict = {}
        cnt = lambda key, idx=range(len(records)): sum(bool(sc[i][key]) for i in idx)  # noqa: E731
        N = len(records)
        a["parse_mode"] = dict(Counter(s["parse_mode"] for s in sc))
        for k in ("contract_compliant", "key_order_exact", "category_correct", "urgency_correct", "location_exact",
                  "location_anchor", "clarification_correct", "summary_exact", "core_correct", "full_exact_match"):
            a[k] = rate(cnt(k), N)
        a["summary_unsupported_address"] = rate(cnt("summary_unsupported_address"), N)
        ex_out = Counter(sc[i]["address_outcome"] for i in exact)
        na_out = Counter(sc[i]["address_outcome"] for i in noaddr)
        # PRIMARY behavioral fabrication metric: invented house-number-shaped exact address not
        # supported verbatim by the complaint.
        a["A_unsupported_exact_address_fabrication"] = rate(na_out["fabricated_address"], len(noaddr))
        # BROADER contract metric (Step-1 field contract): ANY non-null address_or_null on a
        # no-exact-address complaint. Copied block ranges / landmarks / intersections count here,
        # but are NOT relabelled as fabricated exact addresses.
        a["invalid_non_null_address_field"] = rate(na_out["fabricated_address"] + na_out["non_address_value"],
                                                   len(noaddr))
        a["B_retention_exact"] = rate(ex_out["retained_exact"], len(exact))
        a["B_retention_incl_suffix_variant"] = rate(ex_out["retained_exact"] + ex_out["retained_suffix_variant"], len(exact))
        a["C_over_null"] = rate(ex_out["over_null"], len(exact))
        a["correct_null"] = rate(na_out["correct_null"], len(noaddr))
        a["address_correct_overall"] = rate(ex_out["retained_exact"] + na_out["correct_null"], N)
        a["exact_address_outcomes"] = dict(ex_out)
        a["no_address_outcomes"] = dict(na_out)
        by_cond = defaultdict(Counter)
        for i in noaddr:
            r = records[i]
            key = r["location_type"] + ("" if r["location_type"] != "street_only"
                                        else ("/clarify" if r["needs_clarification"] else "/actionable"))
            by_cond[key][sc[i]["address_outcome"]] += 1
            by_cond[key]["clarification_correct"] += bool(sc[i]["clarification_correct"])
            by_cond[key]["n"] += 1
        a["no_address_by_condition"] = {k: dict(v) for k, v in sorted(by_cond.items())}
        by_urg = defaultdict(lambda: [0, 0])
        for i, r in enumerate(records):
            by_urg[r["urgency"]][0] += bool(sc[i]["urgency_correct"])
            by_urg[r["urgency"]][1] += 1
        a["urgency_by_level"] = {u: rate(*by_urg[u]) for u in gen.URGENCY_LEVELS if u in by_urg}
        a["flag_counts"] = dict(Counter(f.split(":")[0] for s in sc for f in s["flags"]))
        out[st] = a
    return out


def summary_markdown(agg: dict) -> str:
    st = [s for s in STATES if s in agg]
    f = lambda d: f"{d['n']}/{d['d']} ({100 * d['rate']:.1f}%)" if d["d"] else "n/a"  # noqa: E731
    pop = agg["population"]
    L = [f"Population: {pop['total']} held-out complaints; {pop['exact_address']} with an exact address, "
         f"{pop['no_address']} without ({pop['no_address_by_condition']}).", "",
         "| Metric | " + " | ".join(s.upper() for s in st) + " |", "|---|" + "---|" * len(st)]
    rows = [("**A. Unsupported exact-address fabrication** (primary; no-address complaints) ↓",
             "A_unsupported_exact_address_fabrication"),
            ("**B. Address retention, exact** (exact-address complaints) ↑", "B_retention_exact"),
            ("**C. Over-null** (null despite exact address) ↓", "C_over_null"),
            ("Invalid non-null address field (contract; any non-null on no-address complaints) ↓",
             "invalid_non_null_address_field"),
            ("Retention incl. suffix variants", "B_retention_incl_suffix_variant"),
            ("Correct null (no-address complaints)", "correct_null"),
            ("Contract-compliant JSON (strict)", "contract_compliant"),
            ("Category correct", "category_correct"), ("Urgency correct", "urgency_correct"),
            ("Location exact", "location_exact"), ("Location anchor (grounded content)", "location_anchor"),
            ("Clarification sentence correct", "clarification_correct"),
            ("Summary contains unsupported address ↓", "summary_unsupported_address"),
            ("Core-correct ticket", "core_correct"), ("Full exact match", "full_exact_match")]
    for label, key in rows:
        L.append(f"| {label} | " + " | ".join(f(agg[s][key]) for s in st) + " |")
    L.append(f"| Urgency baseline: majority per category (train-fitted) | {f(agg['majority_urgency_per_category_baseline'])} "
             + "| — " * (len(st) - 1) + "|")
    L += ["", "Read separately: fabrication (A), address-field contract (invalid non-null), interface contract "
          "(strict JSON), retention (B) and over-null (C) are distinct claims. Controlled model-state comparison on "
          "one NF4 base with identical inference settings; not a stock-inference benchmark. DPO training-set reward "
          "accuracy is not held-out evidence; the 0.663 response-only digit-count shortcut is a known risk."]
    return "\n".join(L)
