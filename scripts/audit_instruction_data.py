#!/usr/bin/env python3
"""Audit the CivicDesk instruction dataset before it is accepted for training.

Checks fall into two classes:
  HARD FAILURES (exit code 1): output-contract violations, metadata/ticket
    mismatches, fabricated or altered addresses, exact duplicates across the
    split, family leakage, non-deterministic regeneration.
  WARNINGS (exit code 0, printed for human review): distribution skew,
    repeated openings, near-duplicates, template concentration.

Thresholds are fixed constants below, chosen before inspecting the data.

Usage:
    python scripts/audit_instruction_data.py
    python scripts/audit_instruction_data.py --skip-regen   # skip subprocess regeneration check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_instruction_data as gen  # noqa: E402  (shared contract constants)

# --- warning thresholds (fixed a priori) ---
OPENING_SHARE_WARN = 0.10       # any complaint 3-word opening > 10% of records
SUMMARY_OPENING_SHARE_WARN = 0.10
NEAR_DUP_JACCARD_WARN = 0.70    # char-4gram Jaccard, eval vs train
WITHIN_TRAIN_NEAR_DUP_WARN = 0.85
FAMILY_SHARE_WARN = 0.05        # a single family > 5% of train records
CATEGORY_MIN_SHARE_WARN = 0.05
DUP_RESPONSE_SHARE_WARN = 0.02
MIN_EVAL_NO_ADDRESS = 20


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(n: int, d: int) -> str:
    return f"{n:4d} ({100 * n / d:5.1f}%)" if d else f"{n:4d}"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def dist(title: str, counter: Counter, total: int, order: list | None = None) -> None:
    print(f"\n{title}")
    keys = order if order else [k for k, _ in counter.most_common()]
    for k in keys:
        print(f"  {str(k):34s} {pct(counter.get(k, 0), total)}")


def length_stats(name: str, values: list[int]) -> None:
    q = statistics.quantiles(values, n=4)
    print(f"  {name:28s} min={min(values):4d}  p25={q[0]:6.1f}  median={statistics.median(values):6.1f}  "
          f"p75={q[2]:6.1f}  max={max(values):4d}  mean={statistics.mean(values):6.1f}")


def shingles(text: str, k: int = 4) -> set[str]:
    t = re.sub(r"\s+", " ", text.lower())
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def opening(text: str, n: int = 3) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower())[:n])


def norm_addr(s: str) -> str:
    return re.sub(r"[.\s]+", " ", s.lower()).strip()


def check_record(r: dict) -> list[str]:
    """Output-contract + consistency checks for one record. Returns error strings."""
    errs = []
    try:
        t = json.loads(r["response"])
    except json.JSONDecodeError as e:
        return [f"response is not valid JSON: {e}"]
    if not isinstance(t, dict):
        return ["response JSON is not an object"]
    if tuple(t.keys()) != gen.TICKET_KEYS:
        errs.append(f"key order/set mismatch: {list(t.keys())}")
    if t.get("category") not in gen.CATEGORIES:
        errs.append(f"invalid category {t.get('category')!r}")
    if t.get("urgency") not in gen.URGENCY_LEVELS:
        errs.append(f"invalid urgency {t.get('urgency')!r}")
    loc, summ, addr = t.get("location"), t.get("summary"), t.get("address_or_null")
    if not isinstance(loc, str) or not loc.strip():
        errs.append("location must be a non-empty string (never null)")
    if not isinstance(summ, str) or not summ.strip():
        errs.append("summary must be a non-empty string")
    if addr is not None and not isinstance(addr, str):
        errs.append("address_or_null must be string or null")

    # metadata <-> ticket consistency
    if t.get("category") != r["category"] or t.get("urgency") != r["urgency"]:
        errs.append("metadata category/urgency differ from ticket")
    if (addr is not None) != r["has_exact_address"]:
        errs.append("address_or_null presence disagrees with has_exact_address")
    if (r["location_type"] == "exact_address") != r["has_exact_address"]:
        errs.append("location_type/has_exact_address mismatch")
    if (loc == gen.LOCATION_UNSPECIFIED) != (r["location_type"] == "missing"):
        errs.append("location 'unspecified' must appear iff location_type == missing")
    if isinstance(summ, str) and (gen.CLARIFICATION_SENTENCE in summ) != r["needs_clarification"]:
        errs.append("clarification sentence disagrees with needs_clarification")

    # non-fabrication: address must be copied from the complaint (case/period-insensitive)
    if isinstance(addr, str):
        if norm_addr(addr) not in norm_addr(r["complaint"]):
            errs.append(f"address {addr!r} not found verbatim in complaint")
        if not re.match(r"^\d+ \S", addr):
            errs.append(f"address {addr!r} is not house-number + street")
    # summaries must not carry location/address details
    if isinstance(summ, str) and re.search(r"\b\d{1,5} [A-Z][a-z]+ (St|Ave|Rd|Blvd|Dr|Ln|Way|Ct|Pl)\b", summ):
        errs.append("summary contains an address-like string")
    if r["location_type"] == "missing" and re.search(r"\b\d{2,5}\b", r["complaint"]) and \
            re.search(r"\b\d{2,5} [a-z]+ (st|ave|rd|street|avenue|road)\b", r["complaint"].lower()):
        errs.append("location_type missing but complaint contains an address-like string")
    return errs


def regen_check(train_path: Path, eval_path: Path) -> tuple[bool, str]:
    """Regenerate in a subprocess with a different PYTHONHASHSEED and compare bytes."""
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, PYTHONHASHSEED="12345")
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_instruction_data.py"),
                        "--out-dir", td], check=True, env=env, capture_output=True)
        ok = True
        lines = []
        for name, orig in (("instruction_train.jsonl", train_path), ("instruction_eval.jsonl", eval_path)):
            h1 = hashlib.sha256(orig.read_bytes()).hexdigest()
            h2 = hashlib.sha256((Path(td) / name).read_bytes()).hexdigest()
            ok &= h1 == h2
            lines.append(f"  {name:26s} committed={h1[:16]}  regenerated={h2[:16]}  {'MATCH' if h1 == h2 else 'DIFFERENT'}")
        return ok, "\n".join(lines)


def main() -> int:
    cfg = json.loads((REPO_ROOT / "configs" / "adapter_config.json").read_text())["data"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=REPO_ROOT / cfg["instruction_train_path"])
    ap.add_argument("--eval", type=Path, default=REPO_ROOT / cfg["instruction_eval_path"])
    ap.add_argument("--skip-regen", action="store_true")
    args = ap.parse_args()

    train, evl = load(args.train), load(args.eval)
    allr = train + evl
    N = len(allr)
    failures: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    section("1. COUNTS")
    print(f"  total records : {N}")
    print(f"  train         : {len(train)}")
    print(f"  eval          : {len(evl)}")

    # ------------------------------------------------------------------
    section("2. OUTPUT CONTRACT / CONSISTENCY")
    n_bad = 0
    for r in allr:
        errs = check_record(r)
        if errs:
            n_bad += 1
            for e in errs:
                failures.append(f"{r['id']}: {e}")
    print(f"  records with contract/consistency errors: {n_bad} / {N}")

    # ------------------------------------------------------------------
    section("3. DISTRIBUTIONS")
    for name, recs in (("ALL", allr), ("TRAIN", train), ("EVAL", evl)):
        n = len(recs)
        print(f"\n--- {name} (n={n}) ---")
        dist("category", Counter(r["category"] for r in recs), n, gen.CATEGORIES)
        dist("urgency", Counter(r["urgency"] for r in recs), n, gen.URGENCY_LEVELS)
        na = sum(not r["has_exact_address"] for r in recs)
        print(f"\nexact address present : {pct(n - na, n)}")
        print(f"no exact address      : {pct(na, n)}")
        dist("location_type", Counter(r["location_type"] for r in recs), n, gen.LOCATION_TYPES)
        nulls = [r for r in recs if not r["has_exact_address"]]
        print(f"\nno-address records: clarification required {pct(sum(r['needs_clarification'] for r in nulls), len(nulls))}, "
              f"actionable (no clarification) {pct(sum(not r['needs_clarification'] for r in nulls), len(nulls))}")
        print(f"multi-issue           : {pct(sum(r['multi_issue'] for r in recs), n)}")

    cat_all = Counter(r["category"] for r in allr)
    for c in gen.CATEGORIES:
        if cat_all[c] / N < CATEGORY_MIN_SHARE_WARN:
            warnings.append(f"category {c} under-represented ({cat_all[c]}/{N})")
    for split, recs in (("train", train), ("eval", evl)):
        missing_urg = [u for u in gen.URGENCY_LEVELS if not any(r["urgency"] == u for r in recs)]
        if missing_urg:
            warnings.append(f"{split}: urgency levels absent: {missing_urg}")

    eval_null = [r for r in evl if not r["has_exact_address"]]
    if len(eval_null) < MIN_EVAL_NO_ADDRESS:
        failures.append(f"eval has {len(eval_null)} no-address records (< {MIN_EVAL_NO_ADDRESS})")
    missing_conds = [c for c in gen.NO_ADDRESS_TYPES if not any(r["location_type"] == c for r in eval_null)]
    if missing_conds:
        failures.append(f"eval missing no-address conditions: {missing_conds}")

    section("3b. EVAL NO-ADDRESS DETAIL (primary safety-metric population)")
    print(f"  eval no-address records: {len(eval_null)} (required >= {MIN_EVAL_NO_ADDRESS})")
    tab = Counter((r["location_type"], r["needs_clarification"]) for r in eval_null)
    for c in gen.NO_ADDRESS_TYPES:
        print(f"  {c:14s} clarify={tab.get((c, True), 0):2d}  actionable={tab.get((c, False), 0):2d}")
    # street_only is the one condition whose clarification rule depends on the issue
    # (point vs street-wide). If every record lands on one side, that rule is untaught.
    for split, recs in (("train", train), ("eval", evl)):
        so = Counter(r["needs_clarification"] for r in recs if r["location_type"] == "street_only")
        print(f"  [{split}] street_only: actionable={so.get(False, 0)}  clarification={so.get(True, 0)}")
        if not so.get(False) or not so.get(True):
            msg = (f"{split}: street_only lacks an actionable or a clarification case "
                   f"(actionable={so.get(False, 0)}, clarify={so.get(True, 0)})")
            (failures if split == "eval" else warnings).append(msg)
    # street-only actionable records must come from street-wide families, clarification from point families
    for r in allr:
        if r["location_type"] == "street_only":
            fam_cat, fam_i = r["family_id"].split("/")
            point = gen.SCENARIOS[fam_cat][int(fam_i)].point
            if r["needs_clarification"] != point:
                failures.append(f"{r['id']}: street_only clarification {r['needs_clarification']} "
                                f"inconsistent with family point={point}")

    section("3c. URGENCY x CATEGORY (all records)")
    header = "  " + f"{'category':24s}" + "".join(f"{u:>10s}" for u in gen.URGENCY_LEVELS)
    print(header)
    for c in gen.CATEGORIES:
        row = Counter(r["urgency"] for r in allr if r["category"] == c)
        print("  " + f"{c:24s}" + "".join(f"{row.get(u, 0):10d}" for u in gen.URGENCY_LEVELS))
    single_urg = [c for c in gen.CATEGORIES if len({r['urgency'] for r in allr if r['category'] == c}) == 1]
    if single_urg:
        warnings.append(f"categories with a single urgency level (urgency inferable from category): {single_urg}")

    section("3d. URGENCY CLAIM vs LABEL (keyword-shortcut check)")
    claims = [r for r in allr if "urgency_claim" in r["noise"]]
    print(f"  records with an injected resident urgency claim: {len(claims)}")
    urgent_words = re.compile(r"urgent|emergency|asap|today", re.I)
    calm_words = re.compile(r"not urgent|no rush|low priority|not a big deal|whenever|no hurry", re.I)
    for lab in gen.URGENCY_LEVELS:
        u = sum(1 for r in claims if r["urgency"] == lab and urgent_words.search(r["complaint"]) and not calm_words.search(r["complaint"]))
        cl = sum(1 for r in claims if r["urgency"] == lab and calm_words.search(r["complaint"]))
        print(f"  label={lab:10s} urgent-sounding claim={u:3d}   calm-sounding claim={cl:3d}")

    # ------------------------------------------------------------------
    section("4. LENGTH STATISTICS")
    print("\ncomplaint")
    length_stats("words (all)", [len(r["complaint"].split()) for r in allr])
    length_stats("chars (all)", [len(r["complaint"]) for r in allr])
    print("\nresponse (JSON ticket)")
    length_stats("chars (all)", [len(r["response"]) for r in allr])
    length_stats("summary words (all)", [len(json.loads(r["response"])["summary"].split()) for r in allr])
    print("  (token counts are measured with the real tokenizer in Step 3)")

    # ------------------------------------------------------------------
    section("5. DUPLICATES")
    comp = Counter(r["complaint"] for r in allr)
    resp = Counter(r["response"] for r in allr)
    dup_c = sum(v - 1 for v in comp.values() if v > 1)
    dup_r = sum(v - 1 for v in resp.values() if v > 1)
    print(f"  duplicate complaints (exact)                : {dup_c}")
    print(f"  duplicate complaints (case/punct-normalized): "
          f"{sum(v - 1 for v in Counter(re.sub(r'[^a-z0-9]', '', r['complaint'].lower()) for r in allr).values() if v > 1)}")
    print(f"  duplicate responses (exact)                 : {dup_r}")
    if dup_c:
        failures.append(f"{dup_c} exact duplicate complaints")
    if dup_r / N > DUP_RESPONSE_SHARE_WARN:
        warnings.append(f"duplicate responses {dup_r}/{N} exceed {DUP_RESPONSE_SHARE_WARN:.0%}")
    if dup_r:
        print("  duplicated responses (count x response):")
        for k, v in resp.most_common():
            if v > 1:
                print(f"    {v}x {k}")
    cross = {r["complaint"] for r in train} & {r["complaint"] for r in evl}
    if cross:
        failures.append(f"{len(cross)} complaints appear in both train and eval")

    # ------------------------------------------------------------------
    section("6. OPENING PATTERNS")
    print("\ncomplaint first-3-words (top 12)")
    op = Counter(opening(r["complaint"]) for r in allr)
    for k, v in op.most_common(12):
        print(f"  {pct(v, N)}  {k!r}")
        if v / N > OPENING_SHARE_WARN:
            warnings.append(f"complaint opening {k!r} is {v}/{N}")
    print("\nsummary first-3-words (top 12)")
    so = Counter(opening(json.loads(r["response"])["summary"]) for r in allr)
    for k, v in so.most_common(12):
        print(f"  {pct(v, N)}  {k!r}")
        if v / N > SUMMARY_OPENING_SHARE_WARN:
            warnings.append(f"summary opening {k!r} is {v}/{N}")
    print(f"\n  distinct summary openings: {len(so)}")
    print("\nresponse raw opening (first 14 chars) — identical by design (fixed JSON contract):")
    for k, v in Counter(r["response"][:14] for r in allr).most_common(3):
        print(f"  {pct(v, N)}  {k!r}")

    # ------------------------------------------------------------------
    section("7. TEMPLATE / FAMILY / STYLE DISTRIBUTION")
    dist("style", Counter(r["style"] for r in allr), N)
    dist("frame", Counter(r["frame"] for r in allr), N)
    dist("noise op (records containing)", Counter(n for r in allr for n in r["noise"]), N)
    fam = Counter(r["family_id"] for r in train)
    fv = sorted(fam.values())
    print(f"\ntrain families: {len(fam)}  records/family min={fv[0]} median={statistics.median(fv)} max={fv[-1]}")
    for f, v in fam.items():
        if v / len(train) > FAMILY_SHARE_WARN:
            warnings.append(f"family {f} is {v}/{len(train)} of train")
    efam = Counter(r["family_id"] for r in evl)
    print(f"eval families : {len(efam)}  records/family: {dict(sorted(Counter(efam.values()).items()))} (size: n_families)")

    # ------------------------------------------------------------------
    section("8. TRAIN/EVAL FAMILY LEAKAGE")
    train_f = {r["family_id"] for r in train} | {r["secondary_family_id"] for r in train if r["secondary_family_id"]}
    eval_f = {r["family_id"] for r in evl} | {r["secondary_family_id"] for r in evl if r["secondary_family_id"]}
    leak = sorted(train_f & eval_f)
    print(f"  families (incl. secondary issues) in train: {len(train_f)}, eval: {len(eval_f)}, shared: {len(leak)}")
    if leak:
        failures.append(f"family leakage: {leak}")
    # issue-clause leakage: any eval issue paraphrase text present in a train complaint
    eval_issue_texts = {p for f in {r['family_id'] for r in evl}
                        for p in gen.SCENARIOS[f.split('/')[0]][int(f.split('/')[1])].issue}
    train_text = " || ".join(r["complaint"].lower() for r in train)
    leaked_phr = sorted(p for p in eval_issue_texts if p.lower() in train_text)
    print(f"  eval-family issue paraphrases found verbatim in train complaints: {len(leaked_phr)}")
    for p in leaked_phr:
        warnings.append(f"eval issue phrase appears in a train complaint: {p!r}")
    train_addr = {json.loads(r["response"])["address_or_null"] for r in train} - {None}
    eval_addr = {json.loads(r["response"])["address_or_null"] for r in evl} - {None}
    print(f"  exact addresses shared train/eval (informational): {len(train_addr & eval_addr)}")

    # ------------------------------------------------------------------
    section("9. NEAR-DUPLICATE CHECK (char-4gram Jaccard)")
    tr_sh = [shingles(r["complaint"]) for r in train]
    ev_sh = [shingles(r["complaint"]) for r in evl]
    best = []
    for i, e in enumerate(ev_sh):
        j, s = max(((j, jaccard(e, t)) for j, t in enumerate(tr_sh)), key=lambda x: x[1])
        best.append((s, i, j))
    best.sort(reverse=True)
    sims = [b[0] for b in best]
    print(f"  eval->nearest-train similarity: max={sims[0]:.3f}  median={statistics.median(sims):.3f}  "
          f"(warn >= {NEAR_DUP_JACCARD_WARN})")
    print("  top 3 eval/train pairs:")
    for s, i, j in best[:3]:
        print(f"    {s:.3f}\n      EVAL : {evl[i]['complaint'][:110]}\n      TRAIN: {train[j]['complaint'][:110]}")
    n_near = sum(1 for s in sims if s >= NEAR_DUP_JACCARD_WARN)
    if n_near:
        warnings.append(f"{n_near} eval complaints have a train near-duplicate (Jaccard >= {NEAR_DUP_JACCARD_WARN})")
    within = 0
    for a in range(len(tr_sh)):
        for b in range(a + 1, len(tr_sh)):
            if jaccard(tr_sh[a], tr_sh[b]) >= WITHIN_TRAIN_NEAR_DUP_WARN:
                within += 1
    print(f"  within-train pairs with Jaccard >= {WITHIN_TRAIN_NEAR_DUP_WARN}: {within}")
    if within:
        warnings.append(f"{within} within-train near-duplicate pairs")

    # ------------------------------------------------------------------
    section("10. DETERMINISTIC REGENERATION")
    if args.skip_regen:
        print("  skipped (--skip-regen)")
    else:
        ok, report = regen_check(args.train, args.eval)
        print(report)
        if not ok:
            failures.append("regeneration with same seed (different PYTHONHASHSEED) is not byte-identical")

    # ------------------------------------------------------------------
    section("RESULT")
    print(f"  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"    - {w}")
    print(f"  HARD FAILURES ({len(failures)}):")
    for f in failures[:50]:
        print(f"    - {f}")
    if len(failures) > 50:
        print(f"    ... {len(failures) - 50} more")
    print(f"\n  AUDIT {'FAILED' if failures else 'PASSED (review warnings above)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
