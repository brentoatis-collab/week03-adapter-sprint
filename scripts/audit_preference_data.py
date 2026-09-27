#!/usr/bin/env python3
"""Audit the CivicDesk DPO preference dataset before it is accepted.

HARD FAILURES (exit 1): count, schema/key order, chosen == rejected, unexpected field
differences, address-grounding violations, duplicates, leakage into eval families or
instruction complaints, non-deterministic regeneration, and any shortcut classifier at
or above SHORTCUT_FAIL.
WARNINGS: length imbalance, shortcut classifiers at or above SHORTCUT_WARN, skew.

The shortcut diagnostic asks: can chosen vs rejected be predicted from the RESPONSE ALONE
(surface features, no prompt)? A response-only classifier well above chance means DPO
could improve its objective without reading the complaint. Thresholds are fixed a priori.

Usage:
    python scripts/audit_preference_data.py              # token counts if transformers is importable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import generate_instruction_data as gen  # noqa: E402
import generate_preferences as gp  # noqa: E402

# --- thresholds (fixed before inspecting the data) ---
SHORTCUT_WARN = 0.60
SHORTCUT_FAIL = 0.70
LENGTH_RATIO_TOL = 0.15          # |rejected/chosen - 1| beyond this is "imbalanced"
LENGTH_IMBALANCED_SHARE_WARN = 0.10
NEAR_DUP_JACCARD_WARN = 0.70
CV_FOLDS = 5


def section(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def pct(n: int, d: int) -> str:
    return f"{n:4d} ({100 * n / d:5.1f}%)" if d else f"{n:4d}"


def stats(vals: list[float]) -> str:
    q = statistics.quantiles(vals, n=4) if len(vals) > 1 else [vals[0]] * 3
    return (f"min={min(vals):7.2f} p25={q[0]:7.2f} median={statistics.median(vals):7.2f} "
            f"p75={q[2]:7.2f} max={max(vals):7.2f} mean={statistics.mean(vals):7.2f}")


def norm(s: str) -> str:
    return re.sub(r"[.\s]+", " ", s.lower()).strip()


def shingles(t: str, k: int = 4) -> set[str]:
    t = re.sub(r"\s+", " ", t.lower())
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


# ---------------------------------------------------------------------------
# Shortcut diagnostics (pure Python; no sklearn dependency)
# ---------------------------------------------------------------------------
def response_features(resp: str, tok=None) -> dict[str, float]:
    t = json.loads(resp)
    addr, loc = t["address_or_null"], t["location"]
    f = {
        "char_len": len(resp),
        "word_len": len(resp.split()),
        "digit_count": sum(ch.isdigit() for ch in resp),
        "address_is_null": float(addr is None),
        "location_starts_with_digit": float(bool(re.match(r"\d", loc))),
        "location_has_parenthetical": float("(" in loc),
        "location_len": len(loc),
        "address_has_suffix": float(bool(addr) and bool(re.search(r"\b(St|Ave|Rd|Blvd|Dr|Ln|Way|Ct|Pl|Street|Avenue|Road|"
                                                                  r"Boulevard|Drive|Lane|Court|Place)$", addr))),
        "clarification_sentence": float(gen.CLARIFICATION_SENTENCE in t["summary"]),
    }
    if tok is not None:
        f["token_len"] = len(tok(resp, add_special_tokens=False)["input_ids"])
    return f


def folds_by_group(n_groups: int, k: int, seed: int = 0) -> list[list[int]]:
    idx = list(range(n_groups))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


def stump_fit(xs: list[float], ys: list[int]) -> tuple[float, int]:
    """Best threshold/direction on training data. Predict 1 if dir*(x - thr) > 0."""
    cands = sorted(set(xs))
    thrs = [(a + b) / 2 for a, b in zip(cands, cands[1:])] or [cands[0]]
    best = (0.0, thrs[0], 1)
    for thr in thrs:
        for d in (1, -1):
            acc = sum((1 if d * (x - thr) > 0 else 0) == y for x, y in zip(xs, ys)) / len(ys)
            if acc > best[0]:
                best = (acc, thr, d)
    return best[1], best[2]


def logreg_fit(X: list[list[float]], y: list[int], epochs: int = 400, lr: float = 0.1, l2: float = 1e-3):
    n, d = len(X), len(X[0])
    mu = [statistics.mean(r[j] for r in X) for j in range(d)]
    sd = [statistics.pstdev([r[j] for r in X]) or 1.0 for j in range(d)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(d)] for r in X]
    w, b = [0.0] * d, 0.0
    for _ in range(epochs):
        gw, gb = [0.0] * d, 0.0
        for zi, yi in zip(Z, y):
            p = 1 / (1 + math.exp(-max(-30, min(30, sum(wj * zj for wj, zj in zip(w, zi)) + b))))
            e = p - yi
            for j in range(d):
                gw[j] += e * zi[j]
            gb += e
        w = [wj - lr * (gj / n + l2 * wj) for wj, gj in zip(w, gw)]
        b -= lr * gb / n
    return lambda r: 1 if sum(wj * (r[j] - mu[j]) / sd[j] for j, wj in enumerate(w)) + b > 0 else 0


def grouped_cv(rows: list[tuple[int, list[float], int]], fit) -> float:
    """rows = (group, features, label). Both responses of a pair share a group (fold)."""
    groups = sorted({g for g, _, _ in rows})
    correct = total = 0
    for fold in folds_by_group(len(groups), CV_FOLDS):
        test_g = {groups[i] for i in fold}
        tr = [(x, y) for g, x, y in rows if g not in test_g]
        te = [(x, y) for g, x, y in rows if g in test_g]
        predict = fit([x for x, _ in tr], [y for _, y in tr])
        correct += sum(predict(x) == y for x, y in te)
        total += len(te)
    return correct / total


def run_shortcut_diagnostics(recs: list[dict], tok, warnings: list[str], failures: list[str]) -> None:
    feats_c = [response_features(r["chosen"], tok) for r in recs]
    feats_r = [response_features(r["rejected"], tok) for r in recs]
    names = list(feats_c[0])

    def flag(label: str, acc: float) -> str:
        if acc >= SHORTCUT_FAIL:
            failures.append(f"shortcut: {label} accuracy {acc:.3f} >= {SHORTCUT_FAIL}")
            return "FAIL"
        if acc >= SHORTCUT_WARN:
            warnings.append(f"shortcut: {label} accuracy {acc:.3f} >= {SHORTCUT_WARN}")
            return "WARN"
        return "ok"

    print("\n(a) POINTWISE, response only: classify a single response as chosen(1)/rejected(0).")
    print(f"    {CV_FOLDS}-fold CV grouped by pair; chance = 0.500; warn >= {SHORTCUT_WARN}, fail >= {SHORTCUT_FAIL}")
    rows_all = []
    for i, (fc, fr) in enumerate(zip(feats_c, feats_r)):
        rows_all.append((i, [fc[k] for k in names], 1))
        rows_all.append((i, [fr[k] for k in names], 0))
    for j, name in enumerate(names):
        rows = [(g, [x[j]], y) for g, x, y in rows_all]
        acc = grouped_cv(rows, lambda X, y: (lambda thr, d: (lambda r: 1 if d * (r[0] - thr) > 0 else 0))(
            *stump_fit([x[0] for x in X], y)))
        print(f"    stump[{name:28s}] {acc:.3f}  {flag('pointwise stump ' + name, acc)}")
    acc = grouped_cv(rows_all, logreg_fit)
    print(f"    logistic regression, all {len(names)} features   {acc:.3f}  {flag('pointwise logistic regression', acc)}")

    print("\n(b) PAIRWISE, response only: given both responses in random order, which is chosen?")
    print("    features = f(first) - f(second); same CV/thresholds.")
    order = random.Random(1)
    prow = []
    for i, (fc, fr) in enumerate(zip(feats_c, feats_r)):
        if order.random() < 0.5:
            prow.append((i, [fc[k] - fr[k] for k in names], 1))
        else:
            prow.append((i, [fr[k] - fc[k] for k in names], 0))
    for j, name in enumerate(names):
        rows = [(g, [x[j]], y) for g, x, y in prow]
        acc = grouped_cv(rows, lambda X, y: (lambda thr, d: (lambda r: 1 if d * (r[0] - thr) > 0 else 0))(
            *stump_fit([x[0] for x in X], y)))
        print(f"    diff stump[{name:23s}] {acc:.3f}  {flag('pairwise stump ' + name, acc)}")
    acc = grouped_cv(prow, logreg_fit)
    print(f"    logistic regression on differences     {acc:.3f}  {flag('pairwise logistic regression', acc)}")

    print("\n(c) FIXED PAIRWISE RULES (no fitting):")
    rules = {
        "shorter response (chars) is chosen": lambda c, r: (c["char_len"] < r["char_len"]) - (c["char_len"] > r["char_len"]),
        "longer response (chars) is chosen": lambda c, r: (c["char_len"] > r["char_len"]) - (c["char_len"] < r["char_len"]),
        "response with null address is chosen": lambda c, r: int(c["address_is_null"] - r["address_is_null"]),
        "fewer digits is chosen": lambda c, r: (c["digit_count"] < r["digit_count"]) - (c["digit_count"] > r["digit_count"]),
    }
    if tok is not None:
        rules["fewer tokens is chosen"] = lambda c, r: (c["token_len"] < r["token_len"]) - (c["token_len"] > r["token_len"])
    for name, rule in rules.items():
        s = [rule(c, r) for c, r in zip(feats_c, feats_r)]
        right, wrong, tie = s.count(1), s.count(-1), s.count(0)
        acc = (right + 0.5 * tie) / len(s)   # ties scored as coin flips
        print(f"    {name:40s} right={right:3d} wrong={wrong:3d} tie={tie:3d}  acc={acc:.3f}  {flag('rule ' + name, acc)}")

    print("\n(d) PROMPT-CONDITIONED REFERENCE RULE (the intended signal, for contrast):")
    ok = 0
    for r in recs:
        has_addr = r["has_exact_address"]
        def supported(resp: str) -> bool:
            a = json.loads(resp)["address_or_null"]
            return (a is None) if not has_addr else (a is not None and norm(a) in norm(r["prompt"]))
        ok += supported(r["chosen"]) and not supported(r["rejected"])
    print(f"    'choose the response whose address_or_null is supported by the complaint': {ok}/{len(recs)} "
          f"= {ok / len(recs):.3f}  (expected 1.000: the preference is decidable only by reading the prompt)")


# ---------------------------------------------------------------------------
def main() -> int:
    cfg = json.loads((REPO_ROOT / "configs" / "adapter_config.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=REPO_ROOT / cfg["data"]["preference_train_path"])
    ap.add_argument("--skip-regen", action="store_true")
    ap.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.path.read_text().splitlines() if l.strip()]
    failures: list[str] = []
    warnings: list[str] = []
    N = len(recs)

    tok = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg["model"]["model_id"])
        except Exception as e:  # tokenizer optional; reported, not hidden
            print(f"[info] token counts unavailable ({type(e).__name__}); using chars/words only")

    section("1. COUNT / COMPOSITION")
    print(f"  pairs: {N} (expected {cfg['data']['n_preference_examples']})")
    if N != cfg["data"]["n_preference_examples"]:
        failures.append(f"pair count {N} != {cfg['data']['n_preference_examples']}")
    pt = Counter(r["pair_type"] for r in recs)
    for t in gp.PAIR_TYPES:
        print(f"  {t:34s} {pct(pt[t], N)}   (configured {cfg['data']['preference_mix'][t]})")
        if pt[t] != cfg["data"]["preference_mix"][t]:
            failures.append(f"pair_type {t}: {pt[t]} != configured {cfg['data']['preference_mix'][t]}")
    print("\n  pair_type x location_type")
    tab = Counter((r["pair_type"], r["location_type"]) for r in recs)
    print("  " + f"{'':34s}" + "".join(f"{lt[:10]:>11s}" for lt in gen.LOCATION_TYPES))
    for t in gp.PAIR_TYPES:
        print("  " + f"{t:34s}" + "".join(f"{tab.get((t, lt), 0):11d}" for lt in gen.LOCATION_TYPES))
    so = Counter(r["needs_clarification"] for r in recs if r["location_type"] == "street_only")
    print(f"\n  street_only: actionable={so.get(False, 0)} clarification={so.get(True, 0)}")
    nulls = [r for r in recs if not r["has_exact_address"]]
    print(f"  no-address pairs: clarification={sum(r['needs_clarification'] for r in nulls)} "
          f"actionable={sum(not r['needs_clarification'] for r in nulls)}")
    for name, key, order in (("category", "category", gen.CATEGORIES), ("urgency", "urgency", gen.URGENCY_LEVELS),
                             ("style", "style", None)):
        c = Counter(r[key] for r in recs)
        print(f"\n  {name}: " + ", ".join(f"{k}={c.get(k, 0)}" for k in (order or [k for k, _ in c.most_common()])))
    print(f"\n  multi-issue: {sum(r['secondary_family_id'] is not None for r in recs)}")

    section("2. SCHEMA / CONTRACT / MINIMAL-DIFFERENCE")
    allowed = {"fabricated_address": {"address_or_null"},
               "fabricated_address_and_location": {"address_or_null", "location"},
               "over_null": {"address_or_null"}, "altered_address": {"address_or_null", "location"}}
    bad = 0
    for r in recs:
        errs = []
        try:
            c, j = json.loads(r["chosen"]), json.loads(r["rejected"])
        except json.JSONDecodeError as e:
            failures.append(f"{r['id']}: invalid JSON {e}")
            continue
        for lab, t in (("chosen", c), ("rejected", j)):
            if tuple(t) != gen.TICKET_KEYS:
                errs.append(f"{lab} key order {list(t)}")
            if t["category"] not in gen.CATEGORIES or t["urgency"] not in gen.URGENCY_LEVELS:
                errs.append(f"{lab} category/urgency out of enum")
            if not isinstance(t["location"], str) or not t["location"]:
                errs.append(f"{lab} location empty/non-string")
        if r["chosen"] == r["rejected"]:
            errs.append("chosen == rejected")
        diff = {k for k in gen.TICKET_KEYS if c[k] != j[k]}
        if diff != allowed[r["pair_type"]]:
            errs.append(f"differing fields {sorted(diff)} != allowed {sorted(allowed[r['pair_type']])}")
        if c["category"] != r["category"] or c["urgency"] != r["urgency"]:
            errs.append("metadata category/urgency != chosen")
        # --- address grounding ---
        ca, ra = c["address_or_null"], j["address_or_null"]
        if r["has_exact_address"]:
            if ca is None or norm(ca) not in norm(r["prompt"]):
                errs.append(f"chosen address {ca!r} not grounded in complaint")
            if r["pair_type"] == "over_null" and ra is not None:
                errs.append("over_null rejected must be null")
            if r["pair_type"] == "altered_address" and (ra is None or ra == ca or norm(ra) in norm(r["prompt"])):
                errs.append(f"altered address {ra!r} must differ from chosen and be absent from complaint")
            if r["pair_type"] == "altered_address" and len(ra) != len(ca):
                errs.append("altered address changed length")
        else:
            if ca is not None:
                errs.append(f"chosen address {ca!r} but complaint has no exact address")
            if ra is None or norm(ra) in norm(r["prompt"]):
                errs.append(f"rejected address {ra!r} must be non-null and unsupported by the complaint")
            if ra is not None and not re.match(r"^\d+ \S", ra):
                errs.append(f"fabricated address {ra!r} is not house-number + street")
            if (c["location"] == gen.LOCATION_UNSPECIFIED) != (r["location_type"] == "missing"):
                errs.append("chosen location 'unspecified' inconsistent with location_type")
        if (gen.CLARIFICATION_SENTENCE in c["summary"]) != r["needs_clarification"]:
            errs.append("chosen clarification sentence inconsistent with needs_clarification")
        if r["rejected_value"] != ra:
            errs.append("rejected_value metadata != rejected address_or_null")
        if errs:
            bad += 1
            failures += [f"{r['id']}: {e}" for e in errs]
    print(f"  pairs with schema/grounding/minimal-difference errors: {bad} / {N}")
    fab = [json.loads(r["rejected"])["address_or_null"] for r in recs if r["pair_type"] in gp.NO_ADDRESS_PAIR_TYPES]
    print(f"  fabricated addresses: {len(fab)} total, {len(set(fab))} distinct; "
          f"with suffix {sum(bool(re.search(r'[A-Za-z]{2,}$', a.split(' ', 1)[1]) and len(a.split()) > 2) for a in fab)}")

    section("3. DUPLICATES / LEAKAGE")
    prompts = Counter(r["prompt"] for r in recs)
    triples = Counter((r["prompt"], r["chosen"], r["rejected"]) for r in recs)
    dp = sum(v - 1 for v in prompts.values() if v > 1)
    dt = sum(v - 1 for v in triples.values() if v > 1)
    print(f"  duplicate prompts: {dp}   duplicate (prompt, chosen, rejected) triples: {dt}")
    if dp or dt:
        failures.append(f"duplicates: prompts={dp} triples={dt}")
    inst_train, inst_eval, split_info = gen.build_dataset(cfg["data"]["seed"], cfg["data"])
    eval_fams = {f"{c}/{i:02d}" for c, idx in split_info["eval_families"].items() for i in idx}
    used = {r["family_id"] for r in recs} | {r["secondary_family_id"] for r in recs if r["secondary_family_id"]}
    leak = sorted(used & eval_fams)
    print(f"  families used: {len(used)}; overlap with the 24 held-out eval families: {len(leak)}")
    if leak:
        failures.append(f"eval family leakage: {leak}")
    frozen = {"train": REPO_ROOT / cfg["data"]["instruction_train_path"], "eval": REPO_ROOT / cfg["data"]["instruction_eval_path"]}
    inst_complaints = {json.loads(l)["complaint"]: s for s, p in frozen.items() for l in p.read_text().splitlines()}
    overlap = Counter(inst_complaints[r["prompt"]] for r in recs if r["prompt"] in inst_complaints)
    print(f"  prompts identical to frozen instruction complaints: train={overlap.get('train', 0)} eval={overlap.get('eval', 0)}")
    if overlap:
        failures.append(f"prompt overlap with instruction data: {dict(overlap)}")
    ev_sh = [shingles(json.loads(l)["complaint"]) for l in frozen["eval"].read_text().splitlines()]
    sims = sorted((max(jaccard(shingles(r["prompt"]), e) for e in ev_sh), r["id"]) for r in recs)
    print(f"  preference prompt -> nearest eval complaint Jaccard: max={sims[-1][0]:.3f} ({sims[-1][1]}) "
          f"median={statistics.median(s for s, _ in sims):.3f}  (warn >= {NEAR_DUP_JACCARD_WARN})")
    if sims[-1][0] >= NEAR_DUP_JACCARD_WARN:
        warnings.append(f"preference prompt near-duplicates an eval complaint (Jaccard {sims[-1][0]:.3f})")
    h = {s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in frozen.items()}
    ok = all(h[s] == cfg["data"]["frozen_sha256"][s] for s in h)
    print(f"  frozen instruction data hashes unchanged: {ok}")
    if not ok:
        failures.append("frozen instruction dataset hash mismatch")

    section("4. LENGTH BALANCE (chosen vs rejected)")
    units = [("chars", len), ("words", lambda s: len(s.split()))]
    if tok is not None:
        units.append(("tokens", lambda s: len(tok(s, add_special_tokens=False)["input_ids"])))
    for uname, fn in units:
        cl = [fn(r["chosen"]) for r in recs]
        rl = [fn(r["rejected"]) for r in recs]
        print(f"\n  [{uname}] chosen   {stats(cl)}")
        print(f"  [{uname}] rejected {stats(rl)}")
        d = [b - a for a, b in zip(cl, rl)]
        print(f"  [{uname}] rejected - chosen: {stats(d)}")
        print(f"  [{uname}] rejected longer={sum(x > 0 for x in d)} shorter={sum(x < 0 for x in d)} equal={sum(x == 0 for x in d)}")
        if uname == "chars":
            ratios = [b / a for a, b in zip(cl, rl)]
            imb = [r for r, q in zip(recs, ratios) if abs(q - 1) > LENGTH_RATIO_TOL]
            print(f"  [chars] ratio rejected/chosen: {stats(ratios)}")
            print(f"  [chars] pairs with |ratio - 1| > {LENGTH_RATIO_TOL}: {len(imb)}/{N}")
            print("\n  by pair type (chars, rejected - chosen):")
            for t in gp.PAIR_TYPES:
                dt_ = [x for x, r in zip(d, recs) if r["pair_type"] == t]
                print(f"    {t:34s} n={len(dt_):3d} mean={statistics.mean(dt_):+7.2f} "
                      f"min={min(dt_):+4d} max={max(dt_):+4d}")
            if len(imb) / N > LENGTH_IMBALANCED_SHARE_WARN:
                warnings.append(f"{len(imb)}/{N} pairs have char-length ratio beyond +/-{LENGTH_RATIO_TOL}")
            for r in imb:
                print(f"    imbalanced: {r['id']} {r['pair_type']} {r['location_type']} "
                      f"ratio={len(r['rejected']) / len(r['chosen']):.3f}")

    section("5. SHORTCUT DIAGNOSTICS")
    run_shortcut_diagnostics(recs, tok, warnings, failures)

    section("6. DETERMINISTIC REGENERATION")
    if args.skip_regen:
        print("  skipped")
    else:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "p.jsonl"
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_preferences.py"), "--out", str(out)],
                           check=True, capture_output=True, env=dict(os.environ, PYTHONHASHSEED="12345"))
            a = hashlib.sha256(args.path.read_bytes()).hexdigest()
            b = hashlib.sha256(out.read_bytes()).hexdigest()
            print(f"  committed={a[:16]} regenerated={b[:16]} {'MATCH' if a == b else 'DIFFERENT'}")
            if a != b:
                failures.append("preference regeneration is not byte-identical")

    section("RESULT")
    print(f"  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"    - {w}")
    print(f"  HARD FAILURES ({len(failures)}):")
    for f in failures[:40]:
        print(f"    - {f}")
    print(f"\n  AUDIT {'FAILED' if failures else 'PASSED (review warnings above)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
