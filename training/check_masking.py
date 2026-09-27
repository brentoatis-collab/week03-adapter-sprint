#!/usr/bin/env python3
"""Masking diagnostic on REAL frozen training records (CPU only, tokenizer only).

Prints, for selected records, the prompt boundary, the decoded trainable
assistant region and the masked/trainable token counts. Then encodes every
train and eval record with the same fail-fast checks and reports token-length
statistics against the configured max_seq_length.

Usage:
    python training/check_masking.py                     # default: 3 diverse records
    python training/check_masking.py --ids cd-train-0090 cd-train-0001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def pick_default(records: list[dict]) -> list[dict]:
    """One exact-address, one no-address (clarification), one no-address actionable."""
    picks = [
        next(r for r in records if r["has_exact_address"] and r["style"] == "standard"),
        next(r for r in records if not r["has_exact_address"] and r["needs_clarification"]),
        next(r for r in records if not r["has_exact_address"] and not r["needs_clarification"]),
    ]
    return picks


def main() -> int:
    cfg = C.load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()

    hashes = C.verify_frozen_data(cfg)
    print(f"frozen data verified: train={hashes['train'][:16]}  eval={hashes['eval'][:16]}")

    train = C.load_jsonl(cfg["data"]["instruction_train_path"])
    evl = C.load_jsonl(cfg["data"]["instruction_eval_path"])
    tok = C.load_tokenizer(cfg["model"]["model_id"])
    max_len = cfg["sft"]["max_seq_length"]

    by_id = {r["id"]: r for r in train}
    chosen = [by_id[i] for i in args.ids] if args.ids else pick_default(train)
    for r in chosen:
        enc = C.encode_sft_example(tok, r["complaint"], r["response"], max_len)
        print("\n" + "-" * 78)
        print(C.masking_report(tok, r, enc))

    print("\n" + "=" * 78)
    for name, recs in (("train", train), ("eval", evl)):
        ds = C.SFTDataset(tok, recs, max_len)  # raises MaskingError on any failure
        s = ds.length_stats()
        print(f"{name:5s} n={s['n']:3d}  total tokens min/median/p95/max = {s['total_min']}/{s['total_median']}/"
              f"{s['total_p95']}/{s['total_max']}  (max_seq_length={max_len})  "
              f"trainable min/median/max = {s['trainable_min']}/{s['trainable_median']}/{s['trainable_max']}  "
              f"loss-bearing share = {100 * s['loss_share_mean']:.1f}%")
    sys_ids = C.render_prompt_ids(tok, "")
    print(f"system prompt + empty user turn + assistant header = {len(sys_ids)} tokens (masked in every example)")
    print("all records passed masking checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
