#!/usr/bin/env python3
"""Three-way held-out behavioral evaluation: BASE vs SFT vs DPO (Colab T4 entry point).

One NF4 4-bit base model is loaded once (same load + preparation path as training).
  BASE = that model with all adapters disabled (PeftModel.disable_adapter()).
  SFT  = verified SFT adapter, loaded as adapter "sft".
  DPO  = verified DPO adapter, loaded as adapter "dpo".
Adapters are switched in place with set_adapter(); nothing is merged, trained or saved.
Both adapter files must hash to the configured SHA-256 (which must also equal the logged
training rows) and the in-memory LoRA tensors must equal the files bit-for-bit.

Every state sees the same 60 frozen eval complaints, in file order, with the same system
prompt, chat template, greedy decoding config (batch size 1) and the frozen scoring rules in
scripts/eval_scoring.py (its SHA-256 is recorded in the manifest).

Modes:
  (default)      CUDA required. Writes results/three_way_eval/<eval_id>/{generations.jsonl,
                 scored.jsonl, summary.json, summary.md, manifest.json}.
  --dry-run      No model load: data/adapter integrity, prompts, denominators, generation
                 config, and scoring self-tests (gold must score perfectly).
  --score-only D Re-score an existing results directory's generations.jsonl with the frozen rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "training"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import common as C  # noqa: E402
import eval_scoring as S  # noqa: E402
import generate_instruction_data as gen  # noqa: E402

ADAPTER_FILE = "adapter_model.safetensors"


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------
def verify_adapter(cfg: dict, which: str, adapter_dir: Path) -> dict:
    """which in {'sft','dpo'}: file SHA == config == logged completed training row."""
    e = cfg["evaluation"]
    expected, run_id = e[f"expected_{which}_adapter_sha256"], e[f"{which}_run_id"]
    if "smoke_adapter" in str(adapter_dir):
        raise RuntimeError(f"refusing smoke adapter for {which}: {adapter_dir}")
    weights = adapter_dir / ADAPTER_FILE
    if not weights.exists():
        raise FileNotFoundError(f"{which} adapter weights not found: {weights}")
    actual = C.sha256_file(weights)
    if actual != expected:
        raise RuntimeError(f"{which.upper()} ADAPTER SHA-256 MISMATCH: expected {expected}, got {actual}")
    rows = [r for r in csv.DictReader(open(C.repo_path(cfg["logging"]["experiment_log_path"]))) if r["run_id"] == run_id]
    want_method = {"sft": "qlora_sft", "dpo": "dpo"}[which]
    if len(rows) != 1 or rows[0]["status"] != "completed" or rows[0]["method"] != want_method:
        raise RuntimeError(f"no single completed {want_method} log row for {run_id}")
    if rows[0]["adapter_sha256"] != expected:
        raise RuntimeError(f"config {which} SHA != logged adapter_sha256 of {run_id}")
    if which == "dpo" and rows[0]["sft_adapter_sha256"] != e["expected_sft_adapter_sha256"]:
        raise RuntimeError("DPO row was not trained from the configured SFT adapter")
    return {"run_id": run_id, "path": str(adapter_dir), "sha256": actual, "trained_at_commit": rows[0]["git_commit"]}


def eval_records(cfg: dict) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    C.verify_frozen_data(cfg)
    recs = C.load_jsonl(cfg["data"]["instruction_eval_path"])
    train = C.load_jsonl(cfg["data"]["instruction_train_path"])
    # multi-issue: category of the secondary issue (for the flag only)
    sec = {r["id"]: r["secondary_family_id"].split("/")[0] for r in recs if r.get("secondary_family_id")}
    maj = {}
    for cat in gen.CATEGORIES:
        c = Counter(r["urgency"] for r in train if r["category"] == cat)
        maj[cat] = sorted(c.items(), key=lambda kv: (-kv[1], gen.URGENCY_LEVELS.index(kv[0])))[0][0]
    return recs, sec, maj


def check_population(recs: list[dict]) -> dict:
    exact = sum(r["has_exact_address"] for r in recs)
    cond = Counter(r["location_type"] for r in recs if not r["has_exact_address"])
    so = Counter(r["needs_clarification"] for r in recs if r["location_type"] == "street_only")
    pop = {"total": len(recs), "exact_address": exact, "no_address": len(recs) - exact, "by_condition": dict(cond),
           "street_only": {"actionable": so.get(False, 0), "clarify": so.get(True, 0)}}
    if (len(recs), exact) != (60, 36) or set(cond.values()) != {4} or len(cond) != 6 or pop["street_only"] != {"actionable": 2, "clarify": 2}:
        raise RuntimeError(f"eval population differs from the frozen design: {pop}")
    return pop


def scoring_self_test(recs: list[dict], sec: dict) -> dict:
    """Gold responses must score perfectly; always-null and always-fabricate must be caught."""
    gold = [S.score_example(r, r["response"], sec.get(r["id"])) for r in recs]
    perfect = all(s["core_correct"] and s["full_exact_match"] and s["location_exact"] and s["location_anchor"]
                  and not s["flags"] for s in gold)
    nulls = []
    fabs = []
    for r in recs:
        t = json.loads(r["response"])
        t_null = dict(t, address_or_null=None)
        t_fab = dict(t, address_or_null=t["address_or_null"] or "123 Main St")
        nulls.append(S.score_example(r, json.dumps(t_null), sec.get(r["id"])))
        fabs.append(S.score_example(r, json.dumps(t_fab), sec.get(r["id"])))
    n_ex = sum(r["has_exact_address"] for r in recs)
    res = {"gold_perfect": perfect,
           "always_null_over_null": sum(s["address_outcome"] == "over_null" for s in nulls),
           "always_null_fabrication": sum(s["address_outcome"] == "fabricated_address" for s in nulls),
           "always_fabricate_fabrication": sum(s["address_outcome"] == "fabricated_address" for s in fabs)}
    ok = (perfect and res["always_null_over_null"] == n_ex and res["always_null_fabrication"] == 0
          and res["always_fabricate_fabrication"] == len(recs) - n_ex)
    if not ok:
        raise RuntimeError(f"scoring self-test failed: {res}")
    return res


def greedy_config(cfg: dict):
    from transformers import GenerationConfig
    e = cfg["evaluation"]
    return GenerationConfig(do_sample=e["do_sample"], num_beams=e["num_beams"], max_new_tokens=e["max_new_tokens"],
                            repetition_penalty=e["repetition_penalty"], eos_token_id=e["eos_token_ids"],
                            pad_token_id=e["pad_token_id"], bos_token_id=e["pad_token_id"])


# ---------------------------------------------------------------------------
# Model + adapters
# ---------------------------------------------------------------------------
def load_three_state_model(cfg: dict, sft_dir: Path, dpo_dir: Path, quantized: bool = True):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    if quantized:
        import train_qlora
        base = train_qlora.load_quantized_model(cfg)           # prints + checks NF4 / double-quant evidence
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)  # same numerics as training
    else:                                                        # CPU validation only
        base = AutoModelForCausalLM.from_pretrained(cfg["model"]["model_id"], dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(sft_dir), adapter_name="sft", is_trainable=False)
    model.load_adapter(str(dpo_dir), adapter_name="dpo", is_trainable=False)
    model.eval()
    model.generation_config = greedy_config(cfg)
    return model


def verify_loaded_adapters(model, sft_dir: Path, dpo_dir: Path) -> dict:
    from safetensors.torch import load_file
    params = dict(model.named_parameters())
    out = {}
    for name, d in (("sft", sft_dir), ("dpo", dpo_dir)):
        f = load_file(str(d / ADAPTER_FILE))
        mine = {n: p for n, p in params.items() if f".{name}." in n and "lora_" in n}
        if len(mine) != len(f):
            raise RuntimeError(f"{name}: {len(mine)} loaded LoRA tensors vs {len(f)} in file")
        dev = max((p.detach().float().cpu() - f[n.replace(f".{name}.", ".")].float()).abs().max().item() for n, p in mine.items())
        if dev != 0.0:
            raise RuntimeError(f"{name}: loaded adapter differs from file (max |diff| {dev:.3e})")
        out[name] = {"lora_tensors": len(mine), "max_abs_diff_vs_file": dev}
    if any(p.requires_grad for p in model.parameters()):
        # set_adapter can flip requires_grad on; harmless under no_grad, but recorded
        out["note"] = "some parameters report requires_grad=True (inference runs under torch.no_grad)"
    return out


class State:
    """Context manager selecting BASE (adapters disabled) or a named adapter."""

    def __init__(self, model, name: str):
        self.model, self.name, self._ctx = model, name, None

    def __enter__(self):
        if self.name == "base":
            self._ctx = self.model.disable_adapter()
            self._ctx.__enter__()
        else:
            self.model.set_adapter(self.name)
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
        return False


def probe_logits(model, tok, complaint: str, device) -> dict:
    """Last-position logits under each state; switching must be clean and states must differ."""
    import torch
    ids = torch.tensor([C.render_prompt_ids(tok, complaint)], device=device)
    got = {}
    for st in ("base", "sft", "dpo", "sft", "base"):
        with State(model, st), torch.no_grad():
            got.setdefault(st, []).append(model(input_ids=ids).logits[0, -1].float().cpu())
    d = lambda a, b: (a - b).abs().max().item()  # noqa: E731
    res = {"sft_vs_base": d(got["sft"][0], got["base"][0]), "dpo_vs_sft": d(got["dpo"][0], got["sft"][0]),
           "sft_repeat": d(got["sft"][0], got["sft"][1]), "base_repeat": d(got["base"][0], got["base"][1])}
    if res["sft_vs_base"] == 0 or res["dpo_vs_sft"] == 0:
        raise RuntimeError(f"adapter switching had no effect: {res}")
    if res["sft_repeat"] != 0 or res["base_repeat"] != 0:
        raise RuntimeError(f"adapter switching is not clean (state not reproducible): {res}")
    return res


def generate_one(model, tok, complaint: str, gcfg, device, autocast: bool) -> dict:
    import contextlib
    import torch
    ids = torch.tensor([C.render_prompt_ids(tok, complaint)], device=device)
    ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else contextlib.nullcontext()
    t0 = time.perf_counter()
    with torch.no_grad(), ctx:
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gcfg)
    new = out[0, ids.shape[1]:].tolist()
    eos = set(gcfg.eos_token_id)
    cut = next((i for i, t in enumerate(new) if t in eos), None)
    body = new if cut is None else new[:cut]
    return {"raw_text": tok.decode(body, skip_special_tokens=False), "n_new_tokens": len(new),
            "hit_eos": cut is not None, "stop_token": new[cut] if cut is not None else None,
            "seconds": round(time.perf_counter() - t0, 3), "new_token_ids": new}


# ---------------------------------------------------------------------------
def score_all(recs, sec, maj, gens: list[dict]) -> tuple[list[dict], dict]:
    by = {(g["state"], g["id"]): g for g in gens}
    scored_rows, scored = [], {}
    for st in S.STATES:
        if not any(g["state"] == st for g in gens):
            continue
        sc = []
        for r in recs:
            g = by[(st, r["id"])]
            s = S.score_example(r, g["raw_text"], sec.get(r["id"]))
            s.update({"state": st, "location_type": r["location_type"], "has_exact_address": r["has_exact_address"],
                      "needs_clarification": r["needs_clarification"], "hit_eos": g["hit_eos"]})
            if not g["hit_eos"]:
                s["flags"].append("no_eos_max_new_tokens_reached")
            sc.append(s)
            scored_rows.append(s)
        scored[st] = sc
    return scored_rows, S.aggregate(recs, scored, maj)


def main() -> int:
    cfg = C.load_config()
    e = cfg["evaluation"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-adapter-dir", type=Path)
    ap.add_argument("--dpo-adapter-dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-missing-adapters", action="store_true", help="LOCAL dry runs only")
    ap.add_argument("--score-only", type=Path, help="results dir containing generations.jsonl")
    a = ap.parse_args()
    if a.allow_missing_adapters and not a.dry_run:
        ap.error("--allow-missing-adapters is only valid with --dry-run")

    recs, sec, maj = eval_records(cfg)
    pop = check_population(recs)
    selftest = scoring_self_test(recs, sec)
    scoring_sha = C.sha256_file(REPO_ROOT / "scripts" / "eval_scoring.py")
    if scoring_sha != e["frozen_scoring_rules_sha256"]:
        raise RuntimeError(f"SCORING RULES CHANGED: eval_scoring.py sha256 {scoring_sha} != frozen "
                           f"{e['frozen_scoring_rules_sha256']} in configs/adapter_config.json")
    print(f"eval population: {pop}")
    print(f"scoring rules {S.SCORING_RULES_VERSION} sha256={scoring_sha[:16]}  self-test: {selftest}")

    if a.score_only:
        gens = [json.loads(l) for l in (a.score_only / "generations.jsonl").read_text().splitlines()]
        rows, agg = score_all(recs, sec, maj, gens)
        (a.score_only / "rescored_summary.md").write_text(S.summary_markdown(agg) + "\n")
        print(S.summary_markdown(agg))
        return 0

    adapters = {}
    for which, d in (("sft", a.sft_adapter_dir), ("dpo", a.dpo_adapter_dir)):
        if d is not None and d.exists():
            adapters[which] = verify_adapter(cfg, which, d)
            print(f"{which.upper()} adapter verified: {adapters[which]}")
        elif a.dry_run and a.allow_missing_adapters:
            print(f"NOTE: {which} adapter not present ({d}); SHA-256 check NOT performed in this local dry run.")
        else:
            raise FileNotFoundError(f"{which} adapter directory missing: {d}")

    tok = C.load_tokenizer(cfg["model"]["model_id"])
    gcfg = greedy_config(cfg)
    lens = sorted(len(C.render_prompt_ids(tok, r["complaint"])) for r in recs)
    print(f"prompt tokens min/median/max = {lens[0]}/{lens[len(lens) // 2]}/{lens[-1]}")
    print(f"generation config: {gcfg.to_diff_dict()}  batch_size={e['batch_size']} autocast_fp16={e['autocast_fp16']}")
    if a.dry_run:
        print("\nDRY RUN complete: no model loaded, no generations, nothing written.")
        return 0

    if not C.cuda_available():
        raise RuntimeError("CUDA is not available. The formal evaluation runs on the Colab T4. Use --dry-run locally.")
    import torch
    C.set_seed(e["seed"])
    eval_id = f"three_way-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    out_dir = C.repo_path(e["results_dir"]) / eval_id
    out_dir.mkdir(parents=True, exist_ok=False)
    phases = C.MemoryPhases()
    t_all = time.perf_counter()
    model = load_three_state_model(cfg, a.sft_adapter_dir, a.dpo_adapter_dir)
    loaded = verify_loaded_adapters(model, a.sft_adapter_dir, a.dpo_adapter_dir)
    device = next(model.parameters()).device
    probe = probe_logits(model, tok, recs[0]["complaint"], device)
    eff = model._prepare_generation_config(gcfg)[0]
    effective = {k: getattr(eff, k) for k in ("do_sample", "num_beams", "max_new_tokens", "repetition_penalty",
                                              "eos_token_id", "pad_token_id", "temperature", "top_k", "top_p")}
    print(f"adapters in memory == files: {loaded}\nswitching probe: {probe}\neffective generation config: {effective}")
    phases.mark("load", next_phase="generate")

    gens, per_state = [], {}
    gpath = out_dir / "generations.jsonl"
    with open(gpath, "w") as gf:
        for st in e["state_order"]:
            t0 = time.perf_counter()
            with State(model, st):
                for r in recs:
                    g = generate_one(model, tok, r["complaint"], gcfg, device, e["autocast_fp16"])
                    g.update({"state": st, "id": r["id"]})
                    gens.append(g)
                    gf.write(json.dumps({k: g[k] for k in ("state", "id", "raw_text", "n_new_tokens", "hit_eos",
                                                           "stop_token", "seconds", "new_token_ids")}) + "\n")
                    gf.flush()
            per_state[st] = {"seconds": round(time.perf_counter() - t0, 2),
                             "new_tokens": sum(g["n_new_tokens"] for g in gens if g["state"] == st)}
            print(f"{st.upper()}: {len(recs)} generations in {per_state[st]['seconds']} s")
    n_primary = len(e["state_order"]) * len(recs)
    if len(gens) != n_primary:
        raise RuntimeError(f"expected {n_primary} primary generations, got {len(gens)}")
    # determinism probe: regenerate the first k examples per state and compare token ids.
    # Probe generations are NOT added to `gens` and never enter any behavioral denominator.
    k = e["determinism_probe_examples"]
    det = {}
    for st in e["state_order"]:
        with State(model, st):
            same = [generate_one(model, tok, r["complaint"], gcfg, device, e["autocast_fp16"])["new_token_ids"]
                    == next(g for g in gens if g["state"] == st and g["id"] == r["id"])["new_token_ids"] for r in recs[:k]]
        det[st] = f"{sum(same)}/{k} identical"
    phases.mark("generate")

    rows, agg = score_all(recs, sec, maj, gens)
    with open(out_dir / "scored.jsonl", "w") as f:
        for s in rows:
            f.write(json.dumps(s) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=1))
    md = S.summary_markdown(agg)
    (out_dir / "summary.md").write_text(md + "\n")
    manifest = {
        "eval_id": eval_id, "timestamp_utc": C.utc_now(), "git_commit": C.git_commit(), "gpu_name": C.gpu_name(),
        "packages": C.package_versions(), "environment": C.environment_provenance(),
        "model_id": cfg["model"]["model_id"], "quantization": cfg["quantization"], "adapters": adapters,
        "adapters_loaded_check": loaded, "switching_probe": probe, "states": list(e["state_order"]),
        "eval_data_sha256": C.sha256_file(C.repo_path(cfg["data"]["instruction_eval_path"])), "population": pop,
        "generation_config_passed": gcfg.to_diff_dict(), "generation_config_effective": effective,
        "batch_size": e["batch_size"], "autocast_fp16": e["autocast_fp16"], "system_prompt_sha256":
            __import__("hashlib").sha256(C.SYSTEM_PROMPT.encode()).hexdigest(),
        "scoring_rules_version": S.SCORING_RULES_VERSION, "scoring_rules_sha256": scoring_sha,
        "evaluate_models_sha256": C.sha256_file(Path(__file__)), "scoring_self_test": selftest,
        "determinism_probe": det, "determinism_probe_generations": len(e["state_order"]) * k,
        "primary_generations": len(gens), "per_state": per_state,
        "comparison_type": "controlled three-way model-state comparison (one NF4 base, identical fp32 upcast, fp16 "
                           "autocast, prompts, template, greedy decoding, order); not a stock-inference benchmark", "wall_clock_s": round(time.perf_counter() - t_all, 2),
        **{f"memory_{k}": v for k, v in phases.row_fields().items()}, "allocator": C.allocator_stats(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print("\n" + md)
    print(f"\ndeterminism probe: {det}\nresults written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
