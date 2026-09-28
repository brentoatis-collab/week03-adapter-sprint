#!/usr/bin/env python3
"""DPO on top of the verified baseline QLoRA SFT adapter (Colab T4 entry point).

Architecture (TRL 1.14.0 / peft 0.21.0, verified by reading the installed source):
  * One 4-bit NF4 base model (same load path as SFT), prepared with
    prepare_model_for_kbit_training exactly as in SFT.
  * POLICY    = the SFT adapter loaded trainable under the adapter name "default".
  * REFERENCE = TRL's built-in frozen copy of that adapter under the name "ref":
    DPOTrainer, given a PeftModel with a pretrained adapter and ref_model=None, calls
    add_adapter("ref") and copies the "default" weights into it. Reference log-probs are
    computed under torch.no_grad() with the "ref" adapter active. No second base model,
    no merge into the quantized base, never the adapter-disabled base.
  * TRL casts trainable adapter params of quantized models to bf16. With
    policy_adapter_dtype=fp32 (default) the policy is restored from the fp32 "ref" copy
    after trainer init, so the policy starts bit-identical to the SFT adapter and trains
    in fp32 under fp16 autocast, like SFT. Verified against the adapter file on disk.

Modes:
  (default)   CUDA required. Appends one row (success OR failure) to logs/experiment_log.csv.
  --dry-run   No model load. Verifies frozen data hashes, the SFT adapter SHA-256 (if the
              adapter is present), tokenization/prefix/length of every pair, and the step plan.

Reward accuracy computed on the 150 preference pairs is TRAINING-SET reward accuracy. There is
no held-out preference split; generalization is measured only by the held-out three-way
behavioral evaluation (base vs SFT vs DPO).

Usage (Colab):
    python training/train_dpo.py --sft-adapter-dir <restored>/final_adapter --dry-run
    python training/train_dpo.py --sft-adapter-dir <restored>/final_adapter --run-label baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ADAPTER_FILE = "adapter_model.safetensors"


# ---------------------------------------------------------------------------
# Verification helpers (CPU-safe)
# ---------------------------------------------------------------------------
def verify_preference_data(cfg: dict) -> str:
    path = C.repo_path(cfg["data"]["preference_train_path"])
    actual = C.sha256_file(path)
    expected = cfg["data"]["frozen_sha256"]["preference"]
    if actual != expected:
        raise RuntimeError(f"FROZEN PREFERENCE DATA MISMATCH: expected {expected[:16]}..., got {actual[:16]}...")
    return actual


def verify_sft_adapter(cfg: dict, adapter_dir: Path) -> dict:
    """Stop unless the adapter file hashes to the verified SFT value AND that value is the
    adapter_sha256 of the completed baseline SFT row named in the config."""
    d = cfg["dpo"]
    expected = d["expected_sft_adapter_sha256"]
    weights = adapter_dir / ADAPTER_FILE
    if "smoke_adapter" in str(adapter_dir):
        raise RuntimeError(f"refusing smoke adapter as DPO starting point: {adapter_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"SFT adapter weights not found: {weights}")
    actual = C.sha256_file(weights)
    if actual != expected:
        raise RuntimeError(f"SFT ADAPTER SHA-256 MISMATCH: expected {expected}, got {actual} ({weights})")
    log_path = C.repo_path(cfg["logging"]["experiment_log_path"])
    rows = [r for r in csv.DictReader(open(log_path)) if r["run_id"] == d["sft_run_id"]]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one log row for {d['sft_run_id']}, found {len(rows)}")
    r = rows[0]
    if (r["status"], r["method"], r["run_label"]) != ("completed", "qlora_sft", "baseline"):
        raise RuntimeError(f"SFT row {d['sft_run_id']} is not a completed baseline qlora_sft run: "
                           f"{r['status']}/{r['method']}/{r['run_label']}")
    if r["adapter_sha256"] != expected:
        raise RuntimeError(f"config expected_sft_adapter_sha256 != logged adapter_sha256 for {d['sft_run_id']}")
    cfg_json = json.loads((adapter_dir / "adapter_config.json").read_text())
    return {"sha256": actual, "sft_run_id": d["sft_run_id"], "sft_git_commit": r["git_commit"],
            "sft_trainable_params": int(r["trainable_params"]), "adapter_config": cfg_json}


def to_conversational(rec: dict) -> dict:
    """Frozen preference record -> TRL conversational format with the CivicDesk system prompt."""
    return {"prompt": C.build_messages(rec["prompt"]),
            "chosen": [{"role": "assistant", "content": rec["chosen"]}],
            "rejected": [{"role": "assistant", "content": rec["rejected"]}]}


def precheck_pairs(tok, records: list[dict], max_length: int) -> dict:
    """Replicates TRL 1.14's prompt/completion split and FAILS (TRL only warns) if the prompt is
    not an exact token prefix, the completion does not reproduce the response, or any sequence
    would be truncated by max_length (keep_start truncation would silently cut the completion end)."""
    totals, comp = [], []
    for r in records:
        conv = to_conversational(r)
        p_ids = list(tok.apply_chat_template(conv["prompt"], tokenize=True, add_generation_prompt=True)["input_ids"])
        for side in ("chosen", "rejected"):
            full = list(tok.apply_chat_template(conv["prompt"] + conv[side], tokenize=True)["input_ids"])
            if full[: len(p_ids)] != p_ids:
                raise C.MaskingError(f"{r['id']} {side}: prompt ids are not a prefix of prompt+{side}")
            c_ids = full[len(p_ids):]
            if tok.eos_token_id not in c_ids:
                raise C.MaskingError(f"{r['id']} {side}: no end-of-turn token in completion")
            text = tok.decode(c_ids[: c_ids.index(tok.eos_token_id)], skip_special_tokens=False)
            if text.strip() != r[side].strip():
                raise C.MaskingError(f"{r['id']} {side}: completion does not reproduce the response")
            if len(full) > max_length:
                raise C.MaskingError(f"{r['id']} {side}: {len(full)} tokens > max_length {max_length}; would truncate")
            totals.append(len(full))
            comp.append(len(c_ids))
    totals.sort()
    comp.sort()
    return {"pairs": len(records), "sequences": len(totals), "total_min": totals[0], "total_median": totals[len(totals) // 2],
            "total_max": totals[-1], "completion_min": comp[0], "completion_max": comp[-1]}


def planned_steps(n_pairs: int, micro: int, accum: int, epochs: float) -> dict:
    """Mirrors transformers 5.17 Trainer: steps/epoch = len_dl // accum + int(len_dl % accum > 0)."""
    len_dl = math.ceil(n_pairs / micro)
    per_epoch = max(len_dl // accum + int(len_dl % accum > 0), 1)
    total = per_epoch * math.ceil(epochs)
    return {"micro_batches_per_epoch": len_dl, "steps_per_epoch": per_epoch, "total_steps": total,
            "last_step_micro_batches": (len_dl % accum) or accum, "warmup_steps": math.ceil(0.1 * total)}


def lora_file_tensors(adapter_dir: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(adapter_dir / ADAPTER_FILE))


def setup_policy_and_reference(model, adapter_dir: Path, mode: str) -> dict:
    """After DPOTrainer init: verify the TRL 'ref' adapter is a frozen, exact copy of the SFT file,
    optionally restore the fp32 policy from it, and verify the trainable set. Raises on violation."""
    import torch

    if "ref" not in model.peft_config:
        raise RuntimeError("TRL did not create the 'ref' adapter; reference would be the adapter-disabled base")
    file_t = lora_file_tensors(adapter_dir)
    params = dict(model.named_parameters())
    policy = {n: p for n, p in params.items() if ".default." in n and "lora_" in n}
    ref = {n.replace(".default.", ".ref."): params[n.replace(".default.", ".ref.")] for n in policy}
    if len(policy) != len(file_t):
        raise RuntimeError(f"policy has {len(policy)} LoRA tensors, SFT file has {len(file_t)}")
    policy_dtypes_before = sorted({str(p.dtype) for p in policy.values()})

    max_ref_dev = 0.0
    for pn, pp in policy.items():
        key = pn.replace(".default.", ".")
        rp = ref[pn.replace(".default.", ".ref.")]
        if rp.requires_grad:
            raise RuntimeError(f"reference parameter is trainable: {pn}")
        ft = file_t[key].to(rp.device)
        max_ref_dev = max(max_ref_dev, (rp.detach().float() - ft.float()).abs().max().item())
        if mode == "fp32":
            pp.data = rp.detach().clone().to(torch.float32)
    if max_ref_dev != 0.0:
        raise RuntimeError(f"'ref' adapter differs from the SFT adapter file (max |diff| = {max_ref_dev:.3e})")

    max_pol_dev = max((pp.detach().float() - file_t[pn.replace(".default.", ".")].to(pp.device).float()).abs().max().item()
                      for pn, pp in policy.items())
    if mode == "fp32" and max_pol_dev != 0.0:
        raise RuntimeError(f"fp32 policy restore failed (max |policy - SFT file| = {max_pol_dev:.3e})")

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, _ in trainable if ".default." not in n or "lora_" not in n]
    if bad:
        raise RuntimeError(f"unexpected trainable parameters (only policy LoRA may train): {bad[:5]}")
    n_train = sum(p.numel() for _, p in trainable)
    return {"lora_tensors": len(policy), "trainable_params": n_train, "policy_dtype_after_trl_init": policy_dtypes_before,
            "policy_dtype_final": sorted({str(p.dtype) for p in policy.values()}), "ref_dtype": sorted({str(p.dtype) for p in ref.values()}),
            "max_abs_ref_minus_sft_file": max_ref_dev, "max_abs_policy_minus_sft_file": max_pol_dev}


def dpo_eval_metrics(m: dict, prefix: str) -> dict:
    """TRL 1.14 names its DPO metrics 'eval_rewards/*' regardless of metric_key_prefix; only the
    loss follows the prefix (verified on CPU). Normalize to plain keys and fail if absent."""
    out = {"loss": m[f"{prefix}_loss"]}
    for k in ("rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies", "logps/chosen", "logps/rejected"):
        key = f"eval_{k}"
        if key not in m:
            raise KeyError(f"expected TRL metric {key!r} in evaluate() output; got {sorted(m)}")
        out[k] = m[key]
    return out


def build_dpo_config(cfg: dict, a: argparse.Namespace, out_dir, cpu_check: bool = False):
    from trl import DPOConfig
    d = cfg["dpo"]
    kw = dict(
        output_dir=str(out_dir), beta=a.beta, loss_type=[d["loss_type"]],
        learning_rate=a.learning_rate, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.micro_batch_size, per_device_eval_batch_size=a.micro_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        lr_scheduler_type=d["lr_scheduler_type"], warmup_steps=d["warmup_ratio"],  # float in [0,1) = ratio
        max_grad_norm=d["max_grad_norm"], optim=d["optim"], weight_decay=d["weight_decay"],
        fp16=d["fp16"] and not cpu_check, bf16=d["bf16"],
        max_length=a.max_length, truncation_mode="keep_start",   # prechecked: nothing is truncated
        gradient_checkpointing=d["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        precompute_ref_log_probs=False, disable_dropout=True,
        logging_steps=a.logging_steps, logging_first_step=True, logging_nan_inf_filter=False,
        eval_strategy="no", save_strategy="no", auto_find_batch_size=False,
        report_to="none", seed=d["seed"], data_seed=d["seed"],
    )
    if cpu_check:
        kw["use_cpu"] = True
    return DPOConfig(**kw)


def parse_args(cfg: dict) -> argparse.Namespace:
    d = cfg["dpo"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-adapter-dir", required=True, type=Path,
                    help="restored copy of the logged baseline SFT final_adapter directory")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-missing-adapter", action="store_true",
                    help="LOCAL dry runs only: skip the SFT-adapter SHA check when the adapter is absent")
    ap.add_argument("--run-label", default="baseline")
    ap.add_argument("--beta", type=float, default=d["beta"])
    ap.add_argument("--learning-rate", type=float, default=d["learning_rate"])
    ap.add_argument("--epochs", type=float, default=d["num_train_epochs"])
    ap.add_argument("--micro-batch-size", type=int, default=d["micro_batch_size"])
    ap.add_argument("--grad-accum", type=int, default=d["gradient_accumulation_steps"])
    ap.add_argument("--max-length", type=int, default=d["max_length"])
    ap.add_argument("--logging-steps", type=int, default=d["logging_steps"])
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--policy-adapter-dtype", choices=["fp32", "trl_default"], default=d["policy_adapter_dtype"])
    ap.add_argument("--corrective-action", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if a.allow_missing_adapter and not a.dry_run:
        ap.error("--allow-missing-adapter is only valid with --dry-run")
    if a.max_steps > 0 and a.run_label == "baseline":
        ap.error("--max-steps truncates training; use a non-baseline --run-label")
    return a


def main() -> int:
    cfg = C.load_config()
    a = parse_args(cfg)
    C.set_seed(cfg["dpo"]["seed"])

    # --- integrity (always) ---
    inst_h = C.verify_frozen_data(cfg)
    pref_h = verify_preference_data(cfg)
    print(f"frozen data verified: instruction train={inst_h['train'][:16]} eval={inst_h['eval'][:16]} "
          f"preference={pref_h[:16]}")
    if a.sft_adapter_dir.exists():
        sft = verify_sft_adapter(cfg, a.sft_adapter_dir)
        print(f"SFT adapter verified: {a.sft_adapter_dir}\n  sha256 {sft['sha256']}  (== config == log row "
              f"{sft['sft_run_id']}, trained at commit {sft['sft_git_commit']})")
    elif a.dry_run and a.allow_missing_adapter:
        sft = None
        print(f"NOTE: SFT adapter not present at {a.sft_adapter_dir}; SHA-256 check NOT performed in this dry run.")
    else:
        raise FileNotFoundError(f"SFT adapter directory not found: {a.sft_adapter_dir}")

    records = [json.loads(l) for l in C.repo_path(cfg["data"]["preference_train_path"]).read_text().splitlines() if l.strip()]
    tok = C.load_tokenizer(cfg["model"]["model_id"])
    pre = precheck_pairs(tok, records, a.max_length)
    plan = planned_steps(len(records), a.micro_batch_size, a.grad_accum, a.epochs)
    d = cfg["dpo"]
    print("\nDPO CONFIGURATION")
    for k, v in [("policy", "SFT adapter, trainable, adapter name 'default'"),
                 ("reference", "TRL frozen copy of the SFT adapter, adapter name 'ref', same 4-bit base"),
                 ("policy adapter dtype", a.policy_adapter_dtype), ("loss", d["loss_type"]), ("beta", a.beta),
                 ("learning rate", a.learning_rate), ("scheduler / warmup ratio", f"{d['lr_scheduler_type']} / {d['warmup_ratio']}"),
                 ("epochs / max_steps", f"{a.epochs} / {a.max_steps}"), ("micro-batch (pairs)", a.micro_batch_size),
                 ("grad accumulation", a.grad_accum), ("effective batch (pairs)", a.micro_batch_size * a.grad_accum),
                 ("max length", a.max_length), ("gradient checkpointing", d["gradient_checkpointing"]),
                 ("fp16 / bf16", f"{d['fp16']} / {d['bf16']}"), ("max grad norm", d["max_grad_norm"]),
                 ("dropout", "disabled (TRL disable_dropout=True)")]:
        print(f"  {k:26s} {v}")
    print(f"\npair precheck: {pre}")
    print(f"step plan: {plan}")

    if a.dry_run:
        print("\nDRY RUN complete: no model loaded, nothing logged.")
        return 0

    if not C.cuda_available():
        raise RuntimeError("CUDA is not available. DPO requires a CUDA GPU (Colab T4). Use --dry-run locally.")
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from trl import DPOTrainer
    import train_qlora

    run_id = C.make_run_id("dpo", a.run_label)
    out_dir = C.repo_path(d["output_dir"]) / run_id
    versions = C.package_versions()
    row = {
        "run_id": run_id, "timestamp_utc": C.utc_now(), "git_commit": C.git_commit(), "gpu_name": C.gpu_name(),
        "torch_version": versions["torch"], "transformers_version": versions["transformers"],
        "peft_version": versions["peft"], "trl_version": versions["trl"], "bitsandbytes_version": versions["bitsandbytes"],
        "model_id": cfg["model"]["model_id"], "method": "dpo", "run_label": a.run_label,
        "quantization": f"4bit-{cfg['quantization']['bnb_4bit_quant_type']}-dq{int(cfg['quantization']['bnb_4bit_use_double_quant'])}"
                        f"-{cfg['quantization']['bnb_4bit_compute_dtype']}",
        "lora_rank": sft["adapter_config"]["r"], "lora_alpha": sft["adapter_config"]["lora_alpha"],
        "target_modules": "|".join(sorted(sft["adapter_config"]["target_modules"])),
        "max_seq_length": a.max_length, "micro_batch_size": a.micro_batch_size, "grad_accum_steps": a.grad_accum,
        "effective_batch_size": a.micro_batch_size * a.grad_accum, "gradient_checkpointing": d["gradient_checkpointing"],
        "learning_rate": a.learning_rate, "num_epochs": a.epochs, "dpo_beta": a.beta,
        "sft_adapter_sha256": sft["sha256"], "preference_data_sha256": pref_h,
        "policy_adapter_dtype": a.policy_adapter_dtype, "corrective_action": a.corrective_action,
        "notes": "; ".join(x for x in [a.notes, f"max_steps={a.max_steps}" if a.max_steps > 0 else "",
                                       f"sft_run_id={sft['sft_run_id']}", "micro_batch/effective_batch in PAIRS",
                                       "train_set_* = full pass over the 150 TRAINING pairs, not held-out",
                                       "peak_*_gb = max over load/train/eval phases since model load"] if x),
    }
    row.update(C.environment_provenance())

    step_timer = C.make_step_timer_callback()
    grad_check = C.make_lora_grad_check_callback()
    wall = C.WallClock()
    status, failure = "started", ""
    phases = C.MemoryPhases()
    try:
        base = train_qlora.load_quantized_model(cfg)
        inv = C.linear_module_inventory(base)
        C.verify_target_modules(inv, sorted(sft["adapter_config"]["target_modules"]))
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=d["gradient_checkpointing"],
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        model = PeftModel.from_pretrained(base, str(a.sft_adapter_dir), adapter_name="default", is_trainable=True)
        rep = C.trainable_parameter_report(model)
        C.assert_trainable_adapters(rep)
        if rep["trainable"] != sft["sft_trainable_params"]:
            raise RuntimeError(f"trainable {rep['trainable']} != SFT trainable {sft['sft_trainable_params']}")
        model.config.use_cache = False

        ds = Dataset.from_list([to_conversational(r) for r in records])
        targs = build_dpo_config(cfg, a, out_dir)

        class CheckedDPOTrainer(DPOTrainer):
            def training_step(self, *args, **kwargs):
                loss = super().training_step(*args, **kwargs)
                C.check_finite_loss(float(loss), self.state.global_step)
                return loss

            def evaluate(self, *args, **kwargs):
                prev = phases.active
                phases.mark(prev, next_phase="eval")
                try:
                    return super().evaluate(*args, **kwargs)
                finally:
                    phases.mark("eval", next_phase=prev)

        # eval_dataset = the SAME 150 training pairs: used only for full-pass training-set diagnostics.
        trainer = CheckedDPOTrainer(model=model, ref_model=None, args=targs, train_dataset=ds, eval_dataset=ds,
                                    processing_class=tok, callbacks=[step_timer, grad_check])
        if len(trainer.train_dataset) != len(records):
            raise RuntimeError(f"TRL dropped pairs: {len(trainer.train_dataset)} of {len(records)} remain")
        setup = setup_policy_and_reference(trainer.model, a.sft_adapter_dir, a.policy_adapter_dtype)
        print(f"\nPOLICY / REFERENCE SETUP: {json.dumps(setup)}")
        row.update({"trainable_params": setup["trainable_params"],
                    "trainable_pct": round(100 * setup["trainable_params"] / sum(p.numel() for p in trainer.model.parameters()), 4)})
        phases.mark("load", next_phase="train")

        # Step-0 reference check on the training pairs: policy == reference => rewards exactly 0.
        pre_m = dpo_eval_metrics(trainer.evaluate(metric_key_prefix="pre"), "pre")
        ref_check = max(abs(pre_m["rewards/chosen"]), abs(pre_m["rewards/rejected"]), abs(pre_m["rewards/margins"]))
        print(f"\nSTEP-0 REFERENCE CHECK (training pairs): {pre_m}")
        row["ref_check_max_abs_reward"] = f"{ref_check:.3e}"
        if a.policy_adapter_dtype == "fp32" and not ref_check < 1e-3:
            raise RuntimeError(f"step-0 rewards not ~0 (max |reward| {ref_check:.3e}): policy != reference")

        with wall:
            result = trainer.train()
        phases.mark("train")

        post_m = dpo_eval_metrics(trainer.evaluate(metric_key_prefix="train_set"), "train_set")
        print(f"\nPOST-TRAINING FULL PASS OVER THE 150 TRAINING PAIRS (training-set, not held-out): {post_m}")
        C.check_finite_loss(float(post_m["loss"]), trainer.state.global_step)
        status = "completed"

        hist = trainer.state.log_history
        steps = [h for h in hist if "loss" in h and "step" in h]
        row.update({"total_steps": trainer.state.global_step,
                    "final_train_loss": round(steps[-1]["loss"], 4) if steps else "NA",
                    "final_eval_loss": "",   # no held-out preference split
                    "train_set_reward_accuracy": round(post_m["rewards/accuracies"], 4),
                    "train_set_reward_margin": round(post_m["rewards/margins"], 4),
                    "train_set_chosen_reward": round(post_m["rewards/chosen"], 4),
                    "train_set_rejected_reward": round(post_m["rewards/rejected"], 4)})
        row["notes"] += (f"; mean_train_loss={result.training_loss:.4f}; train_set_loss={post_m['loss']:.4f}"
                         f"; step0_train_set_loss={pre_m['loss']:.4f}")

        adapter_dir = out_dir / "final_adapter"
        trainer.model.save_pretrained(adapter_dir, selected_adapters=["default"], save_embedding_layers=False)
        tok.save_pretrained(adapter_dir)
        (out_dir / "log_history.json").write_text(json.dumps(hist, indent=1))
        weights = adapter_dir / ADAPTER_FILE
        if not weights.exists():
            raise RuntimeError(f"DPO adapter weights not found at {weights}")
        saved = lora_file_tensors(adapter_dir)
        if len(saved) != setup["lora_tensors"] or any(".ref." in k or "lora_" not in k for k in saved):
            raise RuntimeError(f"saved adapter has unexpected tensors ({len(saved)}); expected policy LoRA only")
        sha = C.sha256_file(weights)
        rel = str(adapter_dir.relative_to(C.REPO_ROOT)) if adapter_dir.is_relative_to(C.REPO_ROOT) else str(adapter_dir)
        row.update({"adapter_path": rel, "adapter_sha256": sha})
        (out_dir / "run_manifest.json").write_text(json.dumps({
            "run_id": run_id, "method": "dpo", "git_commit": row["git_commit"], "adapter_path": rel, "adapter_sha256": sha,
            "starting_sft_adapter_sha256": sft["sha256"], "sft_run_id": sft["sft_run_id"],
            "preference_data_sha256": pref_h, "reference": "TRL 'ref' adapter = frozen copy of the SFT adapter",
            "policy_setup": setup, "step0_reference_check": pre_m, "train_set_metrics_post": post_m,
            "note": "train_set_* metrics are on the 150 TRAINING pairs; not held-out"}, indent=1))
        print("\n" + "=" * 78 + f"\nDPO ADAPTER FOR EVALUATION:\n  path   {rel}\n  sha256 {sha}\n  manifest {out_dir / 'run_manifest.json'}")

    except torch.cuda.OutOfMemoryError as e:
        status, failure = "oom", f"CUDA OOM: {str(e).splitlines()[0][:400]}"
        print("\n" + "!" * 78 + "\nCUDA OUT OF MEMORY — not retried, not hidden. Full error:\n")
        traceback.print_exc()
        raise
    except C.NonFiniteLossError as e:
        status, failure = "non_finite_loss", str(e)
        raise
    except C.GradientFlowError as e:
        status, failure = "gradient_flow_error", str(e)[:400]
        raise
    except Exception as e:
        status, failure = "error", f"{type(e).__name__}: {str(e).splitlines()[0][:400] if str(e) else ''}"
        raise
    finally:
        if wall.seconds is None and hasattr(wall, "start"):
            import time
            wall.seconds = time.perf_counter() - wall.start
        phases.mark(phases.active)
        st = step_timer.summary()
        row.update(phases.row_fields())
        row.update(C.allocator_stats())
        row.update({"status": status, "failure_message": failure,
                    "wall_clock_s": round(wall.seconds, 2) if wall.seconds is not None else "NA",
                    "avg_step_time_s": st["avg_step_time_s"]})
        row.setdefault("total_steps", len(step_timer.durations))
        row["notes"] += f"; steps_timed={st['steps_timed']}; first_step_s={st['first_step_time_s']}; avg excludes first step"
        gc = grad_check.result
        row["notes"] += (f"; grad_check=passed@step{gc['step']} lora_nonzero={gc['lora_nonzero_grad']}/{gc['lora_tensors']}"
                         f" amp_overflow_steps={gc['amp_overflow_steps_before']}" if gc else
                         f"; grad_check=not_passed amp_overflow_steps={grad_check.overflow_steps}")
        path = C.append_experiment_log(row)
        print("\n" + "=" * 78 + f"\nRUN RESULT ({status}) appended to {path}")
        for k in ("total_steps", "final_train_loss", "train_set_reward_accuracy", "train_set_reward_margin",
                  "train_set_chosen_reward", "train_set_rejected_reward", "ref_check_max_abs_reward",
                  "peak_allocated_gb", "peak_reserved_gb", "peak_allocated_train_gb", "peak_reserved_train_gb",
                  "alloc_retries", "cuda_ooms", "wall_clock_s", "avg_step_time_s", "adapter_path", "adapter_sha256",
                  "failure_message"):
            print(f"  {k:26s} {row.get(k, '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
