#!/usr/bin/env python3
"""QLoRA supervised instruction tuning for CivicDesk (Colab T4 entry point).

Modes
-----
  (default)       CUDA required. NF4 4-bit base + LoRA, plain HF Trainer, assistant-only loss.
                  Appends one row (success OR failure) to logs/experiment_log.csv.
  --dry-run       No model load. Config, frozen-data check, tokenizer, masking and
                  token-length validation only. Writes nothing to the experiment log.
  --inspect-cpu   Loads the UNQUANTIZED model on CPU (fp32), inspects linear modules,
                  attaches LoRA and reports parameter counts. This is NOT QLoRA and
                  trains nothing. Writes nothing to the experiment log.

API notes for the pinned stack (transformers 5.17.0 / peft 0.21.0), verified by
reading the installed source:
  * TrainingArguments has no `warmup_ratio`; `warmup_steps` accepts a float in
    [0, 1) as a ratio of total steps.
  * `eval_strategy` (not `evaluation_strategy`); Trainer takes `processing_class`.
  * `logging_nan_inf_filter` defaults to True and would hide NaN/inf losses in
    logs; it is set to False, and every step's loss is checked for finiteness.
  * `auto_find_batch_size` would silently retry after an OOM; it is forced False.
  * `from_pretrained(dtype=...)` (`torch_dtype` is deprecated).
  * peft's prepare_model_for_kbit_training freezes the base, upcasts all non-4-bit
    fp16/bf16 params (embeddings, norms, tied lm_head) to fp32, and enables
    gradient checkpointing; with use_reentrant=False it skips the input-grad hook.

Usage (Colab):
    python training/train_qlora.py --run-label baseline
    python training/train_qlora.py --run-label stress-mb16 --micro-batch-size 16 --grad-accum 1
    python training/train_qlora.py --run-label smoke --max-steps 5 --logging-steps 1 --eval-steps 5
"""
from __future__ import annotations

import argparse
import json

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def parse_args(cfg: dict) -> argparse.Namespace:
    s, l = cfg["sft"], cfg["lora"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate config/data/tokenizer/masking only")
    mode.add_argument("--inspect-cpu", action="store_true", help="CPU fp32 architecture + LoRA inspection only")
    ap.add_argument("--run-label", default="baseline", help="short label, e.g. baseline, stress-mb16")
    ap.add_argument("--max-seq-length", type=int, default=s["max_seq_length"])
    ap.add_argument("--micro-batch-size", type=int, default=s["micro_batch_size"])
    ap.add_argument("--grad-accum", type=int, default=s["gradient_accumulation_steps"])
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=s["gradient_checkpointing"])
    ap.add_argument("--learning-rate", type=float, default=s["learning_rate"])
    ap.add_argument("--epochs", type=float, default=s["num_train_epochs"])
    ap.add_argument("--warmup-ratio", type=float, default=s["warmup_ratio"])
    ap.add_argument("--eval-steps", type=int, default=s["eval_steps"])
    ap.add_argument("--logging-steps", type=int, default=s["logging_steps"],
                    help="logging cadence only (does not change optimization); smoke uses 1")
    ap.add_argument("--lora-r", type=int, default=l["r"])
    ap.add_argument("--lora-alpha", type=int, default=l["lora_alpha"])
    ap.add_argument("--lora-dropout", type=float, default=l["lora_dropout"])
    ap.add_argument("--output-dir", default=None,
                    help="default: sft.output_dir, or sft.smoke_output_dir for --run-label smoke")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="optional cap (e.g. a stress probe); -1 = full epochs. Logged in the run row.")
    ap.add_argument("--corrective-action", default="", help="what changed vs the failed run this one corrects")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    # A step-capped run must never be mistaken for (or saved as) the production SFT adapter.
    if args.run_label == "smoke" and args.max_steps <= 0:
        ap.error("--run-label smoke requires --max-steps > 0")
    if args.max_steps > 0 and args.run_label == "baseline":
        ap.error("--max-steps truncates training; use a non-baseline --run-label (e.g. smoke)")
    if args.output_dir is None:
        args.output_dir = s["smoke_output_dir"] if args.run_label == "smoke" else s["output_dir"]
    return args


def print_run_config(cfg: dict, a: argparse.Namespace, targets: list[str] | None) -> None:
    q = cfg["quantization"]
    eff = a.micro_batch_size * a.grad_accum
    print("=" * 78)
    print("RUN CONFIGURATION")
    print("=" * 78)
    rows = [
        ("model", cfg["model"]["model_id"]),
        ("mode", "dry-run" if a.dry_run else "inspect-cpu (fp32, NOT quantized)" if a.inspect_cpu else "QLoRA (CUDA)"),
        ("quantization", "none (CPU inspection)" if a.inspect_cpu else
         f"4-bit {q['bnb_4bit_quant_type']}, double_quant={q['bnb_4bit_use_double_quant']}, "
         f"compute_dtype={q['bnb_4bit_compute_dtype']}"),
        ("lora r / alpha / dropout", f"{a.lora_r} / {a.lora_alpha} / {a.lora_dropout}"),
        ("target modules", targets if targets else f"(candidates, unverified) {cfg['lora']['target_module_candidates']}"),
        ("max sequence length", a.max_seq_length),
        ("micro-batch size", a.micro_batch_size),
        ("gradient accumulation", a.grad_accum),
        ("effective batch size", eff),
        ("gradient checkpointing", a.gradient_checkpointing),
        ("learning rate", a.learning_rate),
        ("scheduler / warmup ratio", f"{cfg['sft']['lr_scheduler_type']} / {a.warmup_ratio}"),
        ("epochs / max_steps", f"{a.epochs} / {a.max_steps}"),
        ("max grad norm", cfg["sft"]["max_grad_norm"]),
        ("eval every N steps", a.eval_steps),
        ("fp16 / bf16", f"{cfg['sft']['fp16']} / {cfg['sft']['bf16']}"),
        ("seed", cfg["sft"]["seed"]),
    ]
    for k, v in rows:
        print(f"  {k:26s} {v}")


def load_quantized_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    print(f"\nquantization config: {bnb.to_dict()}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["model_id"], quantization_config=bnb, dtype=torch.float16,
        device_map={"": 0}, attn_implementation=cfg["model"]["attn_implementation"])
    print_quantization_evidence(model, q)
    return model


def print_quantization_evidence(model, q: dict) -> None:
    """Show that projections really are bitsandbytes 4-bit NF4 (+ double quant); fail if not."""
    from collections import Counter

    classes = Counter(type(m).__name__ for n, m in model.named_modules() if n.endswith("_proj"))
    first = next(m for n, m in model.named_modules() if n.endswith("self_attn.q_proj"))
    qs = getattr(first.weight, "quant_state", None)
    print("\nQUANTIZATION EVIDENCE")
    print(f"  projection module classes      {dict(classes)}")
    print(f"  q_proj weight class / dtype    {type(first.weight).__name__} / {first.weight.dtype}")
    print(f"  quant_state.quant_type         {getattr(qs, 'quant_type', None)}")
    print(f"  quant_state.nested (double q)  {getattr(qs, 'nested', None)}")
    print(f"  model.is_loaded_in_4bit        {getattr(model, 'is_loaded_in_4bit', None)}")
    print(f"  embed / lm_head dtype          {model.get_input_embeddings().weight.dtype} / "
          f"{model.get_output_embeddings().weight.dtype}")
    print(f"  memory footprint (HF)          {model.get_memory_footprint() / 2**30:.3f} GiB")
    if set(classes) != {"Linear4bit"}:
        raise RuntimeError(f"expected every *_proj to be Linear4bit, found {dict(classes)}")
    if getattr(qs, "quant_type", None) != q["bnb_4bit_quant_type"]:
        raise RuntimeError(f"quant_type {getattr(qs, 'quant_type', None)} != {q['bnb_4bit_quant_type']}")
    if q["bnb_4bit_use_double_quant"] and not getattr(qs, "nested", False):
        raise RuntimeError("double quantization requested but quant_state is not nested")


def attach_lora(model, cfg: dict, a: argparse.Namespace, quantized: bool):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("\nLINEAR / PROJECTION MODULES FOUND IN THE LOADED MODEL")
    inv = C.linear_module_inventory(model)
    C.print_linear_inventory(inv)
    targets = C.verify_target_modules(inv, cfg["lora"]["target_module_candidates"])
    print(f"verified LoRA targets: {targets}")

    base_params = sum(p.numel() for p in model.parameters())
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=a.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                      target_modules=targets, bias=cfg["lora"]["bias"], task_type=cfg["lora"]["task_type"])
    model = get_peft_model(model, lcfg)

    rep = C.trainable_parameter_report(model)
    C.assert_trainable_adapters(rep)
    n_wrapped = sum(1 for n, m in model.named_modules() if hasattr(m, "lora_A") and n.rsplit(".", 1)[-1] in targets)
    pred = C.lora_param_prediction(model.config, a.lora_r, targets)
    print("\nPARAMETERS")
    print(f"  base-model parameters      {base_params:,}"
          + ("  (as stored: 4-bit weights are packed, so this undercounts logical params)" if quantized else ""))
    print(f"  total after LoRA           {rep['total']:,}")
    print(f"  trainable (LoRA only)      {rep['trainable']:,}")
    print(f"  trainable %                {rep['trainable_pct']:.4f}% of total after LoRA")
    print(f"  LoRA-wrapped modules       {n_wrapped} ({len(targets)} targets x {pred['layers']} layers expected)")
    print(f"  arithmetic prediction      {pred['total']:,}  per layer {pred['per_layer']}")
    if rep["trainable"] != pred["total"]:
        print(f"  NOTE: measured trainable differs from arithmetic prediction by {rep['trainable'] - pred['total']:,}")
    return model, targets, rep, base_params


def build_training_args(cfg: dict, a: argparse.Namespace, out_dir, cpu_check: bool = False):
    """TrainingArguments for the QLoRA run. cpu_check=True only disables fp16 (GPU-only)
    so argument names/types can be validated on a machine without CUDA."""
    from transformers import TrainingArguments

    s = cfg["sft"]
    kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=a.micro_batch_size,
        per_device_eval_batch_size=a.micro_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.learning_rate,
        lr_scheduler_type=s["lr_scheduler_type"],
        warmup_steps=a.warmup_ratio,  # float in [0,1) = ratio of total steps (transformers 5.x)
        weight_decay=s["weight_decay"],
        max_grad_norm=s["max_grad_norm"],
        optim=s["optim"],
        fp16=s["fp16"] and not cpu_check,
        bf16=s["bf16"],
        gradient_checkpointing=a.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.gradient_checkpointing else None,
        eval_strategy="steps",
        eval_steps=a.eval_steps,
        logging_steps=a.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,     # never hide NaN/inf in logs
        save_strategy="no",               # adapter saved explicitly at the end
        auto_find_batch_size=False,       # never silently recover from OOM
        remove_unused_columns=False,
        report_to="none",
        seed=s["seed"],
        data_seed=s["seed"],
    )
    if cpu_check:
        kwargs["use_cpu"] = True
    return TrainingArguments(**kwargs)


def main() -> int:
    cfg = C.load_config()
    a = parse_args(cfg)
    C.set_seed(cfg["sft"]["seed"])
    print_run_config(cfg, a, None)

    # --- data + masking (always) ---
    hashes = C.verify_frozen_data(cfg)
    print(f"\nfrozen data verified: train={hashes['train'][:16]} eval={hashes['eval'][:16]}")
    train_recs = C.load_jsonl(cfg["data"]["instruction_train_path"])
    eval_recs = C.load_jsonl(cfg["data"]["instruction_eval_path"])
    tok = C.load_tokenizer(cfg["model"]["model_id"])
    train_ds = C.SFTDataset(tok, train_recs, a.max_seq_length)
    eval_ds = C.SFTDataset(tok, eval_recs, a.max_seq_length)
    for name, ds in (("train", train_ds), ("eval", eval_ds)):
        print(f"{name}: {ds.length_stats()}")
    print("\nMASKING CHECK (first training record)")
    print(C.masking_report(tok, train_recs[0], train_ds.items[0]))

    if a.dry_run:
        print("\nDRY RUN complete: no model loaded, nothing logged.")
        return 0

    if a.inspect_cpu:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(cfg["model"]["model_id"], dtype=torch.float32)
        model, targets, rep, _ = attach_lora(model, cfg, a, quantized=False)
        print_run_config(cfg, a, targets)
        print("\nINSPECT-CPU complete: fp32 CPU model, no quantization, no training, nothing logged.")
        return 0

    # --- CUDA training path ---
    if not C.cuda_available():
        raise RuntimeError("CUDA is not available. QLoRA training requires a CUDA GPU (Colab T4). "
                           "Use --dry-run or --inspect-cpu for local validation.")
    import torch
    from transformers import Trainer

    method = "qlora_sft_smoke" if a.run_label == "smoke" else "qlora_sft"
    run_id = C.make_run_id(method, a.run_label)
    out_dir = C.repo_path(a.output_dir) / run_id
    versions = C.package_versions()
    row = {
        "run_id": run_id, "timestamp_utc": C.utc_now(), "git_commit": C.git_commit(), "gpu_name": C.gpu_name(),
        "torch_version": versions["torch"], "transformers_version": versions["transformers"],
        "peft_version": versions["peft"], "trl_version": versions["trl"], "bitsandbytes_version": versions["bitsandbytes"],
        "model_id": cfg["model"]["model_id"], "method": method, "run_label": a.run_label,
        "quantization": f"4bit-{cfg['quantization']['bnb_4bit_quant_type']}-dq{int(cfg['quantization']['bnb_4bit_use_double_quant'])}"
                        f"-{cfg['quantization']['bnb_4bit_compute_dtype']}",
        "lora_rank": a.lora_r, "lora_alpha": a.lora_alpha,
        "target_modules": "|".join(cfg["lora"]["target_module_candidates"]),
        "max_seq_length": a.max_seq_length, "micro_batch_size": a.micro_batch_size, "grad_accum_steps": a.grad_accum,
        "effective_batch_size": a.micro_batch_size * a.grad_accum, "gradient_checkpointing": a.gradient_checkpointing,
        "learning_rate": a.learning_rate, "num_epochs": a.epochs, "dpo_beta": "",
        "corrective_action": a.corrective_action,
        "notes": "; ".join(x for x in [a.notes, f"max_steps={a.max_steps}" if a.max_steps > 0 else "",
                                       "peak_*_gb = max over load/train/eval phases since model load"] if x),
    }
    row.update(C.environment_provenance())

    step_timer = C.make_step_timer_callback()
    grad_check = C.make_lora_grad_check_callback()
    wall = C.WallClock()
    status, failure = "started", ""
    phases = C.MemoryPhases()  # resets peaks: global peak covers model load + training + evaluation
    try:
        model = load_quantized_model(cfg)
        model, targets, rep, _ = attach_lora(model, cfg, a, quantized=True)
        row.update({"target_modules": "|".join(targets), "trainable_params": rep["trainable"],
                    "trainable_pct": round(rep["trainable_pct"], 4)})
        if a.gradient_checkpointing:
            model.config.use_cache = False
        print_run_config(cfg, a, targets)
        phases.mark("load", next_phase="train")

        targs = build_training_args(cfg, a, out_dir)

        class CheckedTrainer(Trainer):
            def training_step(self, *args, **kwargs):
                loss = super().training_step(*args, **kwargs)
                C.check_finite_loss(float(loss), self.state.global_step)
                return loss

            def evaluate(self, *args, **kwargs):
                # attribute evaluation-time CUDA peaks to the "eval" phase (measurement only)
                prev = phases.active
                phases.mark(prev, next_phase="eval")
                try:
                    return super().evaluate(*args, **kwargs)
                finally:
                    phases.mark("eval", next_phase=prev)

        trainer = CheckedTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                                 data_collator=C.PadCollator(tok.pad_token_id), processing_class=tok,
                                 callbacks=[step_timer, grad_check])
        with wall:
            result = trainer.train()
        phases.mark("train")

        # Final evaluation of the final weights (periodic evals may not land on the last step).
        # Monitoring only: the adapter is always the last step; no checkpoint is selected on
        # eval loss, because the eval set is also the held-out behavioral test set.
        final_eval_wall = C.WallClock()
        with final_eval_wall:
            final_metrics = trainer.evaluate()
        C.check_finite_loss(float(final_metrics["eval_loss"]), trainer.state.global_step)
        status = "completed"

        hist = trainer.state.log_history
        train_losses = [h["loss"] for h in hist if "loss" in h]
        row.update({"total_steps": trainer.state.global_step,
                    "final_train_loss": round(train_losses[-1], 4) if train_losses else "NA",
                    "final_eval_loss": round(final_metrics["eval_loss"], 4)})
        row["notes"] += (f"; mean_train_loss={result.training_loss:.4f}; final_eval_after_step="
                         f"{trainer.state.global_step}; final_eval_s={final_eval_wall.seconds:.2f}")

        adapter_dir = out_dir / "final_adapter"
        model.save_pretrained(adapter_dir)   # adapter weights only, not a merged model
        tok.save_pretrained(adapter_dir)
        (out_dir / "log_history.json").write_text(json.dumps(hist, indent=1))
        weights = adapter_dir / "adapter_model.safetensors"
        if not weights.exists():
            raise RuntimeError(f"adapter weights not found at {weights}")
        adapter_sha = C.sha256_file(weights)
        rel_adapter = str(adapter_dir.relative_to(C.REPO_ROOT)) if adapter_dir.is_relative_to(C.REPO_ROOT) else str(adapter_dir)
        row.update({"adapter_path": rel_adapter, "adapter_sha256": adapter_sha})
        manifest = {"run_id": run_id, "method": method, "run_label": a.run_label, "git_commit": row["git_commit"],
                    "adapter_path": rel_adapter, "adapter_sha256": adapter_sha,
                    "base_model": cfg["model"]["model_id"], "target_modules": targets,
                    "frozen_data_sha256": hashes, "final_eval_loss": row["final_eval_loss"],
                    "total_steps": trainer.state.global_step}
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
        print("\n" + "=" * 78)
        print(("SMOKE ADAPTER (disposable, never use for DPO)" if a.run_label == "smoke" else
               "SFT ADAPTER FOR DPO") + f":\n  path   {rel_adapter}\n  sha256 {adapter_sha}"
              f"\n  manifest {out_dir / 'run_manifest.json'}")

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
        phases.mark(phases.active)          # fold in the peak of the phase that was running (incl. on failure)
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
        for k in ("peak_allocated_gb", "peak_reserved_gb", "peak_allocated_load_gb", "peak_reserved_load_gb",
                  "peak_allocated_train_gb", "peak_reserved_train_gb", "peak_allocated_eval_gb", "peak_reserved_eval_gb",
                  "alloc_retries", "cuda_ooms", "peak_inactive_split_gb", "peak_segments", "wall_clock_s",
                  "avg_step_time_s", "total_steps", "final_train_loss", "final_eval_loss", "adapter_path",
                  "adapter_sha256", "python_version", "torch_cuda_build", "failure_message"):
            print(f"  {k:20s} {row.get(k, '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
