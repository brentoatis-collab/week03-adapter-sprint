#!/usr/bin/env python3
"""CPU-only validation of the DPO path (Step 5B). NO optimizer step, no quantization, no
experiment-log write (verified by hash).

The real SFT adapter lives on Google Drive, so a synthetic stand-in adapter (same LoRA config,
nonzero B) is saved to a temp dir. TRL's quantized-only bf16 cast is exercised by flagging the
fp32 CPU base as 4-bit, which reproduces the exact code path that runs on the T4.

Checks:
  1. frozen instruction + preference hashes; SFT-adapter SHA/log verification (mismatch stops)
  2. every real pair passes the prefix / completion / no-truncation precheck; step plan
  3. DPOConfig builds with the pinned API and the intended overrides
  4. DPOTrainer init with a PeftModel + ref_model=None creates a frozen 'ref' copy; no pairs dropped
  5. fp32 policy restore: policy == ref == adapter file bit-for-bit; only policy LoRA trainable
  6. trl_default mode leaves the policy in bf16 (documents what fp32 mode undoes)
  7. step-0 evaluation: rewards exactly 0 when policy == reference; metric keys as parsed
  8. one forward/backward via trainer.training_step: finite loss, grads on policy LoRA only
  9. saving selected_adapters=['default'] writes policy LoRA tensors only (no 'ref')
Usage:  python training/validate_dpo_local.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import train_dpo as T  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def expect_raises(name, exc, fn):
    try:
        fn()
    except exc as e:
        check(name, True, f"{exc.__name__}: {str(e)[:100]}")
        return
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
        return
    check(name, False, "no exception")


def build_trainer(cfg, records, adapter_dir, mode, n_pairs=4):
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from trl import DPOTrainer
    import argparse

    tok = C.load_tokenizer(cfg["model"]["model_id"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["model_id"], dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name="default", is_trainable=True)
    model.base_model.model.is_loaded_in_4bit = True        # take TRL's quantized-model branch
    a = argparse.Namespace(beta=0.1, learning_rate=5e-5, epochs=1, max_steps=-1, micro_batch_size=2, grad_accum=1,
                           max_length=512, logging_steps=1)
    args = T.build_dpo_config(cfg, a, tempfile.mkdtemp(), cpu_check=True)
    ds = Dataset.from_list([T.to_conversational(r) for r in records[:n_pairs]])
    tr = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, eval_dataset=ds, processing_class=tok)
    setup = T.setup_policy_and_reference(tr.model, adapter_dir, mode)
    return tr, setup, tok


def main() -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    cfg = C.load_config()
    log_path = C.repo_path(cfg["logging"]["experiment_log_path"])
    log_hash = C.sha256_file(log_path)

    # 1
    C.verify_frozen_data(cfg)
    ph = T.verify_preference_data(cfg)
    check("frozen instruction + preference hashes", True, f"preference={ph[:16]}")
    tmp = Path(tempfile.mkdtemp())
    stand_in = tmp / "final_adapter"
    lc = cfg["lora"]
    torch.manual_seed(0)
    m = get_peft_model(AutoModelForCausalLM.from_pretrained(cfg["model"]["model_id"], dtype=torch.float32),
                       LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
                                  target_modules=lc["target_module_candidates"], bias="none", task_type="CAUSAL_LM"))
    for n, p in m.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 1e-3)
    m.save_pretrained(stand_in)
    del m
    expect_raises("SFT adapter SHA mismatch stops the run", RuntimeError, lambda: T.verify_sft_adapter(cfg, stand_in))
    smoke = tmp / "outputs" / "smoke_adapter" / "x"
    smoke.mkdir(parents=True)
    expect_raises("smoke adapter refused as DPO start", RuntimeError, lambda: T.verify_sft_adapter(cfg, smoke))
    check("config SHA == logged baseline SFT adapter_sha256",
          any(r["run_id"] == cfg["dpo"]["sft_run_id"] and r["adapter_sha256"] == cfg["dpo"]["expected_sft_adapter_sha256"]
              for r in __import__("csv").DictReader(open(log_path))), cfg["dpo"]["sft_run_id"])

    # 2
    records = [json.loads(l) for l in C.repo_path(cfg["data"]["preference_train_path"]).read_text().splitlines()]
    tok = C.load_tokenizer(cfg["model"]["model_id"])
    pre = T.precheck_pairs(tok, records, cfg["dpo"]["max_length"])
    check("all 150 pairs: prompt prefix exact, completion reproduces response, no truncation", pre["pairs"] == 150, str(pre))
    expect_raises("over-length pair would fail precheck (not truncate)", C.MaskingError,
                  lambda: T.precheck_pairs(tok, records[:1], 64))
    plan = T.planned_steps(150, cfg["dpo"]["micro_batch_size"], cfg["dpo"]["gradient_accumulation_steps"],
                           cfg["dpo"]["num_train_epochs"])
    check("step plan", plan["total_steps"] == 10 and plan["warmup_steps"] == 1, str(plan))

    # 3
    import argparse
    a = argparse.Namespace(beta=0.1, learning_rate=5e-5, epochs=1, max_steps=-1, micro_batch_size=2, grad_accum=8,
                           max_length=512, logging_steps=1)
    dc = T.build_dpo_config(cfg, a, tempfile.mkdtemp(), cpu_check=True)
    check("DPOConfig overrides applied",
          dc.beta == 0.1 and dc.learning_rate == 5e-5 and dc.bf16 is False and dc.logging_nan_inf_filter is False
          and dc.max_length == 512 and str(dc.optim).endswith("ADAMW_TORCH") and dc.max_grad_norm == 0.3
          and dc.auto_find_batch_size is False and dc.loss_type == ["sigmoid"] and dc.precompute_ref_log_probs is False,
          f"bf16={dc.bf16} lr={dc.learning_rate} optim={dc.optim} sched={dc.lr_scheduler_type} warmup={dc.warmup_steps}")

    # 4 + 5
    tr, setup, tok = build_trainer(cfg, records, stand_in, "fp32")
    check("TRL created frozen 'ref' adapter", list(tr.model.peft_config) == ["default", "ref"] and setup["ref_dtype"] == ["torch.float32"],
          f"adapters={list(tr.model.peft_config)} ref_dtype={setup['ref_dtype']}")
    check("no pairs dropped by TRL", len(tr.train_dataset) == 4)
    check("TRL cast policy to bf16 (quantized branch reproduced)", setup["policy_dtype_after_trl_init"] == ["torch.bfloat16"],
          str(setup["policy_dtype_after_trl_init"]))
    check("fp32 restore: policy == ref == SFT file bit-exact",
          setup["policy_dtype_final"] == ["torch.float32"] and setup["max_abs_policy_minus_sft_file"] == 0.0
          and setup["max_abs_ref_minus_sft_file"] == 0.0, json.dumps(setup))
    check("only policy LoRA trainable", setup["trainable_params"] == 2162688, f"{setup['trainable_params']:,}")

    # 7
    raw = tr.evaluate(metric_key_prefix="pre")
    check("TRL metric naming as handled by dpo_eval_metrics (eval_rewards/* + pre_loss)",
          "eval_rewards/margins" in raw and "pre_loss" in raw and "pre_rewards/margins" not in raw,
          "TRL ignores metric_key_prefix for rewards/*")
    pre_m = T.dpo_eval_metrics(raw, "pre")
    expect_raises("missing TRL metric fails loudly", KeyError, lambda: T.dpo_eval_metrics({"pre_loss": 1.0}, "pre"))
    mx = max(abs(pre_m["rewards/chosen"]), abs(pre_m["rewards/rejected"]), abs(pre_m["rewards/margins"]))
    check("step-0 rewards == 0 when policy == reference", mx < 1e-6,
          f"max|reward|={mx:.2e}, loss={pre_m['loss']:.6f} (ln2=0.693147)")

    # 8
    tr.model.train()
    batch = next(iter(tr.get_train_dataloader()))
    tr.current_gradient_accumulation_steps = 1   # set by the train() loop; needed to call training_step directly
    loss = tr.training_step(tr.model, batch)
    grads = {n: p.grad for n, p in tr.model.named_parameters() if p.grad is not None}
    pol = [n for n, g in grads.items() if ".default." in n and "lora_" in n and g.abs().sum() > 0]
    other = [n for n in grads if not (".default." in n and "lora_" in n)]
    check("training_step: finite loss", bool(torch.isfinite(loss)), f"loss={float(loss):.6f}")
    check("grads reach policy LoRA only (A and B: SFT B != 0)", len(pol) == 192 and not other,
          f"{len(pol)} policy tensors with grad; {len(other)} non-policy tensors with grad")

    # 9
    out = tmp / "dpo_save"
    tr.model.save_pretrained(out, selected_adapters=["default"], save_embedding_layers=False)
    saved = T.lora_file_tensors(out)
    check("saved adapter = policy LoRA only", len(saved) == 192 and not any(".ref." in k for k in saved)
          and (out / "adapter_config.json").exists() and not (out / "ref").exists(),
          f"{len(saved)} tensors; files={sorted(p.name for p in out.iterdir())}")
    del tr

    # 6
    tr2, setup2, _ = build_trainer(cfg, records, stand_in, "trl_default", n_pairs=2)
    check("trl_default mode keeps TRL bf16 policy (deviation measured)",
          setup2["policy_dtype_final"] == ["torch.bfloat16"] and setup2["max_abs_policy_minus_sft_file"] > 0,
          f"max|policy - SFT file|={setup2['max_abs_policy_minus_sft_file']:.3e}")
    pre2 = T.dpo_eval_metrics(tr2.evaluate(metric_key_prefix="pre"), "pre")
    check("trl_default: step-0 rewards nonzero (policy != reference)",
          abs(pre2["rewards/margins"]) > 0 or abs(pre2["rewards/chosen"]) > 0,
          f"chosen={pre2['rewards/chosen']:.3e} margin={pre2['rewards/margins']:.3e}")

    check("real experiment log unchanged", C.sha256_file(log_path) == log_hash)
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
