#!/usr/bin/env python3
"""CPU-only validation of the Step 3 training infrastructure.

Runs checks that are legitimate without a GPU. It performs NO optimizer step, no
quantization and writes nothing to logs/experiment_log.csv (verified by hash).

  1. frozen-data integrity
  2. masking negative tests (the fail-fast checks actually fire)
  3. collator padding / label masking
  4. loss masking equivalence: HF loss == manual cross-entropy over assistant tokens only
     (one forward/backward on 2 real records, fp32 CPU, LoRA attached)
  5. gradients reach LoRA parameters only
  6. CUDA helpers report NA (not fabricated numbers) without CUDA
  7. experiment logger on a TEMP copy of the CSV header (append-only, unknown keys rejected)
  8. train_qlora.py default mode refuses to run without CUDA and logs nothing

Usage:  python training/validate_local.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def expect_raises(name: str, exc: type, fn) -> None:
    try:
        fn()
    except exc as e:
        check(name, True, f"{exc.__name__}: {str(e)[:90]}")
        return
    except Exception as e:  # wrong exception type is a failure
        check(name, False, f"raised {type(e).__name__} instead of {exc.__name__}: {e}")
        return
    check(name, False, "no exception raised")


def main() -> int:
    import torch

    cfg = C.load_config()
    log_path = C.repo_path(cfg["logging"]["experiment_log_path"])
    log_hash_before = C.sha256_file(log_path)

    # 1
    h = C.verify_frozen_data(cfg)
    check("frozen data hashes match config", True, f"train={h['train'][:12]} eval={h['eval'][:12]}")
    train = C.load_jsonl(cfg["data"]["instruction_train_path"])
    tok = C.load_tokenizer(cfg["model"]["model_id"])
    max_len = cfg["sft"]["max_seq_length"]
    r0 = train[0]

    # 2
    expect_raises("overlength sequence is rejected, not truncated", C.MaskingError,
                  lambda: C.encode_sft_example(tok, r0["complaint"], r0["response"], max_length=64))

    class NoPrefixTok:  # prompt rendering that is NOT a prefix of the full rendering
        eos_token_id = tok.eos_token_id

        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_generation_prompt)["input_ids"]
            return {"input_ids": ([999] + list(ids)) if add_generation_prompt else list(ids)}

        def decode(self, *a, **k):
            return tok.decode(*a, **k)

    expect_raises("non-prefix prompt is rejected", C.MaskingError,
                  lambda: C.encode_sft_example(NoPrefixTok(), r0["complaint"], r0["response"], max_len))

    class NoEotTok(NoPrefixTok):  # full rendering missing the end-of-turn token
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
            ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_generation_prompt)["input_ids"])
            if not add_generation_prompt:
                p = len(tok.apply_chat_template(msgs[:2], tokenize=True, add_generation_prompt=True)["input_ids"])
                ids = ids[:p] + [i for i in ids[p:] if i != tok.eos_token_id]
            return {"input_ids": ids}

    expect_raises("missing assistant end-of-turn is rejected", C.MaskingError,
                  lambda: C.encode_sft_example(NoEotTok(), r0["complaint"], r0["response"], max_len))
    expect_raises("empty assistant response is rejected", C.MaskingError,
                  lambda: C.encode_sft_example(tok, r0["complaint"], "", max_len))

    enc = C.encode_sft_example(tok, r0["complaint"], r0["response"], max_len)
    eot_idx = enc["n_prompt"] + enc["n_trainable"] - 1
    check("end-of-turn token is trainable",
          enc["labels"][eot_idx] == tok.eos_token_id, f"label at boundary = {enc['labels'][eot_idx]}")
    check("all prompt positions masked", all(l == C.IGNORE_INDEX for l in enc["labels"][: enc["n_prompt"]]))
    check("token/label lengths equal", len(enc["labels"]) == len(enc["input_ids"]))

    # 3
    ds = C.SFTDataset(tok, train[:2], max_len)
    batch = C.PadCollator(tok.pad_token_id)([ds[0], ds[1]])
    shorter = min(range(2), key=lambda i: len(ds[i]["input_ids"]))
    n_short = len(ds[shorter]["input_ids"])
    pad_labels = batch["labels"][shorter, n_short:]
    check("collator pads labels with -100 and attention with 0",
          bool((pad_labels == C.IGNORE_INDEX).all()) and int(batch["attention_mask"][shorter, n_short:].sum()) == 0,
          f"batch shape {tuple(batch['input_ids'].shape)}, padded positions {pad_labels.numel()}")

    # 4 + 5
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    C.set_seed(cfg["sft"]["seed"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["model_id"], dtype=torch.float32)
    inv = C.linear_module_inventory(model)
    targets = C.verify_target_modules(inv, cfg["lora"]["target_module_candidates"])
    expect_raises("missing LoRA target is a hard error (no substitution)", RuntimeError,
                  lambda: C.verify_target_modules(inv, ["q_proj", "query_key_value"]))
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=0.0,
                                             target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    model.train()
    out = model(**batch)
    logits = out.logits.float()
    shift_logits, shift_labels = logits[:, :-1, :], batch["labels"][:, 1:]
    manual = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                               shift_labels.reshape(-1), ignore_index=C.IGNORE_INDEX)
    n_tok = int((shift_labels != C.IGNORE_INDEX).sum())
    check("HF loss == manual CE over assistant tokens only",
          torch.allclose(out.loss, manual, atol=1e-5),
          f"hf={out.loss.item():.6f} manual={manual.item():.6f} over {n_tok} supervised tokens (pre-training loss, not a result)")
    check("loss is finite", bool(torch.isfinite(out.loss)))
    out.loss.backward()
    lora_grads = [n for n, p in model.named_parameters() if p.grad is not None and "lora_B" in n and p.grad.abs().sum() > 0]
    base_grads = [n for n, p in model.named_parameters() if p.grad is not None and "lora_" not in n]
    check("gradients reach LoRA B matrices", len(lora_grads) == 96, f"{len(lora_grads)} lora_B tensors with nonzero grad")
    check("no gradients on frozen base parameters", not base_grads, f"{len(base_grads)} base tensors with grad")

    # 5b. LoRA gradient-check callback logic on these real CPU gradients
    from types import SimpleNamespace
    st = SimpleNamespace(global_step=0)
    cb = C.make_lora_grad_check_callback()
    cb.on_pre_optimizer_step(None, st, None, model=model)
    check("grad-check callback passes on real LoRA gradients",
          cb.result is not None and cb.result["lora_nonzero_grad"] == 96, str(cb.result))
    lora_p = next(p for n, p in model.named_parameters() if "lora_B" in n)
    saved = lora_p.grad.clone()
    lora_p.grad[0, 0] = float("inf")                       # simulate an fp16 overflow step
    cb2 = C.make_lora_grad_check_callback()
    cb2.on_pre_optimizer_step(None, st, None, model=model)
    check("grad-check treats AMP overflow step as skip, not failure",
          cb2.result is None and cb2.overflow_steps == [1], f"overflow_steps={cb2.overflow_steps}")
    expect_raises("grad-check fails if training ends with no finite step", C.GradientFlowError,
                  lambda: cb2.on_train_end(None, st, None))
    lora_p.grad = saved
    base_p = next(p for n, p in model.named_parameters() if "lora_" not in n)
    base_p.grad = torch.zeros_like(base_p)
    expect_raises("grad-check fails if a frozen base parameter has a gradient", C.GradientFlowError,
                  lambda: C.make_lora_grad_check_callback().on_pre_optimizer_step(None, st, None, model=model))
    base_p.grad = None
    del model

    # 6
    mem = C.peak_memory_gb()
    check("CUDA memory helper reports NA without CUDA",
          not torch.cuda.is_available() and mem["peak_allocated_gb"] == C.NA_NO_CUDA, str(mem))
    check("gpu_name reports NA without CUDA", C.gpu_name() == C.NA_NO_CUDA)
    check("package versions captured", C.package_versions()["transformers"] == "5.17.0", str(C.package_versions()))
    check("git commit captured", C.git_commit() != "unknown", C.git_commit())

    # 7
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "log.csv"
        shutil.copy(log_path, tmp)
        header_line = tmp.read_text()
        C.append_experiment_log({"run_id": "LOGGER-SELFTEST", "status": "logger_selftest"}, tmp)
        C.append_experiment_log({"run_id": "LOGGER-SELFTEST-2", "status": "logger_selftest"}, tmp)
        lines = tmp.read_text().splitlines()
        check("logger appends without rewriting header",
              tmp.read_text().startswith(header_line) and len(lines) == 3, f"{len(lines)} lines in temp log")
        expect_raises("logger rejects unknown columns", KeyError,
                      lambda: C.append_experiment_log({"run_id": "x", "fake_metric": 1}, tmp))

    # 8a. TrainingArguments built by the real code path (fp16 disabled only because it is GPU-only)
    import train_qlora
    argv, sys.argv = sys.argv, ["train_qlora.py"]
    try:
        a = train_qlora.parse_args(cfg)
    finally:
        sys.argv = argv
    with tempfile.TemporaryDirectory() as td:
        ta = train_qlora.build_training_args(cfg, a, Path(td), cpu_check=True)
    import math
    per_epoch = math.ceil(math.ceil(len(train) / a.micro_batch_size) / a.grad_accum)
    total = per_epoch * int(a.epochs)
    check("TrainingArguments constructs with pinned API names",
          ta.eval_strategy == "steps" and ta.lr_scheduler_type == "cosine" and ta.max_grad_norm == 0.3
          and ta.optim == "adamw_torch" and ta.logging_nan_inf_filter is False and ta.auto_find_batch_size is False
          and ta.gradient_checkpointing is True and ta.gradient_checkpointing_kwargs == {"use_reentrant": False},
          f"eval={ta.eval_strategy} sched={ta.lr_scheduler_type} optim={ta.optim} nan_filter={ta.logging_nan_inf_filter} "
          f"auto_bs={ta.auto_find_batch_size}")
    check("warmup ratio 0.1 resolves to ~10% of steps",
          ta.get_warmup_steps(total) == math.ceil(0.1 * total),
          f"~{total} optimizer steps planned ({per_epoch}/epoch) -> {ta.get_warmup_steps(total)} warmup steps")

    # 8c. smoke-run isolation guards
    script = str(Path(__file__).parent / "train_qlora.py")
    p1 = subprocess.run([sys.executable, script, "--run-label", "smoke", "--dry-run"], capture_output=True, text=True)
    check("'smoke' without --max-steps is rejected", p1.returncode == 2 and "requires --max-steps" in p1.stderr)
    p2 = subprocess.run([sys.executable, script, "--max-steps", "5", "--dry-run"], capture_output=True, text=True)
    check("'baseline' with --max-steps is rejected", p2.returncode == 2 and "non-baseline" in p2.stderr)
    argv, sys.argv = sys.argv, ["train_qlora.py", "--run-label", "smoke", "--max-steps", "5"]
    try:
        sa = train_qlora.parse_args(cfg)
    finally:
        sys.argv = argv
    check("smoke output routed away from production SFT dir",
          sa.output_dir == cfg["sft"]["smoke_output_dir"] != cfg["sft"]["output_dir"], sa.output_dir)
    p3 = subprocess.run([sys.executable, script, "--run-label", "smoke", "--max-steps", "5", "--logging-steps", "1",
                         "--eval-steps", "5", "--dry-run"], capture_output=True, text=True)
    check("smoke dry-run (exact notebook flags) passes", p3.returncode == 0 and "DRY RUN complete" in p3.stdout)
    p4 = subprocess.run([sys.executable, script, "--run-label", "smoke", "--max-steps", "5", "--logging-steps", "1",
                         "--eval-steps", "5"], capture_output=True, text=True)
    check("smoke CUDA path refuses without CUDA", p4.returncode != 0 and "CUDA is not available" in p4.stderr)

    # 8b
    proc = subprocess.run([sys.executable, str(Path(__file__).parent / "train_qlora.py"), "--run-label", "cpu-refusal-check"],
                          capture_output=True, text=True)
    check("train_qlora.py refuses to train without CUDA",
          proc.returncode != 0 and "CUDA is not available" in proc.stderr, f"exit={proc.returncode}")

    check("real experiment log unchanged", C.sha256_file(log_path) == log_hash_before)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
