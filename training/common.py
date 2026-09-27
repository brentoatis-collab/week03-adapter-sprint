"""Shared infrastructure for CivicDesk SFT, DPO and evaluation.

One source of truth for:
  * the system prompt and output contract,
  * chat formatting (the tokenizer's own chat template, never hand-built ChatML),
  * assistant-only label construction with fail-fast checks,
  * architecture inspection and trainable-parameter reporting,
  * CUDA memory / wall-clock instrumentation (safe when CUDA is absent),
  * the append-only experiment log.

Verified against the pinned stack (transformers 5.17.0, peft 0.21.0):
  * apply_chat_template(tokenize=True) returns a BatchEncoding dict by default
    (return_dict=True), so token ids are read from ["input_ids"].
  * The Qwen2.5 chat template has no {% generation %} tag; the built-in
    return_assistant_tokens_mask silently returns an all-zero mask. Masking is
    therefore explicit (prompt-prefix method) and validated here.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "adapter_config.json"

# The output contract lives with the data generator; import it so the prompt,
# data and evaluation can never disagree.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_instruction_data import (  # noqa: E402
    CATEGORIES, CLARIFICATION_SENTENCE, LOCATION_UNSPECIFIED, TICKET_KEYS, URGENCY_LEVELS,
)

IGNORE_INDEX = -100
NA_NO_CUDA = "NA_no_cuda"

# ---------------------------------------------------------------------------
# System prompt (identical for base, SFT and DPO evaluation)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are CivicDesk, a municipal 311 dispatch assistant. Convert the resident's complaint into one dispatch ticket.

Output exactly one JSON object and nothing else, with keys in this order:
{{"category": ..., "urgency": ..., "location": ..., "summary": ..., "address_or_null": ...}}

category: one of {", ".join(CATEGORIES)}. If several issues are reported, use the most urgent one (if tied, the first mentioned).
urgency: emergency (immediate danger to life or safety, or damage happening now), high (hazard likely to cause injury or damage soon, or loss of an essential service), medium (needs action within days, no immediate hazard), low (cosmetic or not time-sensitive). Judge from the stated facts, not from how urgent the resident says it is.
location: the most specific location the resident gave, or "{LOCATION_UNSPECIFIED}" if none.
summary: one or two neutral sentences using only facts in the complaint. Mention a second issue as "Also reports ...". Add "{CLARIFICATION_SENTENCE}" only if a dispatcher could not find the issue from the location given.
address_or_null: the house number and street as the resident wrote them, only if both were given; otherwise null. Never invent, guess or complete an address."""


# ---------------------------------------------------------------------------
# Config / data
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or CONFIG_PATH).read_text())


def repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


REQUIRED_FIELDS = ("id", "complaint", "response", "has_exact_address", "location_type", "category", "urgency")


def load_jsonl(path: str | Path, required: tuple[str, ...] = REQUIRED_FIELDS) -> list[dict]:
    """Load JSONL and validate that each record has the fields and a contract-valid response."""
    records = []
    for n, line in enumerate(repo_path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        missing = [f for f in required if f not in r]
        if missing:
            raise ValueError(f"{path}:{n} missing fields {missing}")
        ticket = json.loads(r["response"])
        if tuple(ticket) != TICKET_KEYS:
            raise ValueError(f"{path}:{n} response keys {list(ticket)} != contract {list(TICKET_KEYS)}")
        records.append(r)
    if not records:
        raise ValueError(f"{path} is empty")
    return records


def verify_frozen_data(cfg: dict) -> dict[str, str]:
    """Compare instruction JSONL hashes with the frozen hashes recorded in the config."""
    frozen = cfg["data"].get("frozen_sha256", {})
    out = {}
    for split, key in (("train", "instruction_train_path"), ("eval", "instruction_eval_path")):
        actual = sha256_file(repo_path(cfg["data"][key]))
        expected = frozen.get(split)
        if expected is None:
            raise RuntimeError(f"no frozen_sha256 recorded for '{split}' in config")
        if actual != expected:
            raise RuntimeError(f"FROZEN DATA MISMATCH for {split}: expected {expected[:16]}..., got {actual[:16]}...")
        out[split] = actual
    return out


# ---------------------------------------------------------------------------
# Chat formatting + assistant-only masking
# ---------------------------------------------------------------------------
class MaskingError(RuntimeError):
    """Raised when assistant-only label construction cannot be trusted."""


def build_messages(complaint: str, response: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": complaint}]
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.chat_template is None:
        raise RuntimeError(f"{model_id} tokenizer has no chat template")
    if tok.pad_token_id is None:
        raise RuntimeError(f"{model_id} tokenizer has no pad token; refusing to guess one")
    if tok.pad_token_id == tok.eos_token_id:
        # If pad == eos, padding-based masking could also mask the real end-of-turn token.
        raise RuntimeError("pad_token_id == eos_token_id; end-of-turn supervision would be ambiguous")
    tok.padding_side = "right"
    return tok


def render_prompt_ids(tokenizer, complaint: str) -> list[int]:
    """Token ids for system + user, through the assistant generation boundary."""
    enc = tokenizer.apply_chat_template(build_messages(complaint), tokenize=True, add_generation_prompt=True)
    return list(enc["input_ids"])


def encode_sft_example(tokenizer, complaint: str, response: str, max_length: int) -> dict:
    """Tokenize one conversation and build assistant-only labels.

    labels == IGNORE_INDEX for every system/user/template token; assistant content
    tokens and the assistant end-of-turn token (<|im_end|>, the EOS the model must
    learn to emit) are trainable. Template tokens after the end-of-turn (Qwen emits
    a trailing newline) are masked: generation stops at EOS, so they are never produced.
    """
    prompt_ids = render_prompt_ids(tokenizer, complaint)
    full = tokenizer.apply_chat_template(build_messages(complaint, response), tokenize=True)
    input_ids = list(full["input_ids"])

    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise MaskingError("prompt token ids are not an exact prefix of the full conversation ids")

    eot = tokenizer.eos_token_id
    try:
        eot_pos = input_ids.index(eot, len(prompt_ids))
    except ValueError:
        raise MaskingError("no assistant end-of-turn token found after the prompt") from None

    labels = [IGNORE_INDEX] * len(input_ids)
    for i in range(len(prompt_ids), eot_pos + 1):
        labels[i] = input_ids[i]

    n_train = sum(1 for x in labels if x != IGNORE_INDEX)
    if len(labels) != len(input_ids):
        raise MaskingError("label/token length mismatch")
    if n_train == 0:
        raise MaskingError("no trainable assistant tokens")
    if n_train == len(labels):
        raise MaskingError("every token is trainable; prompt was not masked")
    if n_train < 2:  # at minimum some content + end-of-turn
        raise MaskingError("assistant region contains only the end-of-turn token")
    decoded = tokenizer.decode(input_ids[len(prompt_ids): eot_pos], skip_special_tokens=False)
    if decoded.strip() != response.strip():
        raise MaskingError(f"decoded assistant region does not reproduce the response: {decoded!r}")
    if len(input_ids) > max_length:
        raise MaskingError(f"sequence has {len(input_ids)} tokens > max_length {max_length}; "
                           "refusing to truncate the supervised response")

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels,
            "n_prompt": len(prompt_ids), "n_trainable": n_train, "n_total": len(input_ids)}


def masking_report(tokenizer, record: dict, enc: dict) -> str:
    ids, labels = enc["input_ids"], enc["labels"]
    masked = [i for i, l in zip(ids, labels) if l == IGNORE_INDEX]
    trainable = [i for i, l in zip(ids, labels) if l != IGNORE_INDEX]
    prompt_tail = tokenizer.decode(ids[max(0, enc["n_prompt"] - 12): enc["n_prompt"]], skip_special_tokens=False)
    after = tokenizer.decode(ids[enc["n_prompt"] + enc["n_trainable"]:], skip_special_tokens=False)
    lines = [
        f"record            : {record['id']} (location_type={record['location_type']}, "
        f"has_exact_address={record['has_exact_address']})",
        f"complaint         : {record['complaint']}",
        f"prompt boundary   : ...{prompt_tail!r}   <-- last MASKED tokens (ends at assistant header)",
        f"TRAINABLE region  : {tokenizer.decode(trainable, skip_special_tokens=False)!r}",
        f"masked after EOT  : {after!r}",
        f"tokens total      : {len(ids)}",
        f"masked tokens     : {len(masked)}",
        f"trainable tokens  : {len(trainable)}",
        f"loss-bearing share: {100 * len(trainable) / len(ids):.1f}%",
    ]
    return "\n".join(lines)


class SFTDataset:
    """Pre-tokenized, masked examples (a plain indexable list for the HF Trainer)."""

    def __init__(self, tokenizer, records: list[dict], max_length: int):
        self.items = [encode_sft_example(tokenizer, r["complaint"], r["response"], max_length) for r in records]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        return {"input_ids": it["input_ids"], "attention_mask": it["attention_mask"], "labels": it["labels"]}

    def length_stats(self) -> dict:
        tot = sorted(x["n_total"] for x in self.items)
        tr = sorted(x["n_trainable"] for x in self.items)
        pct = lambda v, q: v[min(len(v) - 1, int(q * len(v)))]  # noqa: E731
        return {"n": len(tot), "total_min": tot[0], "total_median": pct(tot, 0.5), "total_p95": pct(tot, 0.95),
                "total_max": tot[-1], "trainable_min": tr[0], "trainable_median": pct(tr, 0.5), "trainable_max": tr[-1],
                "loss_share_mean": sum(tr) / sum(tot)}


class PadCollator:
    """Right-pads a batch; padded label positions are IGNORE_INDEX."""

    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            n = width - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad] * n)
            out["attention_mask"].append(b["attention_mask"] + [0] * n)
            out["labels"].append(b["labels"] + [IGNORE_INDEX] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import torch
    from transformers import set_seed as hf_set_seed

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)


# ---------------------------------------------------------------------------
# Architecture inspection / trainable parameters
# ---------------------------------------------------------------------------
def linear_module_inventory(model) -> dict[str, dict]:
    """Group Linear-like modules (incl. bitsandbytes Linear4bit) by leaf name."""
    inv: dict[str, dict] = {}
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if not (cls == "Linear" or cls.startswith("Linear")):
            continue
        leaf = name.rsplit(".", 1)[-1]
        shape = (getattr(mod, "in_features", None), getattr(mod, "out_features", None))
        e = inv.setdefault(leaf, {"count": 0, "classes": set(), "shapes": set(), "example": name})
        e["count"] += 1
        e["classes"].add(cls)
        e["shapes"].add(shape)
    return inv


def print_linear_inventory(inv: dict[str, dict]) -> None:
    print(f"  {'module leaf':14s} {'count':>5s}  {'(in, out)':22s} class        example")
    for leaf, e in sorted(inv.items()):
        shapes = ", ".join(str(s) for s in sorted(e["shapes"]))
        print(f"  {leaf:14s} {e['count']:5d}  {shapes:22s} {'/'.join(sorted(e['classes'])):12s} {e['example']}")


def verify_target_modules(inv: dict[str, dict], candidates: list[str]) -> list[str]:
    """Return candidates verified to exist; raise if any is missing. Never substitutes."""
    missing = [c for c in candidates if c not in inv]
    if missing:
        raise RuntimeError(f"LoRA target modules not found in model: {missing}. "
                           f"Available linear leaves: {sorted(inv)}. Refusing to substitute targets.")
    return list(candidates)


def trainable_parameter_report(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return {"total": total, "trainable": trainable, "trainable_pct": 100.0 * trainable / total if total else 0.0,
            "trainable_names": names}


def assert_trainable_adapters(report: dict) -> None:
    if report["trainable"] == 0:
        raise RuntimeError("ZERO trainable parameters: LoRA adapters were not attached or were frozen.")
    non_lora = [n for n in report["trainable_names"] if "lora_" not in n]
    if non_lora:
        raise RuntimeError(f"non-adapter parameters are trainable (base must be frozen): {non_lora[:5]}")


def lora_param_prediction(cfg_model, r: int, targets: list[str]) -> dict:
    """Arithmetic LoRA parameter count from the model config: r * (in + out) per target per layer."""
    h = cfg_model.hidden_size
    head_dim = getattr(cfg_model, "head_dim", None) or h // cfg_model.num_attention_heads
    kv = cfg_model.num_key_value_heads * head_dim
    dims = {"q_proj": (h, cfg_model.num_attention_heads * head_dim), "k_proj": (h, kv), "v_proj": (h, kv),
            "o_proj": (cfg_model.num_attention_heads * head_dim, h)}
    per_layer = {t: r * (dims[t][0] + dims[t][1]) for t in targets}
    return {"per_layer": per_layer, "per_layer_total": sum(per_layer.values()),
            "layers": cfg_model.num_hidden_layers, "total": sum(per_layer.values()) * cfg_model.num_hidden_layers}


# ---------------------------------------------------------------------------
# CUDA memory + timing (safe without CUDA)
# ---------------------------------------------------------------------------
def cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else NA_NO_CUDA


def reset_peak_memory() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> dict[str, Any]:
    """Peak allocated/reserved since the last reset, in GiB. NA when CUDA is absent."""
    import torch
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": NA_NO_CUDA, "peak_reserved_gb": NA_NO_CUDA}
    torch.cuda.synchronize()
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)}


class MemoryPhases:
    """Per-phase CUDA peaks (load / train / eval) without changing allocator behavior.

    mark(phase) folds the peak since the previous mark into `phase`, then resets the peak
    counters. The global peak is the max over phases, i.e. the same quantity as a single
    reset before model load. Cumulative allocator counters (e.g. num_alloc_retries) are
    never reset, so they cover the whole process.
    """

    def __init__(self):
        self.peaks: dict[str, dict[str, float]] = {}
        self.active = "load"
        reset_peak_memory()

    def mark(self, phase: str, next_phase: str | None = None) -> None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            a, r = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
            p = self.peaks.setdefault(phase, {"allocated": 0.0, "reserved": 0.0})
            p["allocated"], p["reserved"] = max(p["allocated"], a), max(p["reserved"], r)
            torch.cuda.reset_peak_memory_stats()
        self.active = next_phase or phase

    def row_fields(self) -> dict[str, Any]:
        import torch
        if not torch.cuda.is_available():
            out = {"peak_allocated_gb": NA_NO_CUDA, "peak_reserved_gb": NA_NO_CUDA}
            for ph in ("load", "train", "eval"):
                out[f"peak_allocated_{ph}_gb"] = out[f"peak_reserved_{ph}_gb"] = NA_NO_CUDA
            return out
        out = {"peak_allocated_gb": round(max((p["allocated"] for p in self.peaks.values()), default=0.0), 3),
               "peak_reserved_gb": round(max((p["reserved"] for p in self.peaks.values()), default=0.0), 3)}
        for ph in ("load", "train", "eval"):
            p = self.peaks.get(ph)
            out[f"peak_allocated_{ph}_gb"] = round(p["allocated"], 3) if p else "NA_phase_not_reached"
            out[f"peak_reserved_{ph}_gb"] = round(p["reserved"], 3) if p else "NA_phase_not_reached"
        return out


def allocator_stats() -> dict[str, Any]:
    """Caching-allocator counters that separate tensor demand from cache retention.

    alloc_retries > 0 means a cudaMalloc failed and the allocator freed cached blocks and
    retried, i.e. the cache had grown to the device limit (not an OOM by itself).
    peak_inactive_split_gb = cached-but-unused space inside split blocks (fragmentation).
    Values are cumulative/peak since process start or the last peak reset.
    """
    import torch
    if not torch.cuda.is_available():
        return {"alloc_retries": NA_NO_CUDA, "cuda_ooms": NA_NO_CUDA, "peak_inactive_split_gb": NA_NO_CUDA,
                "peak_segments": NA_NO_CUDA}
    s = torch.cuda.memory_stats()
    miss = "NA_key_missing"
    gb = lambda k: round(s[k] / 2**30, 3) if k in s else miss  # noqa: E731
    return {"alloc_retries": s.get("num_alloc_retries", miss), "cuda_ooms": s.get("num_ooms", miss),
            "peak_inactive_split_gb": gb("inactive_split_bytes.all.peak"), "peak_segments": s.get("segment.all.peak", miss)}


def environment_provenance() -> dict[str, Any]:
    import platform
    import torch
    out = {"python_version": platform.python_version(),
           "torch_cuda_build": torch.version.cuda or "none",
           "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "none",
           "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "unset")}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        out["gpu_capability"] = f"{cap[0]}.{cap[1]}"
        out["gpu_total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)
    else:
        out["gpu_capability"] = out["gpu_total_memory_gb"] = NA_NO_CUDA
    return out


class WallClock:
    def __enter__(self):
        self.start = time.perf_counter()
        self.seconds = None
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False  # never swallow exceptions


def make_step_timer_callback():
    """TrainerCallback recording per-optimizer-step durations (excludes evaluation)."""
    import torch
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.durations: list[float] = []
            self._t0 = None

        def on_step_begin(self, args, state, control, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self._t0 is not None:
                self.durations.append(time.perf_counter() - self._t0)

        def summary(self) -> dict:
            d = self.durations
            steady = d[1:] if len(d) > 1 else d
            return {"steps_timed": len(d),
                    "avg_step_time_s": round(sum(steady) / len(steady), 4) if steady else "NA",
                    "first_step_time_s": round(d[0], 4) if d else "NA"}

    return StepTimer()


class GradientFlowError(RuntimeError):
    pass


def make_lora_grad_check_callback():
    """Inspect real gradients once, at the first optimizer step (after clipping/unscaling,
    before optimizer.step()). Raises if no LoRA tensor received a finite nonzero gradient or
    if any frozen base parameter has a gradient."""
    import torch
    from transformers import TrainerCallback

    class LoraGradCheck(TrainerCallback):
        """fp16 AMP may produce inf grads on early steps (GradScaler then skips the step and
        lowers its scale). Such steps are recorded, not treated as failures; the check passes
        at the first step with finite LoRA grads and fails if training ends without one."""

        def __init__(self):
            self.result: dict | None = None
            self.overflow_steps: list[int] = []

        def on_train_end(self, args, state, control, **kw):
            if self.result is None:
                raise GradientFlowError(f"no optimizer step had finite LoRA gradients "
                                        f"(AMP overflow steps: {self.overflow_steps})")

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            if self.result is not None or model is None:
                return
            lora_total = lora_nonzero = lora_nonfinite = 0
            base_with_grad = []
            for n, p in model.named_parameters():
                if "lora_" in n and p.requires_grad:
                    lora_total += 1
                    if p.grad is not None:
                        g = p.grad.detach()
                        if not torch.isfinite(g).all():
                            lora_nonfinite += 1
                        elif g.abs().sum() > 0:
                            lora_nonzero += 1
                elif p.grad is not None:
                    base_with_grad.append(n)
            if base_with_grad:
                raise GradientFlowError(f"frozen base parameters received gradients: {base_with_grad[:5]}")
            step = state.global_step + 1
            if lora_nonfinite:
                self.overflow_steps.append(step)
                print(f"\nGRADIENT CHECK step {step}: {lora_nonfinite} LoRA tensors non-finite "
                      "(fp16 overflow; GradScaler skips this step) - rechecking next step")
                return
            if lora_nonzero == 0:
                raise GradientFlowError(f"step {step}: no LoRA tensor received a nonzero gradient "
                                        f"({lora_total} LoRA tensors)")
            self.result = {"step": step, "lora_tensors": lora_total, "lora_nonzero_grad": lora_nonzero,
                           "base_params_with_grad": 0, "amp_overflow_steps_before": list(self.overflow_steps)}
            print(f"\nGRADIENT CHECK passed: {self.result}")

    return LoraGradCheck()


class NonFiniteLossError(RuntimeError):
    pass


def check_finite_loss(loss_value: float, step: int) -> None:
    if not math.isfinite(loss_value):
        raise NonFiniteLossError(f"non-finite training loss {loss_value} at global step {step}")


# ---------------------------------------------------------------------------
# Provenance + experiment log
# ---------------------------------------------------------------------------
TRACKED_PACKAGES = ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets")


def package_versions() -> dict[str, str]:
    out = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = "not_installed"
    return out


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(method: str, label: str) -> str:
    return f"{method}-{label}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def append_experiment_log(row: dict, path: str | Path | None = None) -> Path:
    """Append one row. The header is read from the existing file (never rewritten).

    Unknown keys raise; missing keys are written as empty strings.
    """
    path = repo_path(path or load_config()["logging"]["experiment_log_path"])
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    unknown = sorted(set(row) - set(header))
    if unknown:
        raise KeyError(f"experiment-log row has keys not in the CSV header: {unknown}")
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writerow({k: row.get(k, "") for k in header})
    return path
