#!/usr/bin/env python3
"""Environment preflight for the Colab QLoRA path (also runnable locally without CUDA).

Reports and checks, in order:
  1. Python + installed package versions vs requirements.txt pins (exact pins must match;
     torch must satisfy its range).
  2. CUDA runtime, GPU name, compute capability, total memory, compiled arch list,
     bf16 support (informational: T4 has none, so fp16 is used), fp16 matmul.
  3. bitsandbytes import and a real NF4 + double-quant quantize/dequantize round trip on GPU.
  4. Git commit, clean working tree, optional expected commit, required ancestor commit.
  5. Frozen dataset hashes from configs/adapter_config.json.

Exit code 1 on any FAIL. Nothing is fixed or reinstalled automatically.

Usage:
    python training/preflight.py --require-cuda                       # Colab
    python training/preflight.py --require-cuda --expected-commit abc1234
    python training/preflight.py                                      # local (CUDA checks skipped)
"""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FAILS: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  [FAIL] {msg}")


def info(msg: str) -> None:
    print(f"  [info] {msg}")


def parse_requirements(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    exact, ranged = {}, {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|<=|~=|<|>)(.+)$", line)
        if not m:
            continue
        name, op, rest = m.group(1).lower(), m.group(2), m.group(3).strip()
        (exact if op == "==" else ranged)[name] = rest if op == "==" else op + rest
    return exact, ranged


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v.split("+")[0])[:3])


def satisfies(v: str, spec: str) -> bool:
    for part in spec.split(","):
        m = re.match(r"(>=|<=|<|>|==)\s*(.+)", part.strip())
        op, target = m.group(1), version_tuple(m.group(2))
        cur = version_tuple(v)
        if not {">=": cur >= target, "<=": cur <= target, "<": cur < target,
                ">": cur > target, "==": cur == target}[op]:
            return False
    return True


def section(t: str) -> None:
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--expected-commit", default="", help="if set, HEAD must start with this")
    ap.add_argument("--require-ancestor", default="60a1789", help="commit that must be in HEAD's history")
    args = ap.parse_args()

    section("1. PYTHON + PACKAGES vs requirements.txt")
    info(f"python {platform.python_version()} ({sys.executable})")
    exact, ranged = parse_requirements(C.REPO_ROOT / "requirements.txt")
    for name, want in exact.items():
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            have = None
        if have == want:
            ok(f"{name:14s} {have}")
        elif have is None and not args.require_cuda and name in ("bitsandbytes", "trl", "datasets"):
            info(f"{name:14s} not installed (not needed for local validation)")
        else:
            fail(f"{name:14s} installed={have} pinned={want}")
    for name, spec in ranged.items():
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            (fail if args.require_cuda or name == "torch" else info)(f"{name} not installed (required {spec})")
            continue
        (ok if satisfies(have, spec) else fail)(f"{name:14s} {have} satisfies {spec}" if satisfies(have, spec)
                                                else f"{name} {have} violates {spec}")

    section("2. CUDA / GPU")
    import torch
    info(f"torch {torch.__version__} | built for CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        (fail if args.require_cuda else info)("CUDA not available" + (" - this runtime cannot run QLoRA. In Colab: "
                                              "Runtime > Change runtime type > T4 GPU." if args.require_cuda else
                                              " (local run: CUDA checks skipped)"))
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        total_gib = props.total_memory / 2**30
        arch = torch.cuda.get_arch_list()
        info(f"GPU {name} | compute capability {cap[0]}.{cap[1]} | total memory {total_gib:.2f} GiB")
        info(f"torch compiled arch list: {arch}")
        sm = f"sm_{cap[0]}{cap[1]}"
        if sm in arch or any(a.startswith("compute_") and version_tuple(a) <= version_tuple(f"{cap[0]}{cap[1]}")
                             for a in arch):
            ok(f"torch build includes kernels for {sm}")
        else:
            fail(f"torch build has no kernels for {sm} (arch list {arch})")
        (ok if cap >= (7, 0) else fail)(f"compute capability {cap} {'>=' if cap >= (7, 0) else '<'} 7.0 for fp16 tensor cores")
        if "T4" not in name:
            info(f"GPU is not a T4 ({name}); results will not represent the T4 budget")
        info(f"bf16 supported: {torch.cuda.is_bf16_supported()} (expected False on T4; fp16 compute is used)")
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        prod = a @ a
        (ok if torch.isfinite(prod).all() else fail)("fp16 matmul on GPU is finite")
        info(f"15 GB ceiling vs device: device reports {total_gib:.2f} GiB total")

    section("3. BITSANDBYTES")
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as F
        ok(f"bitsandbytes {bnb.__version__} imported")
        if torch.cuda.is_available():
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
            packed, qs = F.quantize_4bit(x, blocksize=64, compress_statistics=True, quant_type="nf4")
            y = F.dequantize_4bit(packed, qs)
            rel = ((y.float() - x.float()).norm() / x.float().norm()).item()
            info(f"NF4 round trip: packed {tuple(packed.shape)} {packed.dtype}, quant_type={qs.quant_type}, "
                 f"nested(double quant)={qs.nested}, relative error={rel:.4f}")
            (ok if qs.quant_type == "nf4" and qs.nested and torch.isfinite(y).all() and rel < 0.2 else fail)(
                "NF4 + double-quant GPU kernels work")
    except ImportError as e:
        (fail if args.require_cuda else info)(f"bitsandbytes not importable: {e}")

    section("4. GIT")
    head = C.git_commit()
    info(f"HEAD {head}")
    (ok if not head.endswith("-dirty") else fail)("working tree clean" if not head.endswith("-dirty")
                                                  else "working tree has uncommitted changes")
    if args.expected_commit:
        (ok if head.startswith(args.expected_commit) else fail)(f"HEAD matches expected {args.expected_commit}")
    if args.require_ancestor:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", args.require_ancestor, "HEAD"], cwd=C.REPO_ROOT)
        (ok if r.returncode == 0 else fail)(f"HEAD contains required commit {args.require_ancestor}")

    section("5. FROZEN DATA")
    try:
        h = C.verify_frozen_data(C.load_config())
        ok(f"instruction data hashes match config: train={h['train'][:16]} eval={h['eval'][:16]}")
    except RuntimeError as e:
        fail(str(e))

    section("PREFLIGHT RESULT")
    if FAILS:
        print(f"  FAILED ({len(FAILS)}):")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
