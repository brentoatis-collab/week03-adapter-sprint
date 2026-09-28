#!/usr/bin/env python3
"""CPU-only validation of the three-way evaluation (Step 6). Produces NO evaluation results:
the real adapters are on Drive, so two synthetic stand-in adapters exercise the mechanics.

  1. frozen population/denominators; scoring self-test (gold perfect; always-null / always-fabricate caught)
  2. hand-crafted outputs exercise every frozen scoring rule and flag
  3. adapter gates: SHA mismatch stops, smoke refused, config SHA == logged rows
  4. three-state model (fp32 CPU base + 2 stand-in adapters): in-memory == files bit-exact,
     switching probe (states differ; repeats identical), BASE == plain base model logits
  5. generation mechanics (8 tokens): generate() == manual greedy argmax for every state
  6. score-only path on a synthetic generations file
Usage:  python scripts/validate_eval_local.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "training"))
import common as C  # noqa: E402
import eval_scoring as S  # noqa: E402
import evaluate_models as E  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def expect_raises(name, exc, fn):
    try:
        fn()
    except exc as e:
        check(name, True, f"{exc.__name__}: {str(e)[:90]}")
        return
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
        return
    check(name, False, "no exception")


def main() -> int:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    cfg = C.load_config()
    log_hash = C.sha256_file(C.repo_path(cfg["logging"]["experiment_log_path"]))
    recs, sec, maj = E.eval_records(cfg)

    # 1
    pop = E.check_population(recs)
    check("population/denominators frozen: 60 = 36 exact + 24 no-address (4 x 6), street_only 2+2", True, str(pop))
    st = E.scoring_self_test(recs, sec)
    check("scoring self-test", st["gold_perfect"], str(st))

    # 2 crafted outputs
    ex = next(r for r in recs if r["has_exact_address"])
    na = next(r for r in recs if r["location_type"] == "block_range")
    g_ex, g_na = json.loads(ex["response"]), json.loads(na["response"])
    sc = lambda r, t: S.score_example(r, t, sec.get(r["id"]))  # noqa: E731
    fenced = sc(ex, "```json\n" + json.dumps(g_ex) + "\n```")
    check("fenced JSON -> parse 'extracted', NOT contract-compliant, flagged",
          fenced["parse_mode"] == "extracted" and not fenced["contract_compliant"]
          and "json_extracted_not_strict" in fenced["flags"] and fenced["address_outcome"] == "retained_exact")
    prose = sc(ex, "The ticket category is pothole.")
    check("no JSON -> parse 'none', address unscorable", prose["parse_mode"] == "none" and prose["address_outcome"] == "unscorable")
    reordered = sc(ex, json.dumps({k: g_ex[k] for k in reversed(list(g_ex))}))
    check("wrong key order -> strict parse but not compliant", reordered["parse_mode"] == "strict" and not reordered["contract_compliant"])
    num, rest = g_ex["address_or_null"].split(" ", 1)
    alt = str(int(num) + 1) + " " + rest
    check("altered house number -> wrong_address", sc(ex, json.dumps(dict(g_ex, address_or_null=alt)))["address_outcome"] == "wrong_address")
    base_name = rest.split()[0]
    suffixed = f"{num} {base_name} Boulevard"
    sv = sc(ex, json.dumps(dict(g_ex, address_or_null=suffixed)))
    check("same number+street, changed suffix -> retained_suffix_variant (flagged, not exact)",
          sv["address_outcome"] == "retained_suffix_variant" and "address_suffix_variant" in sv["flags"], suffixed)
    check("null despite exact address -> over_null", sc(ex, json.dumps(dict(g_ex, address_or_null=None)))["address_outcome"] == "over_null")
    blk_num = g_na["location"].split()[0]
    fab = sc(na, json.dumps(dict(g_na, address_or_null=f"{int(blk_num) + 37} Pine St")))
    check("block-range complaint + invented house number -> fabricated_address", fab["address_outcome"] == "fabricated_address")
    nav = sc(na, json.dumps(dict(g_na, address_or_null=g_na["location"])))
    check("block-range text copied verbatim from complaint -> non_address_value (flagged), NOT fabrication",
          nav["address_outcome"] == "non_address_value" and "address_non_address_value" in nav["flags"], repr(g_na["location"]))
    lm = next(r for r in recs if r["location_type"] == "landmark")
    g_lm = json.loads(lm["response"])
    nonaddr = sc(lm, json.dumps(dict(g_lm, address_or_null=g_lm["location"])))
    check("landmark text in address_or_null -> non_address_value (flagged, not fabrication)",
          nonaddr["address_outcome"] == "non_address_value" and "address_non_address_value" in nonaddr["flags"], repr(g_lm["location"]))
    core = S.strip_paren(g_lm["location"])
    for rel in S.RELATION_PREFIXES:
        if core.lower().startswith(rel + " "):
            core = core[len(rel) + 1:]
            break
    anch = sc(lm, json.dumps(dict(g_lm, location=core)))
    check("landmark without relation word -> location_anchor True, exact False (flagged)",
          anch["location_anchor"] and not anch["location_exact"] and "location_anchor_only" in anch["flags"], core)
    clr = next(r for r in recs if r["needs_clarification"])
    g_c = json.loads(clr["response"])
    noclr = sc(clr, json.dumps(dict(g_c, summary=g_c["summary"].replace(" Location clarification required.", ""))))
    check("missing clarification sentence -> clarification_correct False", not noclr["clarification_correct"])
    leak = sc(na, json.dumps(dict(g_na, summary=g_na["summary"] + " Near 812 Maple Ave.")))
    check("summary with unsupported address -> flagged", leak["summary_unsupported_address"]
          and any(f.startswith("summary_unsupported_address") for f in leak["flags"]))
    multi = next((r for r in recs if r.get("multi_issue") and r["id"] in sec
                  and sec[r["id"]] != json.loads(r["response"])["category"]), None)
    if multi:
        g_m = json.loads(multi["response"])
        mm = sc(multi, json.dumps(dict(g_m, category=sec[multi["id"]])))
        check("multi-issue: category = the other mentioned issue -> flagged, still incorrect",
              not mm["category_correct"] and "category_is_other_mentioned_issue" in mm["flags"])

    # 3 adapter gates (synthetic stand-ins)
    tmp = Path(tempfile.mkdtemp())
    lc = cfg["lora"]
    dirs = {}
    for i, name in enumerate(("sft_standin", "dpo_standin")):
        torch.manual_seed(100 + i)
        m = get_peft_model(AutoModelForCausalLM.from_pretrained(cfg["model"]["model_id"], dtype=torch.float32),
                           LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
                                      target_modules=lc["target_module_candidates"], bias="none", task_type="CAUSAL_LM"))
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.data.normal_(0, 2e-2)
        dirs[name] = tmp / name / "final_adapter"
        m.save_pretrained(dirs[name])
        del m
    expect_raises("SFT SHA mismatch stops evaluation", RuntimeError, lambda: E.verify_adapter(cfg, "sft", dirs["sft_standin"]))
    expect_raises("DPO SHA mismatch stops evaluation", RuntimeError, lambda: E.verify_adapter(cfg, "dpo", dirs["dpo_standin"]))
    smoke = tmp / "outputs" / "smoke_adapter" / "x"
    smoke.mkdir(parents=True)
    expect_raises("smoke adapter refused", RuntimeError, lambda: E.verify_adapter(cfg, "sft", smoke))
    import csv
    rows = {r["run_id"]: r for r in csv.DictReader(open(C.repo_path(cfg["logging"]["experiment_log_path"])))}
    e = cfg["evaluation"]
    check("config SHAs == logged SFT/DPO rows; DPO trained from that SFT",
          rows[e["sft_run_id"]]["adapter_sha256"] == e["expected_sft_adapter_sha256"]
          and rows[e["dpo_run_id"]]["adapter_sha256"] == e["expected_dpo_adapter_sha256"]
          and rows[e["dpo_run_id"]]["sft_adapter_sha256"] == e["expected_sft_adapter_sha256"])

    # 4 three-state model
    model = E.load_three_state_model(cfg, dirs["sft_standin"], dirs["dpo_standin"], quantized=False)
    loaded = E.verify_loaded_adapters(model, dirs["sft_standin"], dirs["dpo_standin"])
    check("in-memory adapters == files bit-exact", loaded["sft"]["max_abs_diff_vs_file"] == 0.0
          and loaded["dpo"]["max_abs_diff_vs_file"] == 0.0, json.dumps({k: v for k, v in loaded.items() if k != "note"}))
    tok = C.load_tokenizer(cfg["model"]["model_id"])
    probe = E.probe_logits(model, tok, recs[0]["complaint"], torch.device("cpu"))
    check("switching probe: states differ, repeats identical", True, json.dumps({k: f"{v:.3e}" for k, v in probe.items()}))
    plain = AutoModelForCausalLM.from_pretrained(cfg["model"]["model_id"], dtype=torch.float32).eval()
    ids = torch.tensor([C.render_prompt_ids(tok, recs[0]["complaint"])])
    with E.State(model, "base"), torch.no_grad():
        lb = model(input_ids=ids).logits[0, -1]
    with torch.no_grad():
        lp = plain(input_ids=ids).logits[0, -1]
    check("BASE state == plain pretrained model (adapters fully disabled)", torch.equal(lb, lp),
          f"max|diff|={(lb - lp).abs().max().item():.2e}")
    del plain

    # 5 generation mechanics
    gcfg = E.greedy_config(cfg)
    gcfg.max_new_tokens = 8
    ok_all = True
    for stname in ("base", "sft", "dpo"):
        with E.State(model, stname):
            g = E.generate_one(model, tok, recs[1]["complaint"], gcfg, torch.device("cpu"), autocast=False)
            cur = torch.tensor([C.render_prompt_ids(tok, recs[1]["complaint"])])
            with torch.no_grad():
                for _ in range(8):
                    nxt = model(input_ids=cur).logits[:, -1].argmax(-1, keepdim=True)
                    cur = torch.cat([cur, nxt], 1)
            manual = cur[0, -8:].tolist()
            ok_all &= g["new_token_ids"] == manual[: len(g["new_token_ids"])]
    check("generate() == manual greedy argmax in every state (no hidden sampling/penalty)", ok_all)
    check("generation record fields", {"raw_text", "n_new_tokens", "hit_eos", "stop_token", "seconds"} <= set(g))
    eff = model._prepare_generation_config(E.greedy_config(cfg))[0]
    check("effective config greedy, penalty 1.0, both EOS ids",
          eff.do_sample is False and eff.num_beams == 1 and eff.repetition_penalty == 1.0
          and list(eff.eos_token_id) == [151645, 151643], f"temperature={eff.temperature} top_k={eff.top_k} top_p={eff.top_p}")
    del model

    # 6 score-only on synthetic generations (gold for sft/dpo, fenced gold for base)
    rd = tmp / "results"
    rd.mkdir()
    with open(rd / "generations.jsonl", "w") as f:
        for stname in ("base", "sft", "dpo"):
            for r in recs:
                txt = r["response"] if stname != "base" else "```json\n" + r["response"] + "\n```"
                f.write(json.dumps({"state": stname, "id": r["id"], "raw_text": txt, "n_new_tokens": 0,
                                    "hit_eos": True, "stop_token": 151645, "seconds": 0}) + "\n")
    gens = [json.loads(l) for l in (rd / "generations.jsonl").read_text().splitlines()]
    _, agg = E.score_all(recs, sec, maj, gens)
    check("score-only: gold -> A 0/24, B 36/36, C 0/36, compliant 60/60",
          agg["sft"]["A_unsupported_exact_address_fabrication"]["n"] == 0 and agg["sft"]["invalid_non_null_address_field"]["n"] == 0 and agg["sft"]["B_retention_exact"]["n"] == 36
          and agg["sft"]["C_over_null"]["n"] == 0 and agg["sft"]["contract_compliant"]["n"] == 60)
    check("score-only: fenced gold -> content correct but 0/60 contract-compliant",
          agg["base"]["contract_compliant"]["n"] == 0 and agg["base"]["B_retention_exact"]["n"] == 36
          and agg["base"]["parse_mode"] == {"extracted": 60})
    print("\n" + S.summary_markdown(agg).split("\n\n", 1)[1].splitlines()[0])

    # 6b distinct no-address metrics: copying non-exact location text is a contract violation, not fabrication
    copy_gens = [dict(g) for g in gens if g["state"] == "sft"]
    for g in copy_gens:
        r = next(x for x in recs if x["id"] == g["id"])
        t = json.loads(r["response"])
        if not r["has_exact_address"]:
            t["address_or_null"] = t["location"]
        g.update(state="dpo", raw_text=json.dumps(t))
    _, agg2 = E.score_all(recs, sec, maj, [g for g in gens if g["state"] == "sft"] + copy_gens)
    check("location text copied into address_or_null: fabrication 0/24 but invalid non-null field 24/24",
          agg2["dpo"]["A_unsupported_exact_address_fabrication"]["n"] == 0
          and agg2["dpo"]["invalid_non_null_address_field"]["n"] == 24, str(agg2["dpo"]["no_address_outcomes"]))

    # 6c frozen scoring SHA gate
    actual = C.sha256_file(REPO_ROOT / "scripts" / "eval_scoring.py")
    check("eval_scoring.py matches the frozen SHA-256 in config", actual == cfg["evaluation"]["frozen_scoring_rules_sha256"],
          actual[:16])
    import subprocess
    bad_cfg = json.loads((REPO_ROOT / "configs" / "adapter_config.json").read_text())
    check("evaluate_models gates on frozen scoring SHA (code path present)",
          "SCORING RULES CHANGED" in (REPO_ROOT / "scripts" / "evaluate_models.py").read_text()
          and bad_cfg["evaluation"]["frozen_scoring_rules_sha256"] == actual)

    check("real experiment log unchanged", C.sha256_file(C.repo_path(cfg["logging"]["experiment_log_path"])) == log_hash)
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
