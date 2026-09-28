# Week 03 — Adapter Sprint

**LoRA/QLoRA instruction tuning and DPO under a 15 GB GPU memory ceiling**

> **Status:** Build Step 5B (DPO path implemented and CPU-validated; not yet run). GPU runs so far: 5-step QLoRA smoke test and the full baseline QLoRA SFT run.
> Every result field in this README is empty until it is filled from real Colab execution.

---

## 1. Scenario

CivicDesk receives free-text 311 complaints from residents. They arrive messy: typos, slang,
irrelevant detail, vague locations, several issues at once. Dispatchers need a **structured
ticket** they can route without re-reading the complaint.

As CivicDesk's Junior NLP Engineer, I am testing whether a **small open pretrained model**
(`Qwen/Qwen2.5-0.5B-Instruct`) can be adapted on a **free-tier Colab T4 (15 GB)** to do this
rewrite reliably. The main safety requirement is that it must **never invent a street address**.

## 2. Engineering objective

1. Build a synthetic instruction dataset (~400 complaint → ticket pairs).
2. Instruction-tune with **QLoRA** (NF4 4-bit frozen base + LoRA adapters), with loss only on
   assistant tokens.
3. Build a synthetic preference dataset (~150 chosen/rejected pairs) aimed at
   address fabrication.
4. Apply a short **DPO** pass on top of the instruction-tuned adapter.
5. Evaluate **base vs. SFT vs. DPO** on the same held-out complaints.
6. Measure peak GPU memory, wall-clock time and step time for every run, and log them in
   [logs/experiment_log.csv](logs/experiment_log.csv).
7. Run a controlled memory stress test and record whatever actually happens.

---

## 3. CivicDesk behavior specification

### 3.1 Output contract

The assistant outputs **one JSON object and nothing else** (no prose, no Markdown fences).
The keys appear in this **fixed order**:

```json
{
  "category": "<one category slug from §3.2>",
  "urgency": "<emergency | high | medium | low>",
  "location": "<most specific location the resident gave, or \"unspecified\">",
  "summary": "<one or two neutral sentences describing the issue>",
  "address_or_null": "<house number + street, or null>"
}
```

| Field | Type | Rule |
|---|---|---|
| `category` | string (enum) | Exactly one slug from §3.2. For a multi-issue complaint, use the **primary** issue (the one with the highest urgency; if tied, the first one mentioned). Secondary issues are mentioned in `summary`. |
| `urgency` | string (enum) | Exactly one level from §3.3. |
| `location` | string | The most specific location **stated by the resident**: address, intersection, landmark, or partial description, with light cleanup (casing, obvious typos). If the resident gives no location at all: `"unspecified"`. |
| `summary` | string | A neutral restatement of the issue in 1–2 sentences, **using only facts present in the complaint**. A secondary issue is added as *"Also reports …"*. The sentence *"Location clarification required."* is added **only** when a dispatcher could not reasonably locate the issue (§3.5). |
| `address_or_null` | string or `null` | **Only a house number plus a street name** (for example `"1423 Elm St"`) and only if the resident wrote one. Otherwise JSON `null`. |

### 3.2 What counts as an address (decision C)

`address_or_null` is non-null **only** when the complaint contains a house/building number
**and** a street name.

| Resident wrote | `location` | `address_or_null` | Clarification? |
|---|---|---|---|
| "in front of 1423 elm st" | `"1423 Elm St"` | `"1423 Elm St"` | no |
| "corner of Oak and 5th" | `"Oak and 5th (intersection)"` | `null` | no (actionable) |
| "behind Riverside Library" | `"behind Riverside Library"` | `null` | no (actionable) |
| "the 400 block of pine st" | `"400 block of Pine St"` | `null` | no (actionable) |
| "somewhere on Maple Ave" (one pothole) | `"Maple Ave (no house number given)"` | `null` | **yes** (point issue on a whole street) |
| "on Maple Ave" (street never plowed) | `"Maple Ave (no house number given)"` | `null` | no (the issue covers the whole street) |
| "by my place" | `"near resident's home (exact location unclear)"` | `null` | **yes** |
| *(no location at all)* | `"unspecified"` | `null` | **yes** |

Intersections, landmarks, block ranges ("the 400 block of Pine") and street-only mentions
**are not exact addresses**. The address is copied from the complaint with casing normalized.
It is never completed, guessed or geocoded: no invented house numbers, suffixes, ZIP codes or
cities.

### 3.3 Categories

| Slug | Covers |
|---|---|
| `pothole_road_damage` | Potholes, cracked or collapsed pavement, sinkholes in the roadway |
| `streetlight` | Streetlights that are out, flickering, damaged or on during the day |
| `garbage_illegal_dumping` | Missed collection, overflowing public bins, dumped furniture/debris |
| `graffiti` | Graffiti or vandalism markings on public or visible property |
| `water_sewer` | Water main breaks, leaks, low pressure, sewer backups, clogged storm drains, flooding |
| `tree_vegetation` | Fallen or dangerous trees and limbs, overgrown vegetation blocking a path or view |
| `sidewalk_damage` | Broken, heaved or missing sidewalk; curb ramp damage |
| `snow_ice` | Unplowed streets, icy sidewalks or crossings, snow blocking access |
| `noise` | Persistent noise: construction outside permitted hours, parties, alarms, equipment |
| `abandoned_vehicle` | A vehicle apparently abandoned on a public street |
| `traffic_signage` | Signal malfunction, missing or damaged signs, faded markings |
| `other` | A valid municipal request that doesn't fit the categories above |

### 3.4 Urgency taxonomy (decision D)

The model picks the **highest** level whose criteria the complaint's stated facts support.
Urgency is never raised based on things the resident did not say.

| Level | Criteria | Examples |
|---|---|---|
| `emergency` | **Immediate** danger to life or safety, or active property damage happening now. | Water main break flooding a street; traffic signal completely dark at a busy intersection; tree or live-looking wire down across a road; open sinkhole in a travel lane |
| `high` | A **hazard likely to cause injury or damage soon** (≈24 h) if not addressed, or loss of an essential service. | Deep pothole in a busy lane; stop sign knocked down; sewer backing up into a home; ice sheet on a school crossing |
| `medium` | A **quality-of-life or service problem** that needs action within days but has no immediate hazard. | Streetlight out on a residential block; overflowing public bin; persistent late-night construction noise; broken sidewalk slab |
| `low` | **Cosmetic or non-time-sensitive** issues. | Graffiti; a car parked unmoved for weeks and not blocking anything; overgrown hedge |

### 3.5 Non-fabrication rules

1. Never invent a street address, house number, cross street, landmark, date, time, vehicle
   plate, or person's name.
2. **Clarification is about actionability, not about the address.** A `null` address does
   **not** by itself trigger "Location clarification required." Intersections, named
   landmarks and block ranges are actionable. A street name alone is actionable only for a
   street-wide issue (unplowed or unsalted street, whole-street missed collection, faded lane
   lines, racing along the street). Clarification is required for vague locations, missing
   locations, and point issues given only a street name. The model must not learn
   `address_or_null == null → ask for clarification`.
   A vague location is recorded as the resident described it, never resolved.
3. Ignore irrelevant detail (personal anecdotes, complaints about the city in general) unless it
   changes the category or urgency.
4. Being over-conservative is also a failure: if a valid house number + street **is** present,
   `address_or_null` must contain it. Evaluation measures both failure directions (§9).

---

## 4. Repository structure

```
week03_adapter_sprint/
├── README.md                      this document
├── requirements.txt               pinned dependency stack (see §6)
├── .gitignore                     excludes credentials, model weights, outputs/
├── configs/adapter_config.json    baseline hyperparameters (CLI-overridable)
├── data/                          JSONL datasets, written ONLY by scripts/          [Step 2, 4]
├── scripts/
│   ├── generate_instruction_data.py                                               [Step 2]
│   ├── audit_instruction_data.py                                                  [Step 2]
│   ├── generate_preferences.py    includes preference-shortcut audit              [Step 4]
│   └── evaluate_models.py         base vs SFT vs DPO                              [Step 6]
├── training/
│   ├── common.py                  system prompt, chat formatting, loss masking,
│   │                              CUDA memory/time instrumentation, CSV logger    [Step 3]
│   ├── train_qlora.py                                                             [Step 3]
│   └── train_dpo.py                                                               [Step 5]
├── logs/experiment_log.csv        append-only run log (header only until real runs)
└── notebooks/week03_adapter_sprint.ipynb   Colab execution path                   [Step 7]
```

Adapters and checkpoints are written to `outputs/`, which is git-ignored.

---

## 5. Base model

| Property | Value |
|---|---|
| Model ID | `Qwen/Qwen2.5-0.5B-Instruct` |
| Parameters | ~0.49 B, including a tied input/output embedding of about 136 M (vocab 151,936). The script prints the exact count. |
| Architecture | `Qwen2ForCausalLM`: decoder-only, grouped-query attention, RoPE, SwiGLU MLP, RMSNorm |
| License / access | Apache-2.0, **not gated**, no HF token required. The notebook checks this instead of assuming it. |
| Chat template | ChatML: `<\|im_start\|>{role}\n{content}<\|im_end\|>`. If no system message is supplied, the template injects a default Qwen system prompt, so we **always** pass the CivicDesk system prompt. |
| Expected projection names | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. **These are expectations only;** the training script prints the real module names before building the LoRA config. |

**Why this model:** it follows chat instructions well enough to give a meaningful *base*
comparison (decision A: the base evaluation is this Instruct checkpoint with no adapters). It
is small enough that a T4 can iterate quickly, and it is ungated. Alternatives considered:
SmolLM2-360M-Instruct (weaker at structured output), Llama-3.2-1B (gated, more than 1B
parameters), Qwen3-0.6B (thinking-mode template adds `<think>` blocks and complicates masking).

### Why QLoRA, and an honest caveat

This project uses **QLoRA (NF4 4-bit base weights + LoRA)** to **demonstrate parameter-efficient
quantized adaptation under the assignment's memory constraint.**
**A 0.5B model does not inherently require 4-bit quantization to fit on a T4**: its fp16
weights are only about 1 GB. The memory we measure will mostly reflect activations and
logits, not weights. With a vocabulary of 151,936, the fp32 logits alone for one 512-token
sequence are about 0.3 GB. Section 8 interprets the results with that in mind.

### Quantization choice

- **NF4** (4-bit NormalFloat): the QLoRA data type, designed for normally distributed weights.
- **Double quantization:** also quantizes the quantization constants. The saving is small at
  0.5B, but it is part of the standard QLoRA recipe.
- **Compute dtype `float16`:** the T4 (compute capability 7.5) **does not support bf16**.
  Qwen2.5 was trained in bf16, so fp16 overflow (NaN/inf loss) is a known risk. Training logs
  loss every few steps and reports non-finite loss instead of hiding it.

### LoRA configuration rationale

| Setting | Baseline | Rationale |
|---|---|---|
| Rank `r` | 16 | Plenty for a narrow formatting task and a common starting point. It is not tuned. |
| Alpha | 32 | α/r = 2, a common LoRA scaling. |
| Dropout | 0.05 | Light regularization; the dataset is small (~340 training examples). |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` (decision F) | Attention projections only for the baseline. They are **checked against the loaded model** and the run fails if any are missing. Adding the MLP projections would be a separate single-variable experiment. |
| Trainable-parameter check | required | The script prints total, trainable and percentage, and **stops with an error if the trainable count is 0**. |

---

## 6. Dependency strategy

- The HF stack is **pinned exactly** in [requirements.txt](requirements.txt):
  transformers 5.17.0, peft 0.21.0, trl 1.14.0, bitsandbytes 0.50.2, accelerate 1.15.0,
  datasets 5.0.1. Their declared dependency constraints were checked, and pip's resolver
  confirmed the set is conflict-free on Linux x86_64 / Python 3.12 on 2026-09-27. 3.12 was
  the *assumed* Colab Python at that time. Training code is written against **these** APIs.
- **torch is not pinned exactly.** Colab's preinstalled CUDA build is used, subject to
  `torch>=2.4,<3` (the bitsandbytes requirement). The notebook prints the torch and CUDA
  versions and the GPU's compute capability, and each log row records the package versions.
- **Measured environments. Keep these distinct:**

  | Environment | Python | torch | GPU | Used for |
  |---|---|---|---|---|
  | Colab (measured, T4 smoke run) | **3.13.15** | **2.11.0+cu128** | Tesla T4, compute 7.5, 15360 MiB | all GPU runs |
  | Local development (macOS, CPU) | 3.11.15 | 2.14.0 (no CUDA) | none | CPU validation only |
  | Resolver pre-check | 3.12 (assumed) | n/a | n/a | install-compatibility check |

  From Step 5A, every experiment-log row also records `python_version`, `torch_cuda_build`,
  `cudnn_version`, `gpu_capability`, `gpu_total_memory_gb` and `cuda_alloc_conf`. The
  earlier smoke row was not rewritten, so those fields are empty for it. Its Python and CUDA
  values above come from the run's notebook output.
- On Colab, `pip check` reports that `ipython 7.34.0 requires jedi`. That comes from Colab's
  own IPython, and `jedi` is not in this project's dependency tree. It needs no action.
- The resolver only proves the packages can be *installed* together. Runtime compatibility
  (the 4-bit T4 kernels and TRL's DPO adapter-reference path) is checked in the notebook
  before any training.

---

## 7. Method details (filled in as each step is built)

### 7.1 Instruction dataset design (Step 2)

**Generation.** [scripts/generate_instruction_data.py](scripts/generate_instruction_data.py) is
deterministic Python (seed 42) with no LLM API. It builds each complaint from:

- **84 scenario families** (7 per category). A family is one underlying issue, such as "deep
  pothole", with 2–3 paraphrased issue clauses and a short terse form.
- **Criteria-driven urgency.** Each issue and each optional context clause carries §3.4
  criteria *facts* (`COSMETIC`, `QUALITY_OF_LIFE`, `HAZARD_SOON`, `SERVICE_LOSS`,
  `IMMEDIATE_DANGER`, `ACTIVE_DAMAGE`). The label is the highest-ranked fact present. The same
  family therefore appears at several urgency levels ("pothole" is medium; "pothole + cars
  swerving into traffic" is high). Context clauses are restricted to issues where they make
  causal sense (a hairline crack is never escalated by "lots of foot traffic").
- **Decoupled resident urgency claims** ("URGENT!!", "no rush", "not a big deal") are
  injected independently of the label, so urgency cannot be solved by keyword lookup.
- **Seven location conditions**, each with its own rendering and normalization (§3.2).
- **Surface variety:** styles (standard, terse, verbose, typo-heavy, slang, formal,
  shouting), 18 sentence frames, greetings and sign-offs, irrelevant detail, typos, dropped
  apostrophes and punctuation, and casing changes.
- **Multi-issue complaints** (about 9%): a second issue from another category. The ticket's
  category is the more urgent issue (ties go to the first one mentioned), and the other issue
  appears as "Also reports …".
- **Location protection.** Typo and slang noise never touches location text; only its casing
  changes. The target address is therefore always a verbatim, casing-normalized copy of what
  the resident typed. It is never repaired or completed.

Each JSONL record stores the complaint, the target `response` (a JSON string in the fixed key
order), and audit metadata: `family_id`, `location_type`, `needs_clarification`, `context`,
`style`, `frame`, and `noise`. The chat template and system prompt are applied later, in
`training/common.py`.

**Split procedure (deterministic, stratified, leakage-free):**

1. For each category, shuffle its 7 families with `random.Random(42)`. **2 families per
   category go entirely to eval** (24 eval families) and the other 5 to train (60 families).
   No underlying issue appears on both sides. Secondary issues in multi-issue complaints are
   drawn only from the same side's families.
   **Street-wide constraint (step 1b):** eval must contain at least one *street-wide* family
   (`eval_street_wide_families_min = 1`). The seeded shuffle placed none there. So, in the
   first category (list order) that has a street-wide train family, that family was swapped
   with the category's last-drawn eval family. This is deterministic and uses no extra random
   draws. The result:
   `garbage_illegal_dumping/01` ("recycling skipped for the whole street") moved to eval and
   `garbage_illegal_dumping/02` moved to train. The other 23 eval families are unchanged.
2. Eval gets 60 examples (5 per category) and train gets 340.
3. No-address conditions are assigned by stratified quota: **4 per condition in eval (24)**
   and 6 per condition in train (36). That is 60/400 = 15.0% overall, but 40% of eval, so the
   primary safety metric is not computed on about 9 random cases.
   **Street-only quota (step 3b):** at least 2 `street_only` records per split must come from
   street-wide families (actionable, no clarification), and at least one must stay on a
   point-issue family (clarification required). A deterministic swap of location types
   enforces this without changing per-condition counts. Result: eval has **2 actionable + 2
   clarification**, train has 2 + 4. This keeps `street_only` from implying either outcome.
   Street-wide issues are phrased as covering the street ("on X", "all along X", "up and down
   X"). Point issues never use "along X", because that reads as spread out.
4. The audit verifies: no family overlap; no eval issue paraphrase appearing verbatim in
   train; no exact duplicate complaints; eval-to-train near-duplicate similarity below 0.70
   (char-4gram Jaccard); and byte-identical regeneration under a different `PYTHONHASHSEED`.

The eval set **deliberately over-represents** no-address cases relative to train. That is
intentional for measuring safety, and it means aggregate eval accuracy should not be read as
an estimate of production accuracy.

**Audit.** Run `python scripts/audit_instruction_data.py` (exit code 1 on any hard failure).

**Dataset status: FROZEN** (after the Step 2 data-quality corrections). The committed
`data/instruction_*.jsonl` files are the instruction dataset for all experiments. They change
only if a genuine defect is found, and any such change is committed separately with its reason.

| Topic | Status |
|---|---|
### 7.2 Training infrastructure, chat template and loss masking (Step 3)

**Files.**
- [training/common.py](training/common.py) is the single source of truth for SFT, DPO and
  evaluation. It holds the system prompt, formatting, masking, instrumentation and the logger.
- [training/train_qlora.py](training/train_qlora.py) is the SFT entry point.
- [training/check_masking.py](training/check_masking.py) prints the masking diagnostic.
- [training/validate_local.py](training/validate_local.py) runs the CPU-only checks.

**System prompt.** The same `SYSTEM_PROMPT` is used for training and for all three evaluated
models. It states the output contract, the categories, the urgency criteria, the clarification
rule and the address rule. It is **312 tokens** and is masked in every example.

**Assistant-only masking (decision E).** The Qwen2.5 chat template has **no
`{% generation %}` tag**. The built-in `return_assistant_tokens_mask=True` silently returns
an **all-zero mask** (verified), so it cannot be used. Instead, for each example:

1. Render system + user with `apply_chat_template(add_generation_prompt=True)`. This is the
   prompt, ending at `<|im_start|>assistant\n`.
2. Render the full system + user + assistant conversation.
3. Require the prompt ids to be an **exact prefix** of the full ids.
4. Labels are `-100` for every prompt position. The assistant JSON and its end-of-turn token
   `<|im_end|>` (the EOS the model must learn to emit) are trainable. The template's trailing
   `\n` after `<|im_end|>` is masked, because generation stops at EOS and never produces it.
5. Fail loudly if: the prompt is not a prefix; there is no end-of-turn; there are no trainable
   tokens; every token is trainable; the assistant region is only the EOS token; the decoded
   trainable region doesn't reproduce the target JSON; or the sequence exceeds
   `max_seq_length`. **Over-length examples are rejected, never truncated.**

Local verification: all 400 records pass. HF's loss equals a manual cross-entropy computed
over assistant tokens only. Gradients reach all 96 `lora_B` tensors and no base parameter.

**Measured sequence lengths** (real tokenizer, full conversation):

| Split | Total tokens min / median / p95 / max | Trainable tokens min / median / max | Loss-bearing share |
|---|---|---|---|
| train (340) | 364 / 396 / 425 / 457 | 39 / 52 / 70 | 13.1% |
| eval (60) | 367 / 400 / 427 / 430 | 43 / 53 / 66 | 13.2% |

The complaints themselves are short, but the fixed 312-token system prompt puts full
sequences within **55 tokens of the 512 ceiling**. Lowering `max_seq_length` below about 460
would reject examples. That limits what the memory-stress diagnosis can change.

**Architecture verification (measured locally, fp32 CPU, not quantized).** These are the
linear modules found in the loaded model. Targets are verified before the LoRA config is
built, and a missing target is a hard error with no substitution.

| Module | Count | (in, out) |
|---|---|---|
| q_proj | 24 | (896, 896) |
| k_proj | 24 | (896, 128) |
| v_proj | 24 | (896, 128) |
| o_proj | 24 | (896, 896) |
| gate_proj / up_proj | 24 each | (896, 4864) |
| down_proj | 24 | (4864, 896) |
| lm_head | 1 | (896, 151936), tied to the input embedding |

**Trainable parameters (measured).**
- The base model has 494,032,768 parameters.
- LoRA r=16 on q/k/v/o adds **2,162,688 trainable parameters** across 96 wrapped modules.
- That is 0.4359% of the post-LoRA total (0.4378% of the base).
- This matches the arithmetic prediction exactly: per layer, q and o each add 16 × (896 + 896)
  = 28,672, and k and v each add 16 × (896 + 128) = 16,384. That is 90,112 per layer, times 24
  layers.
- In the 4-bit CUDA path, the stored parameter count is lower because 4-bit weights are packed.
  The trainable count does not change.

**API corrections for the pinned stack** (read from the installed source, not assumed):

| Assumed | Actual in transformers 5.17.0 / peft 0.21.0 | Handling |
|---|---|---|
| `warmup_ratio=0.1` | argument **does not exist** | `warmup_steps=0.1` (a float in [0, 1) is a ratio) |
| `apply_chat_template(tokenize=True)` returns ids | returns a `BatchEncoding` dict (`return_dict=True` default) | read `["input_ids"]` |
| `return_assistant_tokens_mask` usable | template lacks `{% generation %}`, so the mask is all zeros | explicit prefix masking |
| NaN/inf losses visible in logs | `logging_nan_inf_filter=True` by default **hides them** | set to `False`, plus a per-step finiteness check that raises |
| OOM surfaces as an error | `auto_find_batch_size` can silently retry smaller batches | forced `False`; an OOM is logged and re-raised |
| `torch_dtype=` | deprecated | `dtype=` |
| `prepare_model_for_kbit_training` only freezes | it also upcasts **all** non-4-bit fp16/bf16 params (embeddings, norms, tied lm_head) to fp32, and enables checkpointing | expected, and counted in the memory interpretation |

**Instrumentation.**
- Peak allocated and reserved memory are measured with `torch.cuda.max_memory_*`, starting
  **just before model load**. `peak_allocated_gb` and `peak_reserved_gb` are the maxima over
  the whole run, covering loading, training and evaluation.
- **Per-phase peaks** (from Step 5A) split this into `load`, `train` and `eval`. The peak
  counters are reset at each phase boundary; nothing changes how memory is allocated.
- **Allocator counters** (from Step 5A):
  - `alloc_retries`: how many times a `cudaMalloc` failed and the caching allocator released
    cached blocks and retried. This means the cache reached the device limit; it is not an OOM.
  - `cuda_ooms`
  - `peak_inactive_split_gb`: cached but unused space inside split blocks.
  - `peak_segments`
- **Allocated vs reserved.** *Allocated* is live tensor memory, the model's actual demand.
  *Reserved* is what PyTorch's caching allocator holds from the device, which includes freed
  blocks it keeps for reuse. Reserved is therefore a cache high-water mark, not demand.
  Neither includes the CUDA context (roughly 0.3–0.5 GB), which `nvidia-smi` does show.
- Wall-clock time covers `trainer.train()`, including the periodic evaluations. After
  training, one more evaluation scores the final weights; its duration is recorded separately
  (`final_eval_s` in `notes`).
- Average step time is the mean optimizer-step duration, CUDA-synchronized, excluding the first
  step and excluding evaluation.
- Every CUDA run, successful or failed, appends one row to `logs/experiment_log.csv`.
- Without CUDA, memory fields are recorded as `NA_no_cuda`, never as numbers.

### 7.3 DPO preference dataset (Step 4)

**Files.**
- [scripts/generate_preferences.py](scripts/generate_preferences.py) is the deterministic
  generator (seed 42).
- [scripts/audit_preference_data.py](scripts/audit_preference_data.py) is the audit and
  shortcut diagnostic.
- [data/preference_train.jsonl](data/preference_train.jsonl) holds the 150 pairs.

**Construction.**
- Each pair is one resident complaint, a **chosen** ticket and a **rejected** ticket.
- Chosen is the canonical contract-correct ticket, produced by the same rendering code as the
  frozen instruction dataset.
- Complaints are rendered only from the **60 training scenario families**. None of the 24
  held-out eval families is used. No prompt duplicates an instruction complaint, and the
  closest preference prompt to any eval complaint has char-4gram Jaccard similarity 0.404.
- **Category, urgency and summary are identical** between chosen and rejected in every pair.
  The pairs differ **only** in address handling:

| Pair type | n | Complaint | Chosen | Rejected | Fields that differ |
|---|---|---|---|---|---|
| `fabricated_address` | 36 | no exact address | `address_or_null: null` | a plausible invented address | `address_or_null` |
| `fabricated_address_and_location` | 24 | no exact address | null + supplied location | the invented address in both fields | `address_or_null`, `location` |
| `over_null` (control) | 60 | exact address | the resident's address | `null` | `address_or_null` |
| `altered_address` (control) | 30 | exact address | the resident's address | the house number changed (same length) | `address_or_null`, `location` |

The 60 no-address pairs cover 10 of each no-address condition: intersection, landmark, block
range, street only (5 street-wide/actionable, 5 point/clarification), vague and missing.

**How rejected addresses are fabricated.** They are built the way a model would hallucinate
them:
- a house number attached to a street the resident mentioned
- a number inside a stated block range ("1200 block of Highland Avenue" → "1277 Highland")
- an invented street when nothing usable was supplied

Fabricated addresses use the same house-number and street-suffix distributions as real
resident addresses, including no suffix, so they cannot be spotted by their form.

**Why the control pairs are necessary.** If every pair rewarded `null` over an address, DPO
could lower its loss by learning "never output an address". That is the over-conservative
failure the evaluation explicitly checks for. The `over_null` controls reward keeping a real
address. The `altered_address` controls reward copying it exactly rather than producing any
plausible number. Together they make the preference decidable only by checking the response
against the complaint.

**Null balance: 60 / 60.** `null` appears in exactly 60 chosen responses (types A+B) and 60
rejected responses (type C). The fixed rule "the response with a null address is chosen"
therefore scores exactly 0.500.

**Length balance** (rejected minus chosen):

| Unit | Mean difference | Rejected longer / shorter / equal |
|---|---|---|
| characters | −2.97 (ratio range 0.88–1.09; 0 pairs beyond ±15%) | 44 / 76 / 30 |
| words | −0.31 | 42 / 72 / 36 |
| Qwen tokens | −0.15 | 59 / 60 / 31 |

**Shortcut diagnostic.** The diagnostic asks whether chosen can be told apart from rejected
using the *response alone*, without reading the complaint. It uses 10 surface features
(character, word and token length; digit count; null flag; location form; suffix;
clarification sentence) and three kinds of classifier:
- pointwise single-feature stumps and logistic regression
- pairwise-difference stumps and logistic regression, each with 5-fold cross-validation grouped
  by pair
- fixed pairwise rules

**Thresholds were set before any data was generated: warn ≥ 0.60, stop ≥ 0.70.**

| Warning | Accuracy | Status |
|---|---|---|
| pointwise digit-count stump | **0.663** (highest observed) | warning, below the 0.70 stop threshold |
| pairwise character-length-difference stump | 0.620 | warning |
| pairwise logistic regression on feature differences | 0.620 | warning |
| fixed rule "longer response (chars) is chosen" | 0.607 | warning |

All other response-only classifiers and rules score 0.39–0.58, including the all-feature
pointwise logistic regression at 0.560. The lowest is "shorter response (chars) is chosen" at
0.393, the mirror image of the 0.607 rule. For contrast, the rule that reads the complaint ("choose the
response whose `address_or_null` is supported by the complaint") scores 150/150.

**Known risks, accepted and documented (not reworked):**

1. **Digit-count signal (0.663).** Every rejected response contains digits, because every error
   type places an address somewhere, so 0 of 150 rejected responses are digit-free. 49 of 150
   chosen responses are digit-free: the 60 no-address chosen responses minus 11 whose location
   contains a number, such as a block range or "Fire Station 7".
   This is **below the predefined 0.70 stop threshold but remains a potential spurious
   preference signal.** It could nudge DPO toward digit-free (null-address, non-numeric
   location) outputs. Within a pair, "fewer digits is chosen" is right exactly as often as it
   is wrong (60 / 60 with 30 ties), and the controls push against the drift. Neither fact
   removes the risk. The held-out over-conservative null rate (§9) is the measurement that
   will show whether it materialized.
2. **Character-length signal (0.607–0.620).** Controls where rejected is `null` make the chosen
   response longer in characters. Token-level length is balanced (59 / 60), and DPO
   likelihoods are sums over tokens. **That does not show the risk is eliminated:** a
   character-level difference still correlates with *which content* is present (an address
   string versus `null`). The risk is recorded, not dismissed.

**No held-out preference split.** All 150 pairs train DPO. Any DPO reward accuracy or reward
margin computed on these pairs is **training-set reward accuracy**. It says how well the
objective was fitted, not how well the model generalizes, and it is never reported as held-out
accuracy. Generalization is assessed only by the held-out three-way (base vs SFT vs DPO)
behavioral evaluation on the frozen eval set (§9).

### 7.4 DPO training path (Step 5B)

**Files.**
- [training/train_dpo.py](training/train_dpo.py) is the entry point.
- [training/validate_dpo_local.py](training/validate_dpo_local.py) holds the CPU checks.
- [notebooks/week03_dpo.ipynb](notebooks/week03_dpo.ipynb) is the Colab path.

**Starting point.** DPO starts from the baseline SFT adapter of run
`qlora_sft-baseline-20260927T235802Z`, with adapter SHA-256
`53aa9a93d6ef1011a3d9395283c62db51d6f327f4671cc3a59c9bf3d3dc85b5a`. The run stops unless
the restored file hashes to that value **and** the value equals the `adapter_sha256` of that
completed baseline row in the experiment log. The notebook restores the adapter from Google
Drive, and a smoke adapter is refused.

**Architecture** (TRL 1.14.0 / peft 0.21.0, verified by reading the installed source and by CPU
tests):

| Component | Implementation |
|---|---|
| Base | One NF4 4-bit base, loaded and prepared exactly as in SFT |
| Policy | The SFT adapter, loaded **trainable** (`is_trainable=True`) as adapter `default` |
| Reference | TRL's **built-in frozen copy** of the SFT adapter, adapter `ref`, on the same base. DPOTrainer creates it when given a `PeftModel` and `ref_model=None`. Reference log-probs are computed under `no_grad` with `ref` active. |
| Not used | a merged model; the adapter-disabled base as reference; a second base model |

**API findings that changed the Step 1 plan.**
- **The config fields don't exist.** The plan assumed `DPOConfig(model_adapter_name, ref_adapter_name)`. Those fields do **not** exist in TRL 1.14, and neither does `max_prompt_length`. The dual-adapter design is still achieved, through TRL's own `ref` copy.
- **TRL casts the trainable adapter of a quantized model to bf16.** This follows the QLoRA
  paper. On our T4/fp16 setup it would:
  - make the starting policy differ from the verified SFT adapter (CPU simulation: up to
    1.2e-4 difference, with non-zero step-0 rewards)
  - train bf16 master weights at LR 5e-5
  - depart from the SFT regime of fp32 adapters under fp16 autocast
  
  `policy_adapter_dtype = fp32` (default) restores the policy from the fp32 `ref` copy after
  TRL initialization and before the optimizer exists. It then verifies **policy == ref ==
  adapter file bit-for-bit**. `--policy-adapter-dtype trl_default` keeps TRL's behavior.
- **TRL's own defaults are unsuitable here** (`bf16=True`, `learning_rate=1e-6`,
  `logging_nan_inf_filter=True`, fused AdamW, `max_length=1024`, `max_grad_norm=1.0`). All of
  them are set explicitly: fp16, 5e-5, NaN visible, `adamw_torch`, 512, 0.3.
- **TRL names its evaluation reward metrics `eval_rewards/*`** regardless of
  `metric_key_prefix`. This was caught on CPU and is handled in `dpo_eval_metrics`, which fails
  loudly if a key is missing.
- **TRL only *warns* on a prompt/completion token-prefix mismatch.** It also truncates silently
  at `max_length` with `keep_start`. A pre-check on all 300 sequences **fails** instead. All
  pairs pass: 360–439 tokens, well under 512.
- **Dropout is disabled** (`disable_dropout=True`), so SFT's LoRA dropout of 0.05 is off during
  DPO.
- **The template's trailing `\n` after `<|im_end|>` is part of both completions.** SFT masked
  it. It is identical in chosen and rejected.

**Step-0 reference check.** Before training, one evaluation pass over the 150 pairs must give
rewards of about 0 (policy == reference). On CPU this gave exactly 0.0, with a loss of exactly
ln 2 = 0.693147.

**Plan.**
- β = 0.1, sigmoid loss, LR 5e-5, cosine schedule, warmup ratio 0.1, 1 epoch.
- 2 pairs per micro-batch × 8 accumulation = 16 pairs per step.
- **10 optimizer steps**: 75 micro-batches = 9 × 8 + 3, and the last step uses 6 pairs.
  Warmup is 1 step.
- Loss is logged every step.

**Metrics.**
- Per-step: loss, `rewards/chosen`, `rewards/rejected`, `rewards/margins`, `rewards/accuracies`.
- After training, one full pass over the same 150 pairs populates `train_set_reward_accuracy`,
  `train_set_reward_margin`, `train_set_chosen_reward` and `train_set_rejected_reward`. These
  are **training-set** metrics.
- The log also records `sft_adapter_sha256`, `preference_data_sha256`, `policy_adapter_dtype`
  and `ref_check_max_abs_reward`. The DPO adapter is saved with `selected_adapters=["default"]`,
  so the `ref` copy is never saved.

**Interpretation limit.** A rising training-set reward accuracy is **not** evidence of better
address grounding. DPO could exploit the documented 0.663 response-only digit-count shortcut
(§7.3). Only the held-out three-way evaluation measures fabrication and over-conservative null
behavior.

| Topic | Status |
|---|---|
| Instruction dataset design + audit | done (see 7.1) |
| Chat template + assistant-only loss masking | done, verified locally (see 7.2) |
| Memory/time instrumentation | measured in the smoke and full SFT runs (`logs/experiment_log.csv`) |
| Preference dataset design + shortcut audit | done (see 7.3) |
| DPO training path | implemented, CPU-validated; **not yet run** (see 7.4) |
| Evaluation methodology | *Step 6* |

---

## 8. Results — **to be populated only from real Colab runs**

### 8.1 Run summary (from `logs/experiment_log.csv`)

| Run ID | Method | Seq len | Micro-batch | Grad accum | Peak alloc (GiB) | Peak reserved (GiB) | Wall clock (s) | Avg step (s) | Status |
|---|---|---|---|---|---|---|---|---|---|
| qlora_sft_smoke-smoke-20260927T220352Z | qlora_sft_smoke (5 steps) | 512 | 4 | 4 | 4.122 | 13.221 | 14.75 | 2.1643 | completed |
| qlora_sft-baseline-20260927T235802Z | qlora_sft (full, 66 steps) | 512 | 4 | 4 | 4.244 | 14.256 | 166.49 | 2.1919 | completed |
| *DPO: pending* | | | | | | | | | |

The smoke row is a 5-step execution check, **not** an SFT result. Its losses (train 0.6475,
eval 0.9421) describe 5 steps only.

**Smoke-run allocated/reserved gap (4.122 vs 13.221 GiB): diagnosis, not yet confirmed.**
- **Allocated (4.1 GiB) is mostly transient vocabulary logits.** With 151,936 vocabulary
  entries, one micro-batch of 4 × ~430–457 tokens produces fp16 logits of about 0.5 GiB. The
  loss upcasts them to fp32 (`logits.float()`, about 1.1 GiB), cross-entropy's log-softmax
  adds about 1.1 GiB, and the backward pass adds gradients of similar size. On top of that sit
  the resident model: roughly 0.2 GiB of 4-bit weights plus 0.54 GiB of fp32 embedding/lm_head
  after `prepare_model_for_kbit_training`. This estimate lands close to 4.1 GiB.
- **Reserved (13.2 GiB) most likely reflects caching-allocator retention.** Each micro-batch
  is padded to its own maximum length, so the ~0.5–1.1 GiB logits tensors have a different
  size almost every forward pass across 20 training micro-batches and 15 evaluation batches.
  Freed blocks stay cached, and a larger request can't reuse a smaller cached block. The cache
  can therefore grow toward the device limit, where the allocator frees cached blocks and
  retries.
- **This is not an OOM.** The run completed. There is also no evidence of a leak, but none
  against one either: the smoke run did not measure allocated memory at the end of the run.
- **The smoke row cannot confirm this.** It has no per-phase peaks or allocator counters. The
  Step 5A instrumentation (`peak_*_{load,train,eval}_gb`, `alloc_retries`,
  `peak_inactive_split_gb`) is designed to confirm or refute it in the full run.
- **Implication for the 15 GB ceiling claim.** Allocated (demand) and reserved (cache) are
  reported separately. A reserved value near the device limit together with `alloc_retries > 0`
  indicates cache pressure, not demand exceeding 15 GB.

### 8.2 Controlled stress test and failure diagnosis

*Pending. Whatever genuinely happens will be recorded here, including the case where no OOM
occurs within the 15 GB ceiling.*

### 8.3 DPO metrics

| Metric | Value |
|---|---|
| Loss | *pending* |
| Chosen reward | *pending* |
| Rejected reward | *pending* |
| Reward margin | *pending* |
| Reward accuracy | *pending* |

---

## 9. Evaluation — **to be populated only from real outputs**

The same held-out complaints are scored for all three models with greedy decoding.

| Metric | Base | SFT (QLoRA) | DPO |
|---|---|---|---|
| Valid JSON ticket (fixed keys) | | | |
| Required-field completeness | | | |
| Category accuracy | | | |
| Address preserved when present | | | |
| Correct `null` when absent | | | |
| **Unsupported address rate** | | | |
| Over-conservative null rate (null when an address exists) | | | |

**Primary safety metric:**
`unsupported_address_rate = (# no-address complaints given a non-null address) / (# no-address complaints evaluated)`

**Binding reporting rules (recorded at Step 2, before any model was run):**

1. **Category/urgency correlation is measured, not removed.** Under §3.4, some categories are
   almost always one urgency level (graffiti is mostly low; noise is never high). Urgency
   accuracy is reported **next to a majority-urgency-per-category baseline** fitted on train.
   A model gets no credit for urgency beyond what category alone predicts.
2. **Per-urgency metrics report sample counts.** Eval has few `low` (9) and **`emergency`
   (3)** cases, so every per-level number is shown as `correct / n`. No emergency-specific
   performance claim is made from n = 3. The eval set is deliberately **not** rebalanced to
   raise this count.
3. **The no-address subset is deliberately enriched in eval** (40% of eval vs 10.6% of train)
   to measure safety. Aggregate eval scores are not estimates of production accuracy.
4. **Complaints are short, but training sequences are not.** *(Corrected in Step 3 after
   measurement.)* Complaint text is short, but each training sequence also carries a 312-token
   system prompt, so real sequences are 364–457 tokens against the 512 ceiling. The
   representative workload is about 400 tokens per sequence. Any run with longer sequences, or
   with micro-batches far above baseline, is labeled a **controlled stress test**, not a
   representative workload.
5. **Summary quality is not a primary quantitative metric** and will not be claimed as one.
   Summaries are shown side by side for qualitative inspection only.

---

## 10. Reproducibility

**Local CPU validation** (no GPU; does not train or quantize):

```bash
python3.11 -m venv .venv          # local CPU validation used 3.11; the measured Colab runtime is 3.13
.venv/bin/pip install "torch>=2.4,<3" transformers==5.17.0 peft==0.21.0 accelerate==1.15.0
.venv/bin/python training/check_masking.py            # masking diagnostic on real records
.venv/bin/python training/train_qlora.py --dry-run    # config + data + masking, no model
.venv/bin/python training/train_qlora.py --inspect-cpu  # fp32 architecture + LoRA count
.venv/bin/python training/validate_local.py           # CPU checks, no training
```

**Colab smoke test (Step 3B).** [notebooks/week03_smoke_test.ipynb](notebooks/week03_smoke_test.ipynb)
runs a real 5-optimizer-step QLoRA check on a T4 before the full experiment. It runs
`training/preflight.py --require-cuda`, the data audit and the masking checks, then:

```bash
python training/train_qlora.py --run-label smoke --max-steps 5 --logging-steps 1 --eval-steps 5
```

Smoke runs are logged with `method=qlora_sft_smoke` and `run_label=smoke`, and they save to
`outputs/smoke_adapter/`. That is separate from the production SFT output, so a smoke
adapter can never be mistaken for the trained model. `--max-steps` requires a non-baseline
label, and `smoke` requires `--max-steps`.

*Full Colab instructions arrive with the notebook (Step 7).* In outline:

1. Colab → Runtime → Change runtime type → **T4 GPU**.
2. Clone the repo, `pip install -r requirements.txt`, restart the runtime.
3. Run the notebook top to bottom. Seeds are fixed (`42`) in `configs/adapter_config.json`.
4. Never commit HF tokens. `.gitignore` excludes `.env` and token files.
