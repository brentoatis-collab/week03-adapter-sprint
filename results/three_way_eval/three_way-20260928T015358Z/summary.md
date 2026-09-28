Population: 60 held-out complaints; 36 with an exact address, 24 without ({'intersection': 4, 'missing': 4, 'vague': 4, 'block_range': 4, 'street_only': 4, 'landmark': 4}).

| Metric | BASE | SFT | DPO |
|---|---|---|---|
| **A. Unsupported exact-address fabrication** (primary; no-address complaints) ↓ | 0/24 (0.0%) | 1/24 (4.2%) | 0/24 (0.0%) |
| **B. Address retention, exact** (exact-address complaints) ↑ | 0/36 (0.0%) | 34/36 (94.4%) | 34/36 (94.4%) |
| **C. Over-null** (null despite exact address) ↓ | 35/36 (97.2%) | 0/36 (0.0%) | 0/36 (0.0%) |
| Invalid non-null address field (contract; any non-null on no-address complaints) ↓ | 0/24 (0.0%) | 22/24 (91.7%) | 24/24 (100.0%) |
| Retention incl. suffix variants | 0/36 (0.0%) | 34/36 (94.4%) | 34/36 (94.4%) |
| Correct null (no-address complaints) | 24/24 (100.0%) | 2/24 (8.3%) | 0/24 (0.0%) |
| Contract-compliant JSON (strict) | 39/60 (65.0%) | 58/60 (96.7%) | 58/60 (96.7%) |
| Category correct | 5/60 (8.3%) | 36/60 (60.0%) | 38/60 (63.3%) |
| Urgency correct | 14/60 (23.3%) | 26/60 (43.3%) | 31/60 (51.7%) |
| Location exact | 22/60 (36.7%) | 38/60 (63.3%) | 37/60 (61.7%) |
| Location anchor (grounded content) | 49/60 (81.7%) | 52/60 (86.7%) | 51/60 (85.0%) |
| Clarification sentence correct | 49/60 (81.7%) | 50/60 (83.3%) | 48/60 (80.0%) |
| Summary contains unsupported address ↓ | 1/60 (1.7%) | 0/60 (0.0%) | 0/60 (0.0%) |
| Core-correct ticket | 0/60 (0.0%) | 11/60 (18.3%) | 13/60 (21.7%) |
| Full exact match | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) |
| Urgency baseline: majority per category (train-fitted) | 30/60 (50.0%) | — | — |

Read separately: fabrication (A), address-field contract (invalid non-null), interface contract (strict JSON), retention (B) and over-null (C) are distinct claims. Controlled model-state comparison on one NF4 base with identical inference settings; not a stock-inference benchmark. DPO training-set reward accuracy is not held-out evidence; the 0.663 response-only digit-count shortcut is a known risk.
