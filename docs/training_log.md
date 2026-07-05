# Training Log — Edible Classifier

Tracks every training run, dataset version, and key findings.
Primary safety metric: **toxic FP rate** (toxic species predicted as edible).
Production target: < 1%.

---

## Dataset Versions

### v1 — Initial scrape (no quality gates)
- ~12,000 images, Texas only (moonseed global)
- No blur filtering, no fruiting filter
- All 12 iNat taxon IDs were wrong (fixed in PR #4 via taxon_name, corrected IDs tracked in species.json)

### v2 — Quality-filtered rescrape (current)
- **8,204 images total** (ilex_decidua updated to 1,185 in v2.1)
- Texas scrape with `--fruiting-only` (fruits/seeds observations only) + blur gate (Laplacian variance ≥ 100)
- Moonseed scraped globally (only ~23 TX observations)
- Species thin after fruiting filter (moonseed: 119, solanum: 120, sambucus: 196) — supplemented to 600 each with blur-only (no fruiting filter)
- Final per-species counts:

| Species | Count | Notes |
|---|---|---|
| rubus_trivialis | 1,000 | |
| ilex_vomitoria | 1,000 | |
| callicarpa_americana | 1,000 | |
| ilex_decidua | 661 | → updated to 1,664 (v2.1) |
| solanum_nigrum | 600 | supplemented |
| sambucus_canadensis | 600 | supplemented |
| menispermum_canadense | 600 | global scrape, supplemented |
| mahonia_trifoliolata | 600 | supplemented |
| celtis_laevigata | 600 | supplemented |
| vitis_mustangensis | 535 | |
| phytolacca_americana | 530 | |
| melia_azedarach | 478 | |

### v2.1 — ilex_decidua blur-only supplement
- ilex_decidua expanded from 661 → **1,664 images** via `--place-id 18` blur-only (no fruiting filter); 1000 new DL, 14 rejected blurry
- Root cause: ilex_decidua was the top source of toxic→edible FPs in Run F (8 dangerous misclassifications)
- Fruiting-only supply exhausted for TX (only 661 images); blur-only added 1003 new images
- Total dataset now ~9,371 images

---

## Key Findings

### Dangerous FP root-cause analysis (7 FPs from Run A)
Analyzed 7 high-confidence toxic→edible misclassifications from the v1 model:
- **3 from blur** — out-of-focus images; blur gate now rejects these
- **3 from no fruit in frame** — stem/leaf-only images with no distinguishing features; fruiting filter addresses at data level; fruit-presence gate (Layer 1b) will address at inference
- **1 genuine confusion** — chinaberry (toxic) → agarita (edible) at 96.5%; agarita training data contained yellow-toned images similar to chinaberry berries

### Calibration findings (Run D)
- Model was overconfident (temperature T=1.54 needed to calibrate)
- Per-class thresholds needed for 5/6 edible classes: sambucus (θ=0.982), hackberry (θ=0.895), beautyberry (θ=0.854), blackberry (θ=0.649), mustang grape (θ=0.526)
- Mahonia needed no threshold (no toxic confusion on val set)

### ASL findings (Run E, γ-=4)
- ASL model was underconfident (T=0.65), suggesting γ-=4 is too aggressive
- Improved toxic FP vs weighted CE but edible FN spiked to 13.9% (unacceptable)
- γ-=2 is the next candidate

---

## Training Runs

All runs: EfficientNet-B0, AdamW, cosine annealing LR, early stop on val toxic FP (patience=7, min_delta=0.005). Dataset v2 unless noted.

### Run A — Weighted CE, dataset v1
- **Config:** toxic_mult=3×, lr=5e-4, patience=7
- **Stopped:** epoch 6, val toxic_fp=2.09%
- **Checkpoint:** `checkpoints/run_a/best_safety.pt`
- **Test (on v2 test set):** acc=87.4%, toxic_fp=4.40%, edible_fn=9.91%
- **Notes:** Baseline; dataset v1 (no quality gates). Moonseed: 95.2% recall (after global rescrape from 46→1,000 images).

### Run D — Weighted CE, dataset v2
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=WeightedCrossEntropy
- **Stopped:** epoch 9 (early stop), best at epoch 8, val toxic_fp=3.48%
- **Checkpoint:** `checkpoints/run_d/best_accuracy.pt` (epoch 8, val_acc=89.4%)
- **Test:** acc=91.0%, toxic_fp=3.77% (18/477), edible_fn=5.32%
- **Test + calibration:** acc=92.3%, toxic_fp=1.68% (8/477), edible_fn=5.32%, rejection=8.3%
- **Notes:** Quality-filtered dataset improved overall accuracy. Temperature T=1.54 (overconfident).

### Run E — ASL γ-=4, dataset v2
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=4, margin=0.05)
- **Stopped:** epoch 10 (early stop), best at epoch 3, val toxic_fp=2.39%
- **Checkpoint:** `checkpoints/run_e/best_safety.pt` (epoch 3, val_acc=84.6%)
- **Test:** acc=85.7%, toxic_fp=2.94% (14/477), edible_fn=13.94%
- **Test + calibration:** acc=86.6%, toxic_fp=1.05% (5/477), edible_fn=13.94%, rejection=5.6%
- **Notes:** ASL too aggressive. Edible FN rate unacceptably high. T=0.65 (underconfident — γ-=4 over-suppressed negatives).

### Run F — ASL γ-=2, dataset v2 ✅ new best
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=2, margin=0.05)
- **Stopped:** epoch 8 (early stop), best_accuracy at epoch 5, val_acc=87.3%
- **Checkpoint:** `checkpoints/run_f/best_accuracy.pt` (epoch 5, val_acc=87.3%)
- **Test:** acc=89.6%, toxic_fp=2.94% (14/477), edible_fn=7.71%
- **Test + calibration:** acc=90.9%, toxic_fp=1.05% (5/477), edible_fn=7.71%, rejection=6.7%
- **Notes:** T=0.97 — nearly perfect calibration out of the box. γ-=2 hits the sweet spot: matches Run E's toxic FP without the edible FN blowout. Edible FN (7.71%) is 2.4pp worse than Run D but 6.2pp better than Run E. New best model.

### Run G — ASL γ-=4, CutMix (prob=0.5, α=1.0), dataset v2
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=4, margin=0.05), cutmix_prob=0.5, cutmix_alpha=1.0
- **Best_safety checkpoint:** epoch 4 (see run g calibration)
- **Test + calibration (best_safety):** acc=?, toxic_fp=1.28% (calibrated)
- **Notes:** CutMix alone vs Run F. Worse than Run F on safety metric. Run F still best.

### Run H — ASL γ-=4, CutMix + balanced sampling + label smoothing, dataset v2
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=4, margin=0.05), cutmix_prob=0.5, cutmix_alpha=1.0, balanced_sampling=True, label_smoothing=0.1
- **Stopped:** epoch 11 (patience=7), best_safety at epoch 4 (val_acc=0.815, val_toxic_fp=2.83%)
- **Best_safety calibration:** T=0.778, toxic_fp=2.73% (13/476) ← worse than Run F/G
- **Best_accuracy calibration:** epoch 10 (val_acc=0.889), toxic_fp=1.89% (9/476) ← better than best_safety but worse than Run F
- **Notes:** All three regularization techniques together (CutMix + balanced sampling + label smoothing) add too much regularization for this dataset size. Early stopping hit epoch 4 for safety checkpoint — model hadn't converged. γ-=4 already aggressive; adding more regularization on top made it worse.

### Run I — ASL γ-=2 + label smoothing ε=0.1, dataset v2 ✅ new best safety
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=2, margin=0.05), label_smoothing=0.1
- **Stopped:** epoch 8 (patience=7), best_safety at epoch 1, best_accuracy at epoch 8
- **best_safety:** epoch 1, val_acc=80.7%, val_toxic_fp=2.17%
  - Calibration (v2.1 dataset): T=0.707, toxic_fp=**0.18%** (1/544), acc=78.8%, rejection=10.2%
  - Calibration saved → `checkpoints/run_i/calibration.json`
- **best_accuracy:** epoch 8, val_acc=89.2%, val_toxic_fp=3.91%
  - Calibration (v2.1 dataset): T=1.100, toxic_fp=1.82% (10/548), acc=88.7%, rejection=7.0%
- **Notes:** Label smoothing (ε=0.1) on top of ASL γ-=2 is the winning combination. best_safety checkpoint is epoch 1 — model is underfitted, but calibration thresholds make up for it with aggressive per-class gating. 4× safety improvement over Run F best_safety on comparable eval. Run F re-evaluated on v2.1 dataset = 0.71% (4/562); I best_safety = 0.18% — I wins by 4×.

---

## Current Best Model

**Run I best_safety + calibration** (as of 2026-07-05):
- Test acc=78.8% (accepted), toxic_fp=**0.18%**, rejection=10.2%
- Checkpoint: `checkpoints/run_i/best_safety.pt`
- Calibration: T=0.707, per-class thresholds in `checkpoints/run_i/calibration.json`
- Loss: ASL(γ+=1, γ-=2, margin=0.05) + label_smoothing=0.1 + toxic_mult=3×

Run scorecard (calibrated toxic FP, all evaluated on v2.1 dataset):
| Run | Checkpoint | Calibrated toxic FP | Accepted acc | Rejection | Notes |
|-----|---|---|---|---|---|
| I best_safety | epoch 1 | **0.18%** (1/544) | 78.8% | 10.2% | ← production checkpoint |
| J best_safety | epoch 2 | 0.50% (3/604) | 84.9% | 5.6% | v2.1 dataset; better UX but worse safety |
| F best_accuracy | epoch 5 | 0.71% (4/562) | 88.1% | 8.2% | re-evaluated on v2.1 |
| J best_accuracy | epoch 8 | 1.82% (11/604) | **91.7%** | 3.0% | best accuracy overall |
| I best_accuracy | epoch 8 | 1.82% (10/548) | 88.7% | 7.0% | |
| G best_safety | epoch 4 | 1.28% | ? | ? | CutMix only (old dataset eval) |
| H best_accuracy | epoch 10 | 1.89% | ? | ? | All regularization (old dataset eval) |
| H best_safety | epoch 4 | 2.73% | ? | ? | Early stop (old dataset eval) |

### Run J — ASL γ-=2 + label smoothing ε=0.1, dataset v2.1 (1,664 ilex_decidua)
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=2, margin=0.05), label_smoothing=0.1
- **Stopped:** epoch 9 (patience=7), best_safety at epoch 2, best_accuracy at epoch 8
- **best_safety:** epoch 2, val_acc=82.8%, val_toxic_fp=2.07%
  - Calibration: T=0.743, toxic_fp=**0.50%** (3/604), acc=84.9%, rejection=5.6%
  - Calibration saved → `checkpoints/run_j/calibration.json`
- **best_accuracy:** epoch 8, val_acc=89.5%, val_toxic_fp=3.63%
  - Calibration: T=0.994, toxic_fp=1.82% (11/604), acc=**91.7%**, rejection=3.0%
- **Notes:** Training on v2.1 (2.5× ilex_decidua) improved best_safety convergence — epoch 2 vs Run I's epoch 1 — and raised accepted accuracy from 78.8% → 84.9% with rejection dropping from 10.2% → 5.6%. But calibrated toxic FP (0.50%) does not beat Run I best_safety (0.18%). Run I remains production. best_accuracy is the best accuracy checkpoint overall (91.7%) but 1.82% toxic FP.

---

## Pending Investigations
- [x] Run F (ASL γ-=2) results + calibration — complete; new best model
- [x] Fruit-presence gate (Layer 1b) — implemented in `src/edible/model/gate.py` (`FruitPresenceGate`); CLIP zero-shot, lazy import, 16 tests
- [x] Intra-class CutMix for toxic species — implemented (feature/training-experiments); Run H showed marginal or negative effect combined with other regularization
- [x] FastAPI backend — implemented and working; `src/edible/api/`
- [x] React frontend — implemented; `frontend/`
- [x] Scrape more ilex_decidua images: expanded 661 → 1,664 (v2.1, blur-only Texas)
- [x] Ablation: Run F config + label smoothing only → Run I; new production checkpoint (0.18% toxic FP)
- [x] Run J: retrain Run I config on v2.1 dataset (1,664 ilex_decidua) — J best_safety (0.50%) does not beat I best_safety (0.18%); Run I stays in production
- [ ] Hard negative mining: boost sampling frequency of known FP images (3 remaining in Run J best_safety)
- [ ] GPS location re-ranking: API accepts lat/lon but ignores it; USDA PLANTS + BONAP needed
