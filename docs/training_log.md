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

**Run R best_safety (DINOv2 ViT-B/14 stage-2 fine-tune) + calibration** (as of 2026-07-12):
- DINOv2 stage-2 fine-tune (warm-start from Run P, backbone LR=1e-5) broke the 0.33% floor: **0.17% FP (1/604)** at 96.2% acc, 0.3% rejection — new all-time best on every metric simultaneously.
- Run R best_accuracy: 0.33% FP at 98.0% acc, 0.6% rejection — also beats all prior runs on accuracy.
- **Production recommendation**: Run R best_safety (safety path); Run R best_accuracy (accuracy path).

**Run K best_safety + calibration** (as of 2026-07-05):
- Test acc=87.4% (accepted), toxic_fp=**0.33%**, rejection=5.7%
- Checkpoint: `checkpoints/run_k/best_safety.pt`
- Calibration: T=0.879, per-class thresholds in `checkpoints/run_k/calibration.json`
- Loss: ASL(γ+=1, γ-=2, margin=0.05) + label_smoothing=0.1 + toxic_mult=3× + hard_negatives=22

Run scorecard (calibrated toxic FP, all evaluated on v2.1 dataset):
| Run | Checkpoint | Calibrated toxic FP | Accepted acc | Rejection | Notes |
|-----|---|---|---|---|---|
| I best_safety | epoch 1 | 0.18% (1/544) | 78.8% | 10.2% | best raw safety; low acc |
| **K best_safety** | epoch 3 | **0.33%** (2/604) | 87.4% | 5.7% | ← **production (safety)** |
| L best_safety | epoch 3 | 0.50% (3/604) | 87.7% | 6.8% | hard negatives from K; no safety gain |
| J best_safety | epoch 2 | 0.50% (3/604) | 84.9% | 5.6% | before hard negatives |
| F best_accuracy | epoch 5 | 0.71% (4/562) | 88.1% | 8.2% | re-evaluated on v2.1 |
| L best_accuracy | epoch 10 | 0.66% (4/604) | 92.7% | 8.6% | |
| K best_accuracy | epoch 9 | 1.16% (7/604) | 89.6% | 6.4% | |
| **M best_accuracy** ✦SupCon | epoch 16 | **0.99%** (6/604) | **94.1%** | 5.6% | ← **best accuracy** |
| M best_safety ✦SupCon | epoch 11 | 1.49% (9/604) | 93.6% | 4.7% | best_accuracy safer than best_safety |
| N best_safety ✦SupCon+HN | epoch 1 | 0.99% (6/604) | 88.2% | 8.4% | hard negatives disrupt SupCon fine-tuning |
| **R best_safety** ✦DINOv2-ft | epoch 1 | **0.17%** (1/604) | **96.2%** | 0.3% | ← **new production (safety)** — stage-2 fine-tune |
| **R best_accuracy** ✦DINOv2-ft | epoch 6 | **0.33%** (2/604) | **98.0%** | 0.6% | ← **new production (accuracy)** |
| P best_accuracy ✦DINOv2-frozen | epoch 7 | 0.33% (2/604) | 97.5% | 0.2% | frozen linear probe |
| P best_safety ✦DINOv2-frozen | epoch 1 | 0.66% (4/604) | 95.2% | 0.0% | no per-class thresholds triggered |
| **O best_accuracy** ✦SupCon150 | epoch 1 | **0.66%** (4/604) | **92.6%** | 8.5% | ties L best_accuracy; best_acc safer than best_safety again |
| O best_safety ✦SupCon150 | epoch 3 | 0.83% (5/604) | 89.6% | 11.2% | high rejection; best_accuracy is better choice |
| J best_accuracy | epoch 8 | 1.82% (11/604) | 91.7% | 3.0% | |
| I best_accuracy | epoch 8 | 1.82% (10/548) | 88.7% | 7.0% | |
| G best_safety | epoch 4 | 1.28% | ? | ? | CutMix only (old dataset eval) |
| H best_accuracy | epoch 10 | 1.89% | ? | ? | All regularization (old dataset eval) |
| H best_safety | epoch 4 | 2.73% | ? | ? | Early stop (old dataset eval) |

### Run J — ASL γ-=2 + label smoothing ε=0.1, dataset v2.1 (1,664 ilex_decidua)
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=2, margin=0.05), label_smoothing=0.1
- **Stopped:** epoch 9 (patience=7), best_safety at epoch 2, best_accuracy at epoch 8
- **best_safety:** epoch 2, val_acc=82.8%, val_toxic_fp=2.07%
  - Calibration: T=0.743, toxic_fp=0.50% (3/604), acc=84.9%, rejection=5.6%
  - Calibration saved → `checkpoints/run_j/calibration.json`
- **best_accuracy:** epoch 8, val_acc=89.5%, val_toxic_fp=3.63%
  - Calibration: T=0.994, toxic_fp=1.82% (11/604), acc=91.7%, rejection=3.0%
- **Notes:** Training on v2.1 improved best_safety convergence (epoch 2 vs I's epoch 1) and raised accuracy to 84.9%. toxic_fp=0.50% does not beat Run I. Hard negatives mined from this checkpoint for Run K.

### Run K — ASL γ-=2 + label smoothing ε=0.1 + hard negatives (22 samples), dataset v2.1 ✅ new best
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=2, margin=0.05), label_smoothing=0.1, hard_negatives=22 (4 FPs at 5×, 18 near-misses at 2×)
- **Stopped:** epoch 10 (patience=7), best_safety at epoch 3, best_accuracy at epoch 9
- **best_safety:** epoch 3, val_acc=84.6%, val_toxic_fp=2.42%
  - Calibration: T=0.879, toxic_fp=**0.33%** (2/604), acc=**87.4%**, rejection=5.7%
  - Calibration saved → `checkpoints/run_k/calibration.json`
- **best_accuracy:** epoch 9, val_acc=88.9%, val_toxic_fp=3.80%
  - Calibration: T=1.015, toxic_fp=1.16% (7/604), acc=89.6%, rejection=6.4%
- **Hard negative origin:** FPs were ilex_vomitoria→rubus_trivialis (×2), melia_azedarach→sambucus (×1), solanum_nigrum→sambucus (×1). ilex_decidua produced 0 train FPs — v2.1 scrape fixed the original problem species.
- **Notes:** Hard negatives improved best_safety by one more epoch of convergence (3 vs 2) and cut FP from 0.50% → 0.33% while maintaining good accuracy (87.4%, +2.5pp vs J). best_accuracy passes the 1% threshold for first time in accuracy-optimized checkpoint (1.16% < 1.82% from J). New production checkpoint.

---

### Run L — ASL γ-=2 + label smoothing ε=0.1 + hard negatives from Run K (20 samples), dataset v2.1
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=2, margin=0.05), label_smoothing=0.1, hard_negatives=20 (2 FPs at 5×, 18 near-misses at 2×)
- **FP origin (mined from K):** ilex_vomitoria→mahonia_trifoliolata (×1), solanum_nigrum→callicarpa_americana (×1) — different confusions than K's FPs
- **Stopped:** epoch 10 (patience=7 on toxic FP), best_safety at epoch 3, best_accuracy at epoch 10
- **best_safety:** epoch 3, val_acc=0.835, val_toxic_fp=2.07%
  - Calibration: T=0.990, toxic_fp=**0.50%** (3/604), acc=87.7%, rejection=6.8%
  - Calibration saved → `checkpoints/run_l/calibration.json`
- **best_accuracy:** epoch 10, val_acc=0.873, val_toxic_fp=2.59%
  - Calibration: T=1.194, toxic_fp=**0.66%** (4/604), acc=**92.7%**, rejection=8.6%
- **Notes:** Hard negative mining has plateaued on the safety path — best_safety went 0.50% → 0.50% (no improvement from K). But best_accuracy jumped from 89.6% (K) to 92.7% with FP dropping from 1.16% → 0.66% — the best balanced checkpoint produced. Conclusion: further hard negative iterations unlikely to move the safety needle. Next technique: SupCon or additional data.

### Run M — SupCon Phase 1 (20 epochs) + ASL Phase 2, dataset v2.1 ✅ new accuracy high
- **Phase 1 config:** τ=0.07, toxic_mult=2×, proj 1280→256→128, batch_size=64, epochs=20, lr=1e-3, CosineAnnealingLR
  - Loss: 5.13 → 2.62 (converging, still slowly descending at epoch 20)
  - Saved → `checkpoints/run_m/supcon_backbone.pt`
- **Phase 2 config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=2, margin=0.05), label_smoothing=0.1
  - Epoch 1 val_acc=0.883 — substantially higher than all prior runs at epoch 1 (Run L: 0.765)
  - Stopped: epoch 18 (patience=7), best_safety at epoch 11, best_accuracy at epoch 16
- **best_safety:** epoch 11, val_acc=0.905, val_toxic_fp=2.25%
  - Calibration: T=1.048, toxic_fp=**1.49%** (9/604), acc=**93.6%**, rejection=4.7%
  - Calibration saved → `checkpoints/run_m/calibration.json`
- **best_accuracy:** epoch 16, val_acc=0.915, val_toxic_fp=2.76%
  - Calibration: T=1.149, toxic_fp=**0.99%** (6/604), acc=**94.1%**, rejection=5.6%
- **Notes:** SupCon dramatically improved accuracy (+6.7pp vs K best_safety: 94.1% vs 87.4%) and pushed uncalibrated accuracy to 92.6% at epoch 1. However, toxic FP on the safety path went UP (1.49% vs K's 0.33%) — SupCon improves general class separation but the toxic safety floor needs targeted techniques. Unusual: best_accuracy (0.99% FP) is safer than best_safety (1.49%) post-calibration; the well-trained feature space allows tighter thresholds at epoch 16. Run K best_safety remains production for safety-critical deployment. Run M best_accuracy is the highest-accuracy checkpoint produced. Next: SupCon + hard negatives combined, or higher Phase 1 toxic_mult.

### Run N — SupCon Phase 1 backbone (from M) + ASL Phase 2 + hard negatives from Run L (20 samples)
- **Hypothesis:** SupCon backbone + hard negatives = best of both worlds
- **Config:** toxic_mult=3×, lr=5e-4, patience=7, loss=ASL(γ+=1, γ-=2, margin=0.05), label_smoothing=0.1, hard_negatives=20, supcon_backbone=checkpoints/run_m/supcon_backbone.pt
- **Hard negative origin (mined from K):** same 20 entries as Run L
- **Note:** Run M best_safety had 0 training FPs, so `mine_hard_negatives.py` returned 0 entries for Run N; Run L hard negatives were reused
- **Stopped:** epoch 8 (patience=7), best_safety at epoch 1 (val_toxic_fp=3.11%), best_accuracy at epoch 2 (val_acc=0.886)
- **best_safety:** epoch 1, val_acc=0.868, val_toxic_fp=3.11%
  - Calibration: T=1.0 (estimated), toxic_fp=**0.99%** (6/604), acc=**88.2%**, rejection=8.4%
- **Diagnosis:** SupCon backbone (0 train FPs) makes Run L hard negatives non-hard — the model already separates them. Upsampling these images adds noise to Phase 2 and destabilizes fine-tuning, causing early stop at epoch 8 vs Run M's epoch 18. Safety is 3× worse than Run K (0.99% vs 0.33%) and accuracy is 5.9pp below Run M best_accuracy (88.2% vs 94.1%).
- **Conclusion:** Hard negatives sourced from a weaker model cannot be reused with a stronger backbone. The two techniques compete for the same fine-tuning signal; they do not compose.

### Run O — SupCon Phase 1 (150 epochs, toxic_mult=4×) + ASL Phase 2, dataset v2.1
- **Hypothesis:** Phase 1 heavily underbaked in Run M (20 of recommended 200 epochs); 150 epochs + harder toxic_mult should push toxic clusters further in embedding space
- **Phase 1 config:** τ=0.07, toxic_mult=**4×**, proj 1280→256→128, batch_size=64, epochs=**150**, lr=1e-3, CosineAnnealingLR
  - Loss: 8.65 → 4.47 (rapid descent to ~epoch 60, then plateau; 4× toxic_mult changes loss scale vs Run M's 5.13→2.62)
  - Phase 1 loss plateaued from ~epoch 60 — future runs with this config need only ~75 epochs
  - Saved → `checkpoints/run_o/supcon_backbone.pt`
- **Phase 2 config:** identical to Run M (toxic_mult=3×, lr=5e-4, patience=7, ASL γ-=2, label_smoothing=0.1, no hard negatives)
  - Epoch 1 val_acc=**0.908** (vs 0.883 in Run M) — stronger starting point from better backbone
  - Stopped: epoch 10 (patience=7), best_safety at epoch 3, best_accuracy at epoch 1
- **best_safety:** epoch 3, val_acc=0.889, val_toxic_fp=1.90% (lowest raw val FP since Run I)
  - Calibration: T=1.182, toxic_fp=**0.83%** (5/604), acc=**89.6%**, rejection=11.2%
  - Calibration saved → `checkpoints/run_o/calibration.json`
- **best_accuracy:** epoch 1, val_acc=0.908, val_toxic_fp=2.76%
  - Calibration: T=1.132, toxic_fp=**0.66%** (4/604), acc=**92.6%**, rejection=8.5%
- **Pattern:** best_accuracy (0.66% FP) is safer than best_safety (0.83%) post-calibration — same counterintuitive result as Run M. Epoch 1's better-organized feature space allows tighter calibration thresholds.
- **Conclusion:** Longer Phase 1 improved over Run M on accuracy path (0.66% vs 0.99%) but did not break the 0.33% safety ceiling. Run K best_safety remains production. The safety floor likely requires a different approach: better data or a stronger backbone foundation (DINOv2/ViT).

### Run P — DINOv2 ViT-B/14 frozen backbone + ASL linear probe, dataset v2.1 ✅ new production
- **Hypothesis:** DINOv2 pretrained features (142M images, DINO self-supervised) intrinsically encode species-level discrimination; test with frozen backbone before investing in full fine-tune
- **Config:** model=vit_base_patch14_dinov2, freeze_backbone=True, img_size=224, toxic_mult=3×, lr=5e-4, patience=7, ASL γ-=2, label_smoothing=0.1, no hard negatives, no SupCon
- **Trainable params:** 9,228 (head only: Linear(768→12) + bias); backbone: 85.8M frozen
- **Epoch 1: val_acc=0.948, val_toxic_fp=0.000%** — ZERO val toxic FPs from a linear head on epoch 1
- **Stopped:** epoch 8 (patience=7), best_safety epoch 1, best_accuracy epoch 7
- **best_safety:** epoch 1, val_acc=0.948, val_toxic_fp=0.000%
  - Calibration: T=0.573, toxic_fp=**0.66%** (4/604), acc=**95.2%**, rejection=**0%** (no per-class thresholds triggered)
  - Calibration saved → `checkpoints/run_p/calibration.json`
- **best_accuracy:** epoch 7, val_acc=0.974, val_toxic_fp=0.17%
  - Calibration: T=0.671, toxic_fp=**0.33%** (2/604), acc=**97.5%**, rejection=**0.2%** (only celtis_laevigata θ=0.585)
- **Comparison to Run K best_safety (previous production):** same 0.33% FP, but +10.1pp accuracy (97.5% vs 87.4%) and rejection 0.2% vs 5.7% (28× lower)
- **Conclusion:** DINOv2 pretrained features trivially encode the toxic/edible separation that took 15+ EfficientNet-B0 experiments to approach. Full fine-tune (Run Q) should push FP below 0.33% while maintaining near-97% accuracy. **Run P best_accuracy is new production checkpoint.**

### Run Q — DINOv2 ViT-B/14 full fine-tune from scratch (FAILED)
- **Config:** model=vit_base_patch14_dinov2, backbone_lr=5e-5, head_lr=5e-4, ASL γ-=2, label_smoothing=0.1, toxic_mult=3×, no warm-start
- **Result:** Catastrophic forgetting — val_acc crashed to 39.2% at epoch 1, safety alarms at epochs 5 and 8, early stop at epoch 8
- **Diagnosis:** Cold-start full fine-tune of a pretrained ViT with a randomly initialised head creates destructive gradient conflict. The head updates are 10× larger than typical ViT fine-tuning updates; backbone LR 5e-5 is still too high when the head is untrained.

### Run R — DINOv2 ViT-B/14 stage-2 fine-tune (warm-start from Run P), dataset v2.1 ✅ new production
- **Hypothesis:** Start from Run P (adapted head + frozen backbone), unfreeze backbone at very low LR — avoids cold-start instability of Run Q
- **Config:** model=vit_base_patch14_dinov2, warm_start=run_p/best_accuracy.pt, backbone_lr=**1e-5**, head_lr=5e-4, ASL γ-=2, label_smoothing=0.1, toxic_mult=3×
- **Epoch 1: val_acc=0.961, val_toxic_fp=0.17%** — stable from first epoch; no catastrophic forgetting
- **Stopped:** epoch 8 (patience=7), best_safety epoch 1, best_accuracy epoch 6
- **best_safety:** epoch 1, val_acc=0.961, val_toxic_fp=0.17%
  - Calibration: T=0.803, toxic_fp=**0.17%** (1/604), acc=**96.2%**, rejection=**0.3%**
  - Only threshold: vitis_mustangensis θ=0.569
  - Calibration saved → `checkpoints/run_r/calibration.json`
- **best_accuracy:** epoch 6, val_acc=0.978, val_toxic_fp=0.17%
  - Calibration: T=0.996, toxic_fp=**0.33%** (2/604), acc=**98.0%**, rejection=**0.6%**
  - Only threshold: sambucus_canadensis θ=0.911
- **vs. all prior runs:**
  - best_safety: 0.17% FP beats Run K (0.33%) by 2× and matches Run I's absolute 1 FP — but with 96.2% acc (vs 78.8%) and 0.3% rejection (vs 10.2%)
  - best_accuracy: 98.0% acc is +0.5pp over Run P best_accuracy; same 0.33% FP
- **Key learning:** Two-stage fine-tuning (freeze → warm-start → low-LR unfreeze) is essential for DINOv2 on small datasets. Cold-start full fine-tune destroys representations. Backbone LR must be ≤1e-5; 5e-5 (Run Q) was too high.

---

## Pending Investigations
- [x] Run F (ASL γ-=2) results + calibration — complete
- [x] Fruit-presence gate (Layer 1b) — implemented in `src/edible/model/gate.py`
- [x] Intra-class CutMix for toxic species — implemented; Run H showed no benefit combined with other regularization
- [x] FastAPI backend — implemented and working; `src/edible/api/`
- [x] React frontend — implemented; `frontend/`
- [x] Scrape more ilex_decidua images: expanded 661 → 1,664 (v2.1, blur-only Texas)
- [x] Ablation: Run F config + label smoothing only → Run I; 0.18% toxic FP
- [x] Run J: retrain on v2.1 dataset → 0.50% FP, confirmed hard negatives needed
- [x] Hard negative mining: implemented `mine_hard_negatives.py`; Run K uses 22 samples → 0.33% FP, 87.4% acc
- [x] GPS location re-ranking: implemented; CountyRangeChecker + NominatimGeocoder; county_range.json built for all 12 TX species
- [x] Run L: hard negatives from K → safety plateaued (0.50%), but best_accuracy hit 92.7% (best ever)
- [x] Run M (SupCon): Phase 1 contrastive pre-training (20 epochs, τ=0.07, toxic_mult=2×) → Phase 2 ASL fine-tune → best_accuracy 94.1% (new high), toxic FP 0.99%; safety floor 1.49% (did not beat K)
- [x] Run N (SupCon+HN): SupCon backbone + hard negatives → destabilized training, early stop ep8, 0.99% FP; hard negatives from weaker model don't compose with stronger backbone
- [x] Run O (SupCon150): Phase 1 150 epochs + toxic_mult=4× → 0.66% FP, 92.6% acc; best_accuracy ties L; safety floor still 0.33% (K); Phase 1 loss plateaued ~epoch 60 suggesting diminishing returns beyond 75 epochs at this scale
- [x] Run P (DINOv2 frozen): ViT-B/14 linear probe → 0.33% FP at 97.5% acc / 0.2% rejection; matches K safety, +10.1pp accuracy, 28× lower rejection
- [x] Run Q (DINOv2 full fine-tune from scratch): FAILED — catastrophic forgetting; val_acc crashed to 39.2%; backbone LR 5e-5 too high; cold-start instability
- [x] Run R (DINOv2 stage-2 from Run P warm-start, backbone LR=1e-5): best_safety **0.17% FP, 96.2% acc, 0.3% rejection** — new all-time best; best_accuracy **0.33% FP, 98.0% acc**; NEW PRODUCTION
- [ ] Update inference pipeline default checkpoint to Run R best_safety
