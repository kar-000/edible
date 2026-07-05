"""
Mine hard negative training samples for the next training run.

Runs a calibrated checkpoint over the TRAIN split and identifies:
  - FPs:         toxic image where calibrated model predicts edible AND
                 confidence >= per-class threshold → boost by --fp-boost (default 5.0)
  - Near-misses: toxic image where calibrated model predicts edible but
                 confidence < threshold (would be rejected) → boost by
                 --near-miss-boost (default 2.0)

Output JSON format::

    {
      "checkpoint": "checkpoints/run_j/best_safety.pt",
      "fp_boost": 5.0,
      "near_miss_boost": 2.0,
      "counts": {"fps": N, "near_misses": M},
      "samples": {
        "data/images/ilex_decidua/12345_67890.jpg": 5.0,
        ...
      }
    }

Paths in ``samples`` are relative to the repository root so the file is
portable across machines.

Usage
-----
    uv run python scripts/mine_hard_negatives.py \\
        checkpoints/run_j/best_safety.pt \\
        --calibration checkpoints/run_j/calibration.json \\
        --out data/hard_negatives.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edible.model.classifier import ClassifierConfig, build_classifier
from edible.model.dataset import EdibleDataset

DATA_DIR = Path(__file__).parent.parent / "data"
REPO_ROOT = Path(__file__).parent.parent


def _load_calibration(cal_path: Path) -> tuple[float, dict[str, float]]:
    """Return (temperature, {species_id: threshold}) from calibration.json."""
    data = json.loads(cal_path.read_text())
    return data["temperature"], data.get("thresholds", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negative training samples")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--calibration", type=Path, default=None,
        help="calibration.json produced by calibrate.py (default: <checkpoint_dir>/calibration.json)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/hard_negatives.json"),
        help="Output JSON path (default: data/hard_negatives.json)",
    )
    parser.add_argument("--fp-boost", type=float, default=5.0,
                        help="Sampling boost for confirmed FPs (default: 5.0)")
    parser.add_argument("--near-miss-boost", type=float, default=2.0,
                        help="Sampling boost for near-misses (default: 2.0)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    cal_path = args.calibration or (args.checkpoint.parent / "calibration.json")
    if not cal_path.exists():
        print(f"Calibration file not found: {cal_path}")
        print("Run calibrate.py --save first.")
        sys.exit(1)

    temperature, cal_thresholds = _load_calibration(cal_path)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Calibration : {cal_path}  (T={temperature:.4f})")

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    train_ds = EdibleDataset(DATA_DIR / "images", DATA_DIR / "species.json", split="train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = train_ds.num_classes()
    toxic_indices = train_ds.toxic_class_indices()
    edible_indices = set(range(num_classes)) - toxic_indices
    species_ids = train_ds.species_ids()

    # Build per-class threshold array (index → threshold)
    thresholds = np.zeros(num_classes)
    for sid, thr in cal_thresholds.items():
        if sid in train_ds.class_to_idx:
            thresholds[train_ds.class_to_idx[sid]] = thr

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ClassifierConfig(num_classes=num_classes)
    model = build_classifier(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Train split : {len(train_ds)} samples  |  device={device}\n")
    print("Scanning training set for hard negatives...")

    # Collect logits sample-by-sample (preserve index ordering)
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in train_loader:
            logits = model(images.to(device)).cpu().numpy()
            all_logits.append(logits)

    logits_np = np.concatenate(all_logits)  # (N, C)
    probs = torch.softmax(torch.tensor(logits_np / temperature), dim=1).numpy()
    preds = probs.argmax(axis=1)
    confidences = probs[np.arange(len(preds)), preds]

    samples: dict[str, float] = {}
    n_fps = 0
    n_near_misses = 0

    for i, record in enumerate(train_ds.samples):
        if record.class_idx not in toxic_indices:
            continue  # only mine toxic training images

        pred = int(preds[i])
        conf = float(confidences[i])

        if pred not in edible_indices:
            continue  # correctly predicted as some toxic class — skip

        # Predicted as edible
        rel_path = str(record.image_path.relative_to(REPO_ROOT))
        if conf >= thresholds[pred]:
            # Would pass through as a live FP
            samples[rel_path] = args.fp_boost
            n_fps += 1
        else:
            # Predicted edible but low confidence — rejected by threshold (near-miss)
            samples[rel_path] = args.near_miss_boost
            n_near_misses += 1

    print(f"\nFound {n_fps} FPs (boost={args.fp_boost}×) and "
          f"{n_near_misses} near-misses (boost={args.near_miss_boost}×) "
          f"in {len(train_ds)} training samples.")

    # Species breakdown
    fp_by_species: dict[str, int] = {}
    nm_by_species: dict[str, int] = {}
    for i, record in enumerate(train_ds.samples):
        if record.class_idx not in toxic_indices:
            continue
        pred = int(preds[i])
        conf = float(confidences[i])
        if pred not in edible_indices:
            continue
        sid = record.species_id
        if conf >= thresholds[pred]:
            fp_by_species[sid] = fp_by_species.get(sid, 0) + 1
        else:
            nm_by_species[sid] = nm_by_species.get(sid, 0) + 1

    if fp_by_species:
        print("\nFPs by toxic species:")
        for sid, cnt in sorted(fp_by_species.items(), key=lambda x: -x[1]):
            pred_classes = [
                species_ids[int(preds[i])]
                for i, r in enumerate(train_ds.samples)
                if r.species_id == sid
                and r.class_idx in toxic_indices
                and int(preds[i]) in edible_indices
                and float(confidences[i]) >= thresholds[int(preds[i])]
            ]
            from collections import Counter
            pred_summary = ", ".join(f"{s}×{c}" for s, c in Counter(pred_classes).most_common(3))
            print(f"  {sid:<35} {cnt:3d}  → predicted as: {pred_summary}")

    if nm_by_species:
        print("\nNear-misses by toxic species:")
        for sid, cnt in sorted(nm_by_species.items(), key=lambda x: -x[1]):
            print(f"  {sid:<35} {cnt:3d}")

    output = {
        "checkpoint": str(args.checkpoint),
        "fp_boost": args.fp_boost,
        "near_miss_boost": args.near_miss_boost,
        "counts": {"fps": n_fps, "near_misses": n_near_misses},
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {args.out}  ({len(samples)} entries)")


if __name__ == "__main__":
    main()
