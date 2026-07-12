"""
Temperature scaling + per-class confidence thresholding.

Steps
-----
1. Collect raw logits on the val set (no gradient).
2. Optimize a single temperature T that minimises NLL on val set (calibration).
3. For each edible class, find the minimum threshold that eliminates every
   toxic→edible FP on val set.  Any toxic sample predicted edible with
   confidence >= that threshold would slip through; we raise the bar until
   none do.
4. Evaluate the calibrated+thresholded model on the test set, reporting
   accepted accuracy, toxic FP rate, and rejection rate.

Usage
-----
    uv run python scripts/calibrate.py checkpoints/run_d/best_accuracy.pt
"""

from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Temperature search
# ---------------------------------------------------------------------------

def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    t = torch.tensor(logits / T, dtype=torch.float32)
    lp = F.log_softmax(t, dim=1)
    return -lp[np.arange(len(labels)), labels].mean().item()


def find_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Golden-section search for optimal temperature in (0.1, 10.0)."""
    lo, hi = 0.1, 10.0
    gr = (5 ** 0.5 - 1) / 2  # golden ratio
    for _ in range(100):
        m1 = hi - gr * (hi - lo)
        m2 = lo + gr * (hi - lo)
        if _nll(logits, labels, m1) < _nll(logits, labels, m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Per-class threshold search
# ---------------------------------------------------------------------------

def find_per_class_thresholds(
    probs: np.ndarray,          # (N, C) calibrated probabilities
    preds: np.ndarray,          # (N,) argmax predictions
    labels: np.ndarray,         # (N,) ground-truth labels
    toxic_indices: set[int],
    num_classes: int,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    For every edible class c, find the minimum confidence threshold such that
    no toxic sample predicted as c on the val set passes through.

    Returns a (C,) array of thresholds (0.0 = no threshold for that class).
    """
    edible_indices = set(range(num_classes)) - toxic_indices
    thresholds = np.zeros(num_classes)

    for c in edible_indices:
        # Val samples where model predicted class c AND ground truth is toxic
        mask = (preds == c) & np.array([l in toxic_indices for l in labels])
        if not mask.any():
            continue  # no toxic FPs for this class on val → no threshold needed
        # Raise threshold just above the highest-confidence toxic intruder
        thresholds[c] = float(probs[mask, c].max()) + eps

    return thresholds


# ---------------------------------------------------------------------------
# Logit collection
# ---------------------------------------------------------------------------

def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device)).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Evaluation with thresholds
# ---------------------------------------------------------------------------

def evaluate_with_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    toxic_indices: set[int],
    species_ids: list[str],
) -> None:
    num_classes = probs.shape[1]
    edible_indices = set(range(num_classes)) - toxic_indices
    preds = probs.argmax(axis=1)
    confidences = probs[np.arange(len(preds)), preds]

    accepted_mask = confidences >= thresholds[preds]
    rejected_mask = ~accepted_mask

    n = len(labels)
    n_accepted = accepted_mask.sum()
    n_rejected = rejected_mask.sum()

    # Among accepted predictions
    acc_preds = preds[accepted_mask]
    acc_labels = labels[accepted_mask]
    overall_acc = (acc_preds == acc_labels).mean() if n_accepted else 0.0

    # Toxic FP: toxic GT → edible prediction, accepted
    toxic_mask = np.array([l in toxic_indices for l in labels])
    toxic_fp_accepted = (
        toxic_mask & accepted_mask &
        np.array([p in edible_indices for p in preds])
    ).sum()
    toxic_total = toxic_mask.sum()
    toxic_accepted = (toxic_mask & accepted_mask).sum()

    # Edible FN: edible GT → toxic prediction, accepted
    edible_mask = ~toxic_mask
    edible_fn_accepted = (
        edible_mask & accepted_mask &
        np.array([p in toxic_indices for p in preds])
    ).sum()
    edible_total = edible_mask.sum()

    print(f"{'─'*60}")
    print(f"Samples          : {n} total  |  {n_accepted} accepted  |  {n_rejected} rejected ({n_rejected/n:.1%})")
    print(f"Accepted accuracy: {overall_acc:.3f}  (on {n_accepted} predictions)")
    print(f"Toxic FP rate    : {toxic_fp_accepted}/{toxic_total} = {toxic_fp_accepted/toxic_total:.4f}  "
          f"({toxic_fp_accepted}/{toxic_accepted} of accepted toxic samples)")
    print(f"Edible FN rate   : {edible_fn_accepted}/{edible_total} = {edible_fn_accepted/edible_total:.4f}")
    print(f"Toxic rejection  : {(toxic_mask & rejected_mask).sum()}/{toxic_total} toxic samples rejected")
    print(f"Edible rejection : {(edible_mask & rejected_mask).sum()}/{edible_total} edible samples rejected")
    print(f"{'─'*60}")

    # Per-class threshold summary
    print("\nPer-class thresholds (edible classes only):")
    for c, sid in enumerate(species_ids):
        if c in edible_indices and thresholds[c] > 0:
            print(f"  {sid:<35} θ={thresholds[c]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--save", action="store_true",
        help="Save calibration (T + thresholds) to <checkpoint_dir>/calibration.json",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    val_ds = EdibleDataset(DATA_DIR / "images", DATA_DIR / "species.json", split="val")
    test_ds = EdibleDataset(DATA_DIR / "images", DATA_DIR / "species.json", split="test")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = val_ds.num_classes()
    toxic_indices = val_ds.toxic_class_indices()
    species_ids = val_ds.species_ids()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ClassifierConfig(
        num_classes=num_classes,
        model_name=ckpt.get("model_name", "efficientnet_b0"),
        img_size=ckpt.get("img_size", None),
        freeze_backbone=ckpt.get("freeze_backbone", False),
    )
    model = build_classifier(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Checkpoint : {args.checkpoint.name}  "
          f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_accuracy',0):.3f}, "
          f"val_toxic_fp={ckpt.get('toxic_fp_rate',0):.4f})")
    print(f"Device     : {device}\n")

    # 1. Collect val logits
    print("Collecting val logits...")
    val_logits, val_labels = collect_logits(model, val_loader, device)

    # 2. Temperature scaling
    T = find_temperature(val_logits, val_labels)
    print(f"Optimal temperature: T={T:.4f}  "
          f"(T<1 = was overconfident, T>1 = was underconfident)\n")

    # 3. Val calibrated probabilities + thresholds
    val_probs = torch.softmax(torch.tensor(val_logits / T), dim=1).numpy()
    val_preds = val_probs.argmax(axis=1)
    thresholds = find_per_class_thresholds(
        val_probs, val_preds, val_labels, toxic_indices, num_classes
    )

    # 4. Test set evaluation
    print("Collecting test logits...")
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_probs = torch.softmax(torch.tensor(test_logits / T), dim=1).numpy()

    print("\n── Uncalibrated test set (baseline) ──────────────────────")
    base_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    zero_thresholds = np.zeros(num_classes)
    evaluate_with_thresholds(base_probs, test_labels, zero_thresholds, toxic_indices, species_ids)

    print("\n── Calibrated + per-class thresholds ─────────────────────")
    evaluate_with_thresholds(test_probs, test_labels, thresholds, toxic_indices, species_ids)

    if args.save:
        cal = {
            "checkpoint": str(args.checkpoint),
            "temperature": round(T, 6),
            "thresholds": {
                species_ids[c]: round(float(thresholds[c]), 6)
                for c in range(num_classes)
                if thresholds[c] > 0
            },
        }
        save_path = args.checkpoint.parent / "calibration.json"
        save_path.write_text(__import__("json").dumps(cal, indent=2))
        print(f"\nCalibration saved → {save_path}")


if __name__ == "__main__":
    main()
