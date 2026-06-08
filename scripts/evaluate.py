"""
Evaluate a saved checkpoint on the test split.

Usage:
    uv run python scripts/evaluate.py checkpoints/best_safety.pt
    uv run python scripts/evaluate.py checkpoints/run_d/best_accuracy.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from edible.model.classifier import ClassifierConfig, build_classifier
from edible.model.dataset import EdibleDataset
from edible.model.evaluate import evaluate_model

DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split")
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint file")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    test_ds = EdibleDataset(DATA_DIR / "images", DATA_DIR / "species.json", split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = test_ds.num_classes()
    toxic_indices = test_ds.toxic_class_indices()
    species_ids = test_ds.species_ids()

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ClassifierConfig(num_classes=num_classes)
    model = build_classifier(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    saved_epoch = ckpt.get("epoch", "?")
    saved_acc = ckpt.get("val_accuracy", "?")
    saved_fp = ckpt.get("toxic_fp_rate", "?")
    print(f"Checkpoint: {args.checkpoint.name}  (epoch={saved_epoch}, val_acc={saved_acc:.3f}, val_toxic_fp={saved_fp:.4f})")
    print(f"Test set:   {len(test_ds)} samples  |  device={device}\n")

    metrics = evaluate_model(model, test_loader, device, num_classes, toxic_indices, species_ids)

    print(metrics.safety_summary())
    print()
    print(metrics.classification_report())


if __name__ == "__main__":
    main()
