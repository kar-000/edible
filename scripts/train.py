"""
Train the Edible species classifier.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --checkpoint-dir checkpoints/run_d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edible.model.classifier import ClassifierConfig
from edible.model.train import TrainConfig, train

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Edible species classifier")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--toxic-mult", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--asl", action="store_true",
        help="Use Asymmetric Loss instead of weighted CrossEntropy",
    )
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0,
                        help="ASL γ- (negative focusing strength, default 4.0)")
    args = parser.parse_args()

    cfg = TrainConfig(
        images_dir=DATA_DIR / "images",
        species_db_path=DATA_DIR / "species.json",
        classifier_config=ClassifierConfig(
            toxic_loss_multiplier=args.toxic_mult,
            use_asl=args.asl,
            asl_gamma_neg=args.asl_gamma_neg,
        ),
        epochs=args.epochs,
        learning_rate=args.lr,
        toxic_fp_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )

    loss_name = "ASL" if args.asl else "WeightedCE"
    print(f"loss={loss_name}  lr={args.lr}  toxic_mult={args.toxic_mult}x  "
          f"patience={args.patience}")
    print(f"Images: {cfg.images_dir}  →  Checkpoints: {cfg.checkpoint_dir}\n")

    history = train(cfg)
    best = min(history, key=lambda r: r.toxic_fp_rate)
    print(
        f"\nBest epoch {best.epoch}: "
        f"acc={best.val_accuracy:.3f}  toxic_fp={best.toxic_fp_rate:.4f}"
    )


if __name__ == "__main__":
    main()
