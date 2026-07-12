"""
Train the Edible species classifier.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --checkpoint-dir checkpoints/run_d
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--model-name", type=str, default="efficientnet_b0",
                        help="timm model name (default: efficientnet_b0)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone weights; train classification head only (linear probe)")
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
    parser.add_argument(
        "--cutmix", action="store_true",
        help="Enable intra-class CutMix for toxic species (prob=0.5, alpha=1.0)",
    )
    parser.add_argument("--cutmix-prob", type=float, default=0.5,
                        help="Probability of applying CutMix per toxic image (default 0.5)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="Beta distribution alpha for CutMix crop size (default 1.0)")
    parser.add_argument(
        "--balanced-sampling", action="store_true",
        help="Use WeightedRandomSampler to balance class representation per batch",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon; 0=disabled, 0.1 recommended (default 0.0)")
    parser.add_argument(
        "--hard-negatives", type=Path, default=None,
        help="JSON file from mine_hard_negatives.py; boosts sampling of known FP images",
    )
    parser.add_argument(
        "--pretrained-backbone", type=Path, default=None,
        help="Path to supcon_backbone.pt from scripts/pretrain.py; "
             "loads backbone weights before fine-tuning",
    )
    args = parser.parse_args()

    hard_negatives: dict = {}
    if args.hard_negatives and args.hard_negatives.exists():
        data = json.loads(args.hard_negatives.read_text())
        repo_root = Path(__file__).parent.parent
        hard_negatives = {repo_root / k: v for k, v in data["samples"].items()}

    cfg = TrainConfig(
        images_dir=DATA_DIR / "images",
        species_db_path=DATA_DIR / "species.json",
        classifier_config=ClassifierConfig(
            model_name=args.model_name,
            freeze_backbone=args.freeze_backbone,
            img_size=224 if "dinov2" in args.model_name or "vit_" in args.model_name else None,
            toxic_loss_multiplier=args.toxic_mult,
            use_asl=args.asl,
            asl_gamma_neg=args.asl_gamma_neg,
            label_smoothing=args.label_smoothing,
        ),
        epochs=args.epochs,
        learning_rate=args.lr,
        toxic_fp_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        cutmix_prob=args.cutmix_prob if args.cutmix else 0.0,
        cutmix_alpha=args.cutmix_alpha,
        balanced_sampling=args.balanced_sampling,
        hard_negatives=hard_negatives,
        pretrained_backbone_path=args.pretrained_backbone,
    )

    loss_name = "ASL" if args.asl else "WeightedCE"
    extras: list[str] = []
    if args.model_name != "efficientnet_b0":
        extras.append(f"model={args.model_name}")
    if args.freeze_backbone:
        extras.append("freeze_backbone=True")
    if args.cutmix:
        extras.append(f"cutmix_prob={args.cutmix_prob}  cutmix_alpha={args.cutmix_alpha}")
    if args.balanced_sampling:
        extras.append("balanced_sampling=True")
    if args.label_smoothing > 0:
        extras.append(f"label_smoothing={args.label_smoothing}")
    if hard_negatives:
        extras.append(f"hard_negatives={len(hard_negatives)}")
    if args.pretrained_backbone and args.pretrained_backbone.exists():
        extras.append(f"supcon_backbone={args.pretrained_backbone.name}")
    extra_str = ("  " + "  ".join(extras)) if extras else ""
    print(f"loss={loss_name}  lr={args.lr}  toxic_mult={args.toxic_mult}x  "
          f"patience={args.patience}{extra_str}")
    print(f"Images: {cfg.images_dir}  →  Checkpoints: {cfg.checkpoint_dir}\n")

    history = train(cfg)
    best = min(history, key=lambda r: r.toxic_fp_rate)
    print(
        f"\nBest epoch {best.epoch}: "
        f"acc={best.val_accuracy:.3f}  toxic_fp={best.toxic_fp_rate:.4f}"
    )


if __name__ == "__main__":
    main()
