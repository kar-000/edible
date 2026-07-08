"""
SupCon Phase 1: pre-train the EfficientNet-B0 backbone with Supervised Contrastive Loss.

Saves backbone weights to <checkpoint-dir>/supcon_backbone.pt.
Use the resulting file with scripts/train.py --pretrained-backbone to run Phase 2.

Usage
-----
    uv run python scripts/pretrain.py
    uv run python scripts/pretrain.py --checkpoint-dir checkpoints/run_m --epochs 20
    uv run python scripts/pretrain.py --temperature 0.07 --toxic-mult 2.0 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edible.model.classifier import ClassifierConfig
from edible.model.train import PretrainConfig, pretrain_supcon

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SupCon Phase 1: pre-train backbone with supervised contrastive loss"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR / "supcon")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Pre-training epochs (default 20)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size; larger = more contrastive pairs (default 64)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="SupCon temperature τ (default 0.07)")
    parser.add_argument("--toxic-mult", type=float, default=2.0,
                        help="Extra weight on toxic-anchor contrastive pairs (default 2.0)")
    parser.add_argument("--proj-hidden", type=int, default=256,
                        help="Projection head hidden dim (default 256)")
    parser.add_argument("--proj-out", type=int, default=128,
                        help="Projection head output dim (default 128)")
    args = parser.parse_args()

    cfg = PretrainConfig(
        images_dir=DATA_DIR / "images",
        species_db_path=DATA_DIR / "species.json",
        classifier_config=ClassifierConfig(pretrained=True),
        projection_hidden_dim=args.proj_hidden,
        projection_out_dim=args.proj_out,
        supcon_temperature=args.temperature,
        toxic_multiplier=args.toxic_mult,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"Checkpoint dir : {cfg.checkpoint_dir}")
    print(f"Images         : {cfg.images_dir}")
    out_path = pretrain_supcon(cfg)
    print(f"\nPhase 1 complete. Run Phase 2 with:")
    print(
        f"  uv run python scripts/train.py "
        f"--pretrained-backbone {out_path} "
        f"--checkpoint-dir <run_dir> "
        f"--asl --asl-gamma-neg 2 --label-smoothing 0.1 --toxic-mult 3"
    )


if __name__ == "__main__":
    main()
