"""
Training loop for the Edible species classifier.

Features
--------
* Toxic-class weighted CrossEntropyLoss (2× default multiplier)
* Per-epoch safety check: alarm if toxic FP rate > 5 % on val set
* Checkpoint saving: best val accuracy + best toxic-FP rate
* Early stopping on toxic FP rate (configurable patience)
* Colab-friendly: all paths are injectable; GPU auto-detected

Typical usage::

    from pathlib import Path
    from edible.model.train import TrainConfig, train

    cfg = TrainConfig(
        images_dir=Path("data/images"),
        species_db_path=Path("data/species.json"),
        checkpoint_dir=Path("checkpoints"),
    )
    history = train(cfg)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edible.model.classifier import ClassifierConfig, build_classifier, build_loss
from edible.model.dataset import EdibleDataset
from edible.model.evaluate import SafetyMetrics, evaluate_model

# Toxic FP rate above this triggers an alarm (but does NOT stop training
# unless patience is exceeded)
TOXIC_FP_ALARM_THRESHOLD = 0.05


def _intra_class_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    toxic_indices: set[int],
    prob: float,
    alpha: float,
) -> torch.Tensor:
    """Paste a random crop from a same-class toxic image into each toxic image.

    Labels are unchanged because mixing is strictly intra-class.
    Skipped silently when a batch contains only one image of a given toxic class.
    """
    images = images.clone()
    B, _, H, W = images.shape
    beta = torch.distributions.Beta(alpha, alpha)

    for i in range(B):
        label_i = int(labels[i].item())
        if label_i not in toxic_indices or torch.rand(1).item() > prob:
            continue

        candidates = [j for j in range(B) if j != i and int(labels[j].item()) == label_i]
        if not candidates:
            continue

        j = candidates[int(torch.randint(len(candidates), (1,)).item())]
        lam = float(beta.sample().item())

        cut_h = int(H * (1 - lam) ** 0.5)
        cut_w = int(W * (1 - lam) ** 0.5)
        cx = int(torch.randint(W, (1,)).item())
        cy = int(torch.randint(H, (1,)).item())

        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, x1 + cut_w)
        y2 = min(H, y1 + cut_h)

        images[i, :, y1:y2, x1:x2] = images[j, :, y1:y2, x1:x2]

    return images


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_metrics: SafetyMetrics
    elapsed_seconds: float

    @property
    def val_accuracy(self) -> float:
        return self.val_metrics.overall_accuracy

    @property
    def toxic_fp_rate(self) -> float:
        return self.val_metrics.toxic_fp_rate


@dataclass
class TrainConfig:
    # Data
    images_dir: Path = Path("data/images")
    species_db_path: Path = Path("data/species.json")

    # Model
    classifier_config: ClassifierConfig = field(default_factory=ClassifierConfig)

    # Training
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0  # 0 = main process (safer on Windows/WSL)

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_best_accuracy: bool = True
    save_best_toxic_fp: bool = True

    # Early stopping on toxic FP rate
    toxic_fp_patience: int = 5  # stop if no improvement for N epochs
    toxic_fp_min_delta: float = 0.005  # improvement threshold

    # Intra-class CutMix (toxic species only)
    cutmix_prob: float = 0.0   # 0 = disabled
    cutmix_alpha: float = 1.0  # Beta distribution shape parameter

    # Device
    device: Optional[torch.device] = None


def _get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device is not None:
        return cfg.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: TrainConfig) -> list[EpochResult]:
    """
    Full training loop.  Returns per-epoch results.

    Raises
    ------
    FileNotFoundError
        If ``images_dir`` or ``species_db_path`` do not exist.
    RuntimeError
        If the training dataset is empty.
    """
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = _get_device(cfg)

    # ------------------------------------------------------------------ #
    # Datasets & loaders                                                   #
    # ------------------------------------------------------------------ #
    train_ds = EdibleDataset(cfg.images_dir, cfg.species_db_path, split="train")
    val_ds = EdibleDataset(cfg.images_dir, cfg.species_db_path, split="val")

    if len(train_ds) == 0:
        raise RuntimeError(
            f"Training dataset is empty. Check that {cfg.images_dir} contains "
            "subdirectories named after species IDs."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    num_classes = train_ds.num_classes()
    toxic_indices = train_ds.toxic_class_indices()
    species_ids = train_ds.species_ids()

    # ------------------------------------------------------------------ #
    # Model, loss, optimiser                                               #
    # ------------------------------------------------------------------ #
    cfg.classifier_config.num_classes = num_classes
    model = build_classifier(cfg.classifier_config).to(device)

    class_weights = train_ds.class_weights().to(device)
    cc = cfg.classifier_config
    criterion = build_loss(
        class_weights=class_weights,
        toxic_indices=toxic_indices,
        toxic_multiplier=cc.toxic_loss_multiplier or 2.0,
        num_classes=num_classes,
        use_asl=cc.use_asl,
        asl_gamma_pos=cc.asl_gamma_pos,
        asl_gamma_neg=cc.asl_gamma_neg,
        asl_margin=cc.asl_margin,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    history: list[EpochResult] = []
    best_val_acc = 0.0
    best_toxic_fp = float("inf")
    toxic_fp_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            if cfg.cutmix_prob > 0:
                images = _intra_class_cutmix(images, labels, toxic_indices, cfg.cutmix_prob, cfg.cutmix_alpha)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = epoch_loss / len(train_ds)

        # Validation
        val_metrics = evaluate_model(
            model, val_loader, device, num_classes, toxic_indices, species_ids
        )
        elapsed = time.time() - t0
        result = EpochResult(epoch, train_loss, val_metrics, elapsed)
        history.append(result)

        # Print progress
        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_acc={val_metrics.overall_accuracy:.3f}  "
            f"toxic_fp={val_metrics.toxic_fp_rate:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Safety alarm
        if val_metrics.toxic_fp_rate > TOXIC_FP_ALARM_THRESHOLD and val_metrics.toxic_total > 0:
            print(
                f"  ⚠ SAFETY ALARM: toxic FP rate {val_metrics.toxic_fp_rate:.3f} "
                f"> {TOXIC_FP_ALARM_THRESHOLD}"
            )

        # Checkpoint: best accuracy
        if cfg.save_best_accuracy and val_metrics.overall_accuracy > best_val_acc:
            best_val_acc = val_metrics.overall_accuracy
            _save_checkpoint(model, cfg.checkpoint_dir / "best_accuracy.pt", epoch, val_metrics)

        # Checkpoint: best toxic FP rate
        if cfg.save_best_toxic_fp:
            improved = val_metrics.toxic_fp_rate < best_toxic_fp - cfg.toxic_fp_min_delta
            if improved or val_metrics.toxic_total == 0:
                best_toxic_fp = val_metrics.toxic_fp_rate
                toxic_fp_no_improve = 0
                _save_checkpoint(
                    model, cfg.checkpoint_dir / "best_safety.pt", epoch, val_metrics
                )
            else:
                toxic_fp_no_improve += 1
                if toxic_fp_no_improve >= cfg.toxic_fp_patience and val_metrics.toxic_total > 0:
                    print(
                        f"  Early stop: toxic FP rate has not improved for "
                        f"{cfg.toxic_fp_patience} epochs."
                    )
                    break

    return history


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    metrics: SafetyMetrics,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_accuracy": metrics.overall_accuracy,
            "toxic_fp_rate": metrics.toxic_fp_rate,
        },
        path,
    )
    print(
        f"  Saved checkpoint → {path.name}  "
        f"(acc={metrics.overall_accuracy:.3f}, toxic_fp={metrics.toxic_fp_rate:.4f})"
    )
