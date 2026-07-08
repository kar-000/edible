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

from edible.model.classifier import (
    ClassifierConfig,
    ProjectionHead,
    SupConLoss,
    build_classifier,
    build_loss,
)
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

    # Balanced batch sampling via WeightedRandomSampler
    balanced_sampling: bool = False

    # Hard negative mining: map image path → boost multiplier applied to sample weight.
    # Works independently of balanced_sampling (activates sampler on its own if non-empty).
    hard_negatives: dict = field(default_factory=dict)  # Path → float

    # SupCon warm-start: path to supcon_backbone.pt produced by pretrain_supcon().
    # When set, backbone weights are replaced after build_classifier() is called,
    # giving the fine-tuning loop a contrastively pre-trained feature extractor.
    pretrained_backbone_path: Optional[Path] = None

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

    class_weights_cpu = train_ds.class_weights()
    use_sampler = cfg.balanced_sampling or bool(cfg.hard_negatives)
    if use_sampler:
        base = class_weights_cpu if cfg.balanced_sampling else torch.ones(len(class_weights_cpu))
        sample_weights = [
            base[s.class_idx].item() * cfg.hard_negatives.get(s.image_path, 1.0)
            for s in train_ds.samples
        ]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
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

    # Warm-start backbone from SupCon pre-training (Phase 1 → Phase 2)
    if cfg.pretrained_backbone_path and cfg.pretrained_backbone_path.exists():
        ckpt = torch.load(
            cfg.pretrained_backbone_path, map_location=device, weights_only=True
        )
        model.backbone.load_state_dict(ckpt["backbone_state_dict"])
        print(f"Loaded SupCon backbone from {cfg.pretrained_backbone_path.name}")

    cc = cfg.classifier_config
    criterion = build_loss(
        class_weights=class_weights_cpu.to(device),
        toxic_indices=toxic_indices,
        toxic_multiplier=cc.toxic_loss_multiplier or 2.0,
        num_classes=num_classes,
        use_asl=cc.use_asl,
        asl_gamma_pos=cc.asl_gamma_pos,
        asl_gamma_neg=cc.asl_gamma_neg,
        asl_margin=cc.asl_margin,
        label_smoothing=cc.label_smoothing,
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
                images = _intra_class_cutmix(
                    images, labels, toxic_indices, cfg.cutmix_prob, cfg.cutmix_alpha
                )
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


# ---------------------------------------------------------------------------
# SupCon pre-training
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    """Hyper-parameters for SupCon backbone pre-training (Phase 1)."""

    # Data (must point at the same image tree as TrainConfig)
    images_dir: Path = Path("data/images")
    species_db_path: Path = Path("data/species.json")

    # Model
    classifier_config: ClassifierConfig = field(default_factory=ClassifierConfig)
    projection_hidden_dim: int = 256
    projection_out_dim: int = 128

    # SupCon loss
    supcon_temperature: float = 0.07
    toxic_multiplier: float = 2.0  # extra weight on toxic-anchor pairs

    # Training
    epochs: int = 20
    batch_size: int = 64   # larger batch = more contrastive pairs per step
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0

    # Output
    checkpoint_dir: Path = Path("checkpoints")

    # Device
    device: Optional[torch.device] = None


def pretrain_supcon(cfg: PretrainConfig) -> Path:
    """
    Phase 1: pre-train the EfficientNet-B0 backbone with SupCon loss.

    The backbone + projection head are trained on the TRAIN split only.
    After training, backbone weights are saved to
    ``<checkpoint_dir>/supcon_backbone.pt``.

    Returns
    -------
    Path to the saved backbone checkpoint.
    """
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device or (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # Dataset (train split only; augmentation helps contrastive learning)
    train_ds = EdibleDataset(cfg.images_dir, cfg.species_db_path, split="train")
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,   # SupCon needs full batches for pair formation
    )

    # Build backbone (no classification head)
    import timm
    backbone = timm.create_model(
        cfg.classifier_config.model_name,
        pretrained=cfg.classifier_config.pretrained,
        num_classes=0,
    ).to(device)
    num_features = backbone.num_features

    proj_head = ProjectionHead(
        num_features=num_features,
        hidden_dim=cfg.projection_hidden_dim,
        out_dim=cfg.projection_out_dim,
    ).to(device)

    # Identify toxic class indices
    from edible.data.schemas import Edibility, load_species_db
    species_db = load_species_db(cfg.species_db_path)
    sorted_ids = sorted(s.id for s in species_db.species)
    toxic_indices = {
        i for i, sid in enumerate(sorted_ids)
        if species_db.get_by_id(sid).edibility == Edibility.TOXIC
    }

    criterion = SupConLoss(
        temperature=cfg.supcon_temperature,
        toxic_indices=toxic_indices,
        toxic_multiplier=cfg.toxic_multiplier,
    )

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(proj_head.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    print(f"SupCon pre-training: {cfg.epochs} epochs  |  device={device}")
    print(f"  Batch size: {cfg.batch_size}  |  τ={cfg.supcon_temperature}")
    print(f"  Toxic classes: {len(toxic_indices)}  |  toxic_mult={cfg.toxic_multiplier}×")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        backbone.train()
        proj_head.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            features = backbone(images)           # (B, num_features)
            projections = proj_head(features)     # (B, out_dim) — L2-normalised

            loss = criterion(projections, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"loss={epoch_loss / n_batches:.4f}  "
            f"({elapsed:.1f}s)"
        )

    # Save backbone weights only (projection head is discarded)
    out_path = cfg.checkpoint_dir / "supcon_backbone.pt"
    torch.save({"backbone_state_dict": backbone.state_dict()}, out_path)
    print(f"\nSaved backbone → {out_path}")
    return out_path
