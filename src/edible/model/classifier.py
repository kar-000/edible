"""
Layer 2 — Species classifier (EfficientNet-B0 fine-tuning wrapper).

Wraps a timm EfficientNet-B0 (or any timm model) and adds:
- Configurable classification head replacement
- Toxic-class weighted loss helper
- Forward pass that returns raw logits
- Confidence-gated prediction (Layer 2 confidence floor)

The model is NOT tied to a specific number of classes at construction;
``build_classifier()`` is the entry point that wires everything together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from edible.data.pipeline import ClassifierResult, ConfidenceCheckResult, SpeciesPrediction
from edible.data.schemas import Edibility

# Default confidence floor from Addendum A §3.2
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class ClassifierConfig:
    """Hyper-parameters that define a classifier build."""

    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 12
    dropout: float = 0.3
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD

    # Toxic-aware loss: if set, the weight applied to toxic classes in
    # CrossEntropyLoss.  None means use class-frequency weights from dataset.
    toxic_loss_multiplier: Optional[float] = 2.0

    # Asymmetric Loss — replaces weighted CrossEntropy when True.
    use_asl: bool = False
    asl_gamma_pos: float = 1.0   # focusing strength for the true class (mild)
    asl_gamma_neg: float = 4.0   # focusing strength for false classes (strong)
    asl_margin: float = 0.05     # probability shift: clips easy negatives to 0

    # Label smoothing — applied to both CE and ASL paths.
    label_smoothing: float = 0.0  # 0 = disabled; 0.1 is a good starting point


class EdibleClassifier(nn.Module):
    """
    Fine-tuned species classifier.

    Parameters
    ----------
    backbone:
        A timm model with ``num_features`` attribute (or any nn.Module
        whose output is a flat feature vector).
    num_classes:
        Number of target species.
    dropout:
        Dropout rate applied before the classification head.
    confidence_threshold:
        Minimum softmax score for the top class to be accepted.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_features: int,
        dropout: float = 0.3,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(num_features, num_classes)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold  # used by check_confidence()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits, shape ``(B, num_classes)``."""
        features = self.backbone(x)  # (B, num_features)
        features = self.dropout(features)
        return self.head(features)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        species_ids: list[str],
        edibility_map: dict[str, Edibility],
    ) -> ClassifierResult:
        """
        Run inference on a single image tensor (C, H, W) and return a
        ``ClassifierResult`` including confidence-floor enforcement.

        Parameters
        ----------
        x:
            Single image tensor, shape ``(C, H, W)``.
        species_ids:
            Ordered list of species IDs matching class indices 0, 1, …
        edibility_map:
            Maps each species_id → its ``Edibility`` value.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, C, H, W)

        logits = self.forward(x)  # (1, num_classes)
        probs = F.softmax(logits, dim=1)[0]  # (num_classes,)

        sorted_indices = torch.argsort(probs, descending=True)
        top_predictions = [
            SpeciesPrediction(
                species_id=species_ids[idx],
                confidence=float(probs[idx]),
                edibility=edibility_map[species_ids[idx]],
            )
            for idx in sorted_indices.tolist()
        ]

        return ClassifierResult(
            predictions=top_predictions,
            top_prediction=top_predictions[0],
        )

    def check_confidence(self, result: ClassifierResult) -> ConfidenceCheckResult:
        """
        Apply the Layer 2 confidence floor to a ``ClassifierResult``.

        Returns a ``ConfidenceCheckResult`` indicating whether the top
        prediction meets the threshold.
        """
        top_conf = result.top_prediction.confidence
        passes = top_conf >= self.confidence_threshold
        thr = self.confidence_threshold
        reason = "" if passes else f"confidence {top_conf:.3f} < threshold {thr:.3f}"
        return ConfidenceCheckResult(
            passes=passes,
            confidence=top_conf,
            threshold_used=thr,
            reason=reason,
        )


def build_classifier(config: ClassifierConfig) -> EdibleClassifier:
    """
    Construct an ``EdibleClassifier`` from *config*.

    The backbone's original classification head is removed; a fresh
    dropout + linear head for *num_classes* is attached.
    """
    backbone = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=0,  # remove classifier head
    )
    num_features = backbone.num_features
    return EdibleClassifier(
        backbone=backbone,
        num_classes=config.num_classes,
        num_features=num_features,
        dropout=config.dropout,
        confidence_threshold=config.confidence_threshold,
    )


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) adapted for multi-class single-label classification.

    Treats each class as an independent sigmoid binary output (one-vs-rest),
    then applies different focal weights for the true class vs. all others:

      - True class  (positive): -(1 - p)^γ+  · log(p)
        γ+ is kept small (0–1) so the gradient on hard correct examples
        is not suppressed too aggressively.

      - False classes (negatives): -(p_shifted)^γ-  · log(1 - p_shifted)
        where p_shifted = max(p - margin, 0).
        The margin clips very-low-probability negatives to zero (ignores
        easy negatives entirely).  γ- is large (2–4) to down-weight the
        remaining easy negatives, forcing the model to focus on hard cases.

    An optional per-sample toxic multiplier is applied on top: samples whose
    ground-truth class is toxic have their loss scaled up, preserving the
    safety-critical weighting from the original CrossEntropyLoss design.
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        margin: float = 0.05,
        toxic_indices: Optional[set[int]] = None,
        toxic_multiplier: float = 3.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.toxic_indices = toxic_indices or set()
        self.toxic_multiplier = toxic_multiplier
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape

        # One-hot targets: shape (B, C), optionally smoothed
        targets = F.one_hot(labels, C).float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / C

        # Per-class sigmoid probability (independent binary treatment)
        probs = torch.sigmoid(logits)

        # Shift negatives: clips easy negatives to 0
        probs_neg = (probs - self.margin).clamp(min=0.0)

        # Blend: use original p for positives, shifted p for negatives
        probs_blended = targets * probs + (1.0 - targets) * probs_neg

        # Log terms (clamped for numerical stability)
        eps = 1e-8
        log_p = torch.log(probs_blended.clamp(min=eps))
        log_1mp = torch.log((1.0 - probs_blended).clamp(min=eps))

        # Asymmetric focal weights
        w_pos = (1.0 - probs).pow(self.gamma_pos)   # down-weights easy positives mildly
        w_neg = probs_blended.pow(self.gamma_neg)    # down-weights easy negatives strongly

        focal = targets * w_pos + (1.0 - targets) * w_neg

        # Per-sample loss: sum over classes
        sample_loss = -(focal * (targets * log_p + (1.0 - targets) * log_1mp)).sum(dim=1)

        # Toxic class upweighting (per sample)
        if self.toxic_indices:
            is_toxic = torch.tensor(
                [int(lbl) in self.toxic_indices for lbl in labels.tolist()],
                device=logits.device,
                dtype=torch.float32,
            )
            sample_loss = sample_loss * (1.0 + (self.toxic_multiplier - 1.0) * is_toxic)

        return sample_loss.mean()


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head for SupCon pre-training.

    Maps backbone features → a unit-norm embedding in a low-dimensional
    space where the contrastive loss operates.  The head is discarded after
    pre-training; only the backbone weights are kept.

    Architecture: Linear(num_features, hidden_dim) → BN → ReLU
                  → Linear(hidden_dim, out_dim) → L2-normalise
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projections, shape ``(B, out_dim)``."""
        return F.normalize(self.net(x), dim=1)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For a batch of embeddings, pulls same-class embeddings together while
    pushing all other-class embeddings away.

    Parameters
    ----------
    temperature:
        Logit scale divisor τ.  Smaller τ sharpens the distribution; the
        original paper uses 0.07 for ImageNet-scale work.
    toxic_indices:
        If set, toxic-class anchors receive additional loss weight
        (``toxic_multiplier``), amplifying the safety-critical contrastive signal.
    toxic_multiplier:
        Scale factor applied to toxic-anchor loss terms.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        toxic_indices: Optional[set[int]] = None,
        toxic_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.toxic_indices = toxic_indices or set()
        self.toxic_multiplier = toxic_multiplier

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            L2-normalised embeddings, shape ``(B, D)``.
        labels:
            Integer class labels, shape ``(B,)``.

        Returns
        -------
        Scalar loss.
        """
        B = features.shape[0]
        device = features.device

        # Cosine similarity matrix (B, B)
        sim = torch.matmul(features, features.T) / self.temperature

        # Positive mask: same-class pairs, excluding self
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        pos_mask = (labels_col == labels_row).float()
        pos_mask.fill_diagonal_(0.0)

        # Exclude self from softmax denominator
        self_mask = torch.eye(B, device=device)
        sim_no_self = sim - 1e9 * self_mask

        # log-softmax over all non-self entries, then weight by positives
        log_prob = sim_no_self - torch.logsumexp(sim_no_self, dim=1, keepdim=True)
        num_pos = pos_mask.sum(dim=1).clamp(min=1.0)
        per_anchor = -(pos_mask * log_prob).sum(dim=1) / num_pos  # (B,)

        # Toxic-anchor upweighting
        if self.toxic_indices:
            is_toxic = torch.tensor(
                [int(lbl) in self.toxic_indices for lbl in labels.tolist()],
                device=device, dtype=torch.float32,
            )
            per_anchor = per_anchor * (1.0 + (self.toxic_multiplier - 1.0) * is_toxic)

        # Anchors with no in-batch positives contribute zero loss
        has_pos = (pos_mask.sum(dim=1) > 0).float()
        return (per_anchor * has_pos).sum() / has_pos.sum().clamp(min=1.0)


def build_loss(
    class_weights: Optional[torch.Tensor] = None,
    toxic_indices: Optional[set[int]] = None,
    toxic_multiplier: float = 2.0,
    num_classes: int = 12,
    use_asl: bool = False,
    asl_gamma_pos: float = 1.0,
    asl_gamma_neg: float = 4.0,
    asl_margin: float = 0.05,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Build a CrossEntropyLoss with optional toxic-class upweighting.

    Parameters
    ----------
    class_weights:
        Per-class frequency weights (e.g. from ``EdibleDataset.class_weights()``).
        Used only for weighted CrossEntropy (ignored when use_asl=True).
    toxic_indices:
        Indices of toxic classes.
    toxic_multiplier:
        Additional per-sample weight for toxic-class training examples.
    num_classes:
        Total number of classes (used only when class_weights is None and use_asl=False).
    use_asl:
        If True, return AsymmetricLoss instead of weighted CrossEntropyLoss.
    asl_gamma_pos, asl_gamma_neg, asl_margin:
        ASL hyper-parameters (ignored when use_asl=False).
    """
    if use_asl:
        return AsymmetricLoss(
            gamma_pos=asl_gamma_pos,
            gamma_neg=asl_gamma_neg,
            margin=asl_margin,
            toxic_indices=toxic_indices,
            toxic_multiplier=toxic_multiplier,
            label_smoothing=label_smoothing,
        )

    if class_weights is None:
        weights = torch.ones(num_classes)
    else:
        weights = class_weights.clone().float()

    if toxic_indices:
        for idx in toxic_indices:
            weights[idx] *= toxic_multiplier

    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
