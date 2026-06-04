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


def build_loss(
    class_weights: Optional[torch.Tensor] = None,
    toxic_indices: Optional[set[int]] = None,
    toxic_multiplier: float = 2.0,
    num_classes: int = 12,
) -> nn.CrossEntropyLoss:
    """
    Build a CrossEntropyLoss with optional toxic-class upweighting.

    Parameters
    ----------
    class_weights:
        Per-class frequency weights (e.g. from ``EdibleDataset.class_weights()``).
        If None, uniform weights are used.
    toxic_indices:
        Indices of toxic classes.  Their weights are multiplied by
        *toxic_multiplier* on top of any frequency weighting.
    toxic_multiplier:
        Additional weight for toxic classes.
    num_classes:
        Total number of classes (used only when *class_weights* is None).
    """
    if class_weights is None:
        weights = torch.ones(num_classes)
    else:
        weights = class_weights.clone().float()

    if toxic_indices:
        for idx in toxic_indices:
            weights[idx] *= toxic_multiplier

    return nn.CrossEntropyLoss(weight=weights)
