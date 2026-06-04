"""
Safety-first evaluation metrics for the Edible classifier.

Critical safety metric: **toxic false-positive rate** — how often the model
predicts an edible species when the ground truth is toxic.  This is the
most dangerous failure mode and must be tracked explicitly.

All public functions accept plain Python lists/tensors so they can be
used during training loops without pulling in heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class SafetyMetrics:
    """Aggregate safety and accuracy metrics for one evaluation pass."""

    # Per-class accuracy
    num_classes: int
    per_class_correct: list[int]
    per_class_total: list[int]

    # Safety-critical confusion counts
    toxic_total: int  # ground-truth toxic samples
    toxic_predicted_edible: int  # toxic GT → edible prediction  (CRITICAL FP)
    edible_total: int  # ground-truth edible samples
    edible_predicted_toxic: int  # edible GT → toxic prediction  (false alarm)

    # Auxiliary
    species_ids: list[str]  # maps class index → species id
    toxic_indices: set[int]  # class indices that are toxic
    raw_preds: list[int] = field(default_factory=list)
    raw_labels: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def overall_accuracy(self) -> float:
        total = sum(self.per_class_total)
        correct = sum(self.per_class_correct)
        return correct / total if total else 0.0

    @property
    def toxic_fp_rate(self) -> float:
        """
        Fraction of toxic samples misclassified as edible.

        Target: < 1 % in production.  Training should alarm if > 5 %.
        """
        return self.toxic_predicted_edible / self.toxic_total if self.toxic_total else 0.0

    @property
    def edible_fn_rate(self) -> float:
        """Fraction of edible samples misclassified as toxic (false alarm rate)."""
        return self.edible_predicted_toxic / self.edible_total if self.edible_total else 0.0

    @property
    def per_class_accuracy(self) -> list[float]:
        return [
            c / t if t else 0.0
            for c, t in zip(self.per_class_correct, self.per_class_total)
        ]

    def classification_report(self) -> str:
        if not self.raw_labels:
            return "(no samples)"
        target_names = self.species_ids
        return classification_report(
            self.raw_labels,
            self.raw_preds,
            target_names=target_names,
            zero_division=0,
        )

    def confusion_matrix(self) -> list[list[int]]:
        if not self.raw_labels:
            return []
        return confusion_matrix(
            self.raw_labels,
            self.raw_preds,
            labels=list(range(self.num_classes)),
        ).tolist()

    def safety_summary(self) -> str:
        lines = [
            f"Overall accuracy : {self.overall_accuracy:.3f}",
            f"Toxic FP rate    : {self.toxic_fp_rate:.4f}  "
            f"({self.toxic_predicted_edible}/{self.toxic_total} toxic→edible)",
            f"Edible FN rate   : {self.edible_fn_rate:.4f}  "
            f"({self.edible_predicted_toxic}/{self.edible_total} edible→toxic)",
        ]
        return "\n".join(lines)


def compute_metrics(
    preds: list[int],
    labels: list[int],
    num_classes: int,
    toxic_indices: set[int],
    species_ids: Optional[list[str]] = None,
) -> SafetyMetrics:
    """
    Compute SafetyMetrics from flat prediction/label lists.

    Parameters
    ----------
    preds:
        Predicted class indices.
    labels:
        Ground-truth class indices.
    num_classes:
        Total number of classes.
    toxic_indices:
        Set of class indices corresponding to toxic species.
    species_ids:
        Optional list mapping class index → species id string.
    """
    if len(preds) != len(labels):
        raise ValueError(f"preds length {len(preds)} != labels length {len(labels)}")

    edible_indices = set(range(num_classes)) - toxic_indices

    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    toxic_total = 0
    toxic_predicted_edible = 0
    edible_total = 0
    edible_predicted_toxic = 0

    for pred, label in zip(preds, labels):
        if not (0 <= label < num_classes):
            raise ValueError(f"Label {label} out of range [0, {num_classes})")
        if not (0 <= pred < num_classes):
            raise ValueError(f"Prediction {pred} out of range [0, {num_classes})")

        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1

        if label in toxic_indices:
            toxic_total += 1
            if pred in edible_indices:
                toxic_predicted_edible += 1
        else:
            edible_total += 1
            if pred in toxic_indices:
                edible_predicted_toxic += 1

    return SafetyMetrics(
        num_classes=num_classes,
        per_class_correct=per_class_correct,
        per_class_total=per_class_total,
        toxic_total=toxic_total,
        toxic_predicted_edible=toxic_predicted_edible,
        edible_total=edible_total,
        edible_predicted_toxic=edible_predicted_toxic,
        species_ids=species_ids or [str(i) for i in range(num_classes)],
        toxic_indices=toxic_indices,
        raw_preds=list(preds),
        raw_labels=list(labels),
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    toxic_indices: set[int],
    species_ids: Optional[list[str]] = None,
) -> SafetyMetrics:
    """
    Run *model* on *loader* and return SafetyMetrics.

    The model must output raw logits of shape ``(B, num_classes)``.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return compute_metrics(
        all_preds, all_labels, num_classes, toxic_indices, species_ids
    )
