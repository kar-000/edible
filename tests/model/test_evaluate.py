"""
Tests for the safety-first evaluation metrics module.

Adversarial focus: toxic FP rate must be computed correctly even in
degenerate edge cases (all toxic, all edible, empty dataset, etc.).
"""

from __future__ import annotations

import pytest

from edible.model.evaluate import SafetyMetrics, compute_metrics

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _simple_metrics(
    preds: list[int],
    labels: list[int],
    num_classes: int = 4,
    toxic_indices: set[int] | None = None,
    species_ids: list[str] | None = None,
) -> SafetyMetrics:
    if toxic_indices is None:
        toxic_indices = {0, 1}  # first two classes are toxic by default
    return compute_metrics(preds, labels, num_classes, toxic_indices, species_ids)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestComputeMetricsBasic:
    def test_perfect_predictions(self):
        labels = [0, 1, 2, 3]
        m = _simple_metrics(labels, labels)
        assert m.overall_accuracy == 1.0
        assert m.toxic_fp_rate == 0.0

    def test_all_wrong(self):
        preds =  [1, 0, 3, 2]
        labels = [0, 1, 2, 3]
        m = _simple_metrics(preds, labels)
        assert m.overall_accuracy == 0.0

    def test_per_class_accuracy(self):
        # class 0: 2 correct, 0 wrong; class 1: 0 correct, 1 wrong
        preds =  [0, 0, 1]
        labels = [0, 0, 0]
        m = compute_metrics(preds, labels, 2, toxic_indices={1})
        assert m.per_class_accuracy[0] == pytest.approx(2 / 3)

    def test_overall_accuracy_fraction(self):
        preds =  [0, 0, 1, 1]
        labels = [0, 1, 0, 1]
        m = _simple_metrics(preds, labels, num_classes=2, toxic_indices={0})
        assert m.overall_accuracy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Safety-critical: toxic FP rate
# ---------------------------------------------------------------------------

class TestToxicFPRate:
    def test_toxic_fp_rate_is_zero_when_all_correct(self):
        labels = [0, 0, 2, 2]  # 0=toxic, 2=edible
        m = compute_metrics(labels, labels, 4, toxic_indices={0, 1})
        assert m.toxic_fp_rate == 0.0

    def test_toxic_fp_rate_is_one_when_all_toxic_predicted_edible(self):
        # Two toxic samples (label 0), both predicted as edible (pred 2)
        preds  = [2, 2]
        labels = [0, 0]
        m = compute_metrics(preds, labels, 4, toxic_indices={0, 1})
        assert m.toxic_fp_rate == 1.0

    def test_toxic_fp_rate_partial(self):
        # 4 toxic, 1 predicted edible → rate = 0.25
        preds  = [2, 0, 0, 0]   # first is wrong (edible)
        labels = [0, 0, 0, 0]   # all toxic
        m = compute_metrics(preds, labels, 4, toxic_indices={0, 1})
        assert m.toxic_fp_rate == pytest.approx(0.25)

    def test_toxic_fp_rate_zero_when_no_toxic_samples(self):
        # No toxic samples in this batch → rate = 0 (not div/0)
        preds  = [2, 3]
        labels = [2, 3]
        m = compute_metrics(preds, labels, 4, toxic_indices={0, 1})
        assert m.toxic_fp_rate == 0.0
        assert m.toxic_total == 0

    def test_toxic_to_toxic_misclassification_not_counted_as_fp(self):
        # toxic → another toxic class: bad but NOT a toxic-to-edible FP
        preds  = [1]   # predicted toxic class 1
        labels = [0]   # ground truth toxic class 0
        m = compute_metrics(preds, labels, 4, toxic_indices={0, 1})
        assert m.toxic_fp_rate == 0.0


# ---------------------------------------------------------------------------
# Edible false alarm rate
# ---------------------------------------------------------------------------

class TestEdibleFNRate:
    def test_no_false_alarms(self):
        preds  = [2, 3]
        labels = [2, 3]
        m = compute_metrics(preds, labels, 4, toxic_indices={0, 1})
        assert m.edible_fn_rate == 0.0

    def test_all_edible_predicted_toxic(self):
        preds  = [0, 0]
        labels = [2, 3]
        m = compute_metrics(preds, labels, 4, toxic_indices={0, 1})
        assert m.edible_fn_rate == 1.0

    def test_no_edible_samples(self):
        preds  = [0, 1]
        labels = [0, 1]
        m = compute_metrics(preds, labels, 4, toxic_indices={0, 1})
        assert m.edible_fn_rate == 0.0
        assert m.edible_total == 0


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_inputs_return_zero_metrics(self):
        m = compute_metrics([], [], 4, toxic_indices={0})
        assert m.overall_accuracy == 0.0
        assert m.toxic_fp_rate == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length"):
            compute_metrics([0, 1], [0], 2, toxic_indices={0})

    def test_label_out_of_range_raises(self):
        with pytest.raises(ValueError, match="range"):
            compute_metrics([0], [5], num_classes=3, toxic_indices={0})

    def test_pred_out_of_range_raises(self):
        with pytest.raises(ValueError, match="range"):
            compute_metrics([5], [0], num_classes=3, toxic_indices={0})

    def test_single_sample_correct(self):
        m = compute_metrics([2], [2], 4, toxic_indices={0, 1})
        assert m.overall_accuracy == 1.0

    def test_species_ids_used_in_report(self):
        labels = [0, 1]
        m = compute_metrics(
            labels, labels, 2,
            toxic_indices={0},
            species_ids=["toxic_berry", "safe_berry"],
        )
        report = m.classification_report()
        assert "toxic_berry" in report
        assert "safe_berry" in report

    def test_classification_report_empty(self):
        m = compute_metrics([], [], 2, toxic_indices={0})
        assert "(no samples)" in m.classification_report()

    def test_confusion_matrix_empty(self):
        m = compute_metrics([], [], 2, toxic_indices={0})
        assert m.confusion_matrix() == []

    def test_confusion_matrix_shape(self):
        labels = [0, 1, 2, 3]
        m = compute_metrics(labels, labels, 4, toxic_indices={0, 1})
        cm = m.confusion_matrix()
        assert len(cm) == 4
        assert all(len(row) == 4 for row in cm)

    def test_safety_summary_contains_rates(self):
        labels = [0, 2]
        preds  = [2, 2]  # toxic (0) → edible (2): FP
        m = compute_metrics(preds, labels, 4, toxic_indices={0, 1})
        summary = m.safety_summary()
        assert "toxic FP" in summary.lower() or "Toxic FP" in summary


# ---------------------------------------------------------------------------
# Adversarial: all-toxic dataset
# ---------------------------------------------------------------------------

class TestAllToxicDataset:
    def test_all_correct(self):
        labels = [0] * 10
        m = compute_metrics(labels, labels, 2, toxic_indices={0})
        assert m.toxic_fp_rate == 0.0
        assert m.overall_accuracy == 1.0

    def test_all_misclassified_as_edible(self):
        preds  = [1] * 10
        labels = [0] * 10
        m = compute_metrics(preds, labels, 2, toxic_indices={0})
        assert m.toxic_fp_rate == 1.0


# ---------------------------------------------------------------------------
# SafetyMetrics properties
# ---------------------------------------------------------------------------

class TestSafetyMetricsProperties:
    def _make(self, preds, labels):
        return compute_metrics(preds, labels, 4, toxic_indices={0, 1})

    def test_per_class_accuracy_length(self):
        m = self._make([0, 1, 2, 3], [0, 1, 2, 3])
        assert len(m.per_class_accuracy) == 4

    def test_per_class_accuracy_correct_values(self):
        m = self._make([0, 1, 2, 3], [0, 1, 2, 3])
        assert all(a == 1.0 for a in m.per_class_accuracy)

    def test_per_class_accuracy_zero_for_empty_class(self):
        # Only labels 0 and 1 present; classes 2 and 3 have 0 samples
        m = self._make([0, 1], [0, 1])
        assert m.per_class_accuracy[2] == 0.0
        assert m.per_class_accuracy[3] == 0.0
