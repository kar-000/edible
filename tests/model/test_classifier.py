"""
Tests for the EdibleClassifier and build_* helpers.

Tests are CPU-only and use tiny synthetic models to avoid downloading
pretrained weights in CI.  The real timm backbone is tested via a single
smoke-test that uses ``pretrained=False``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from edible.data.pipeline import ClassifierResult
from edible.data.schemas import Edibility
from edible.model.classifier import (
    ClassifierConfig,
    EdibleClassifier,
    build_classifier,
    build_loss,
)

# ---------------------------------------------------------------------------
# Minimal stub backbone (no pretrained download needed)
# ---------------------------------------------------------------------------

class _TinyBackbone(nn.Module):
    """Outputs a (B, 16) feature vector regardless of spatial input size."""

    def __init__(self, num_features: int = 16):
        super().__init__()
        self.num_features = num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(3, num_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x)).flatten(1)


def _make_classifier(num_classes: int = 4, num_features: int = 16) -> EdibleClassifier:
    return EdibleClassifier(
        backbone=_TinyBackbone(num_features),
        num_classes=num_classes,
        num_features=num_features,
        dropout=0.0,
        confidence_threshold=0.75,
    )


# ---------------------------------------------------------------------------
# EdibleClassifier.forward
# ---------------------------------------------------------------------------

class TestEdibleClassifierForward:
    def test_output_shape_single(self):
        model = _make_classifier(num_classes=4)
        x = torch.randn(1, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (1, 4)

    def test_output_shape_batch(self):
        model = _make_classifier(num_classes=6)
        x = torch.randn(8, 3, 64, 64)
        logits = model(x)
        assert logits.shape == (8, 6)

    def test_output_is_raw_logits_not_probabilities(self):
        model = _make_classifier(num_classes=4)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        # Softmax of logits should sum to 1, but raw logits generally don't
        row_sums = logits.sum(dim=1)
        # Raw logits rarely sum exactly to 1 for all rows
        assert not all(abs(float(s) - 1.0) < 1e-5 for s in row_sums)

    def test_dropout_disabled_at_eval(self):
        model = _make_classifier(num_classes=4)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# EdibleClassifier.predict
# ---------------------------------------------------------------------------

class TestEdibleClassifierPredict:
    def setup_method(self):
        torch.manual_seed(42)
        self.model = _make_classifier(num_classes=4)
        self.species_ids = ["alpha", "beta", "gamma", "delta"]
        self.edibility_map = {
            "alpha": Edibility.TOXIC,
            "beta": Edibility.TOXIC,
            "gamma": Edibility.EDIBLE_RAW,
            "delta": Edibility.EDIBLE_RAW,
        }

    def test_returns_classifier_result(self):
        x = torch.randn(3, 32, 32)
        result = self.model.predict(x, self.species_ids, self.edibility_map)
        assert isinstance(result, ClassifierResult)

    def test_predictions_length_equals_num_classes(self):
        x = torch.randn(3, 32, 32)
        result = self.model.predict(x, self.species_ids, self.edibility_map)
        assert len(result.predictions) == 4

    def test_predictions_sorted_descending(self):
        x = torch.randn(3, 32, 32)
        result = self.model.predict(x, self.species_ids, self.edibility_map)
        confs = [p.confidence for p in result.predictions]
        assert confs == sorted(confs, reverse=True)

    def test_top_prediction_species_id_valid(self):
        x = torch.randn(3, 32, 32)
        result = self.model.predict(x, self.species_ids, self.edibility_map)
        assert result.predictions[0].species_id in self.species_ids

    def test_edibility_set_correctly(self):
        x = torch.randn(3, 32, 32)
        result = self.model.predict(x, self.species_ids, self.edibility_map)
        for pred in result.predictions:
            assert pred.edibility == self.edibility_map[pred.species_id]

    def test_predict_accepts_3d_input(self):
        x = torch.randn(3, 32, 32)
        result = self.model.predict(x, self.species_ids, self.edibility_map)
        assert len(result.predictions) == 4

    def test_confidence_threshold_stored_on_model(self):
        model = EdibleClassifier(
            backbone=_TinyBackbone(),
            num_classes=4,
            num_features=16,
            confidence_threshold=0.80,
        )
        assert model.confidence_threshold == 0.80

    def test_check_confidence_passes_when_above_threshold(self):
        # Force model to output very high confidence for one class
        class _HighConfModel(nn.Module):
            def forward(self, x):
                logits = torch.full((x.shape[0], 4), -100.0)
                logits[:, 0] = 100.0  # class 0 ≈ 1.0 probability
                return logits

        model = EdibleClassifier(
            backbone=_HighConfModel(),
            num_classes=4,
            num_features=16,
            dropout=0.0,
            confidence_threshold=0.75,
        )
        # Replace head to be identity (backbone already outputs logits)
        model.head = nn.Identity()
        x = torch.randn(3, 32, 32)
        clf_result = model.predict(x, self.species_ids, self.edibility_map)
        conf_result = model.check_confidence(clf_result)
        assert conf_result.passes is True


# ---------------------------------------------------------------------------
# build_classifier (smoke test with pretrained=False)
# ---------------------------------------------------------------------------

class TestBuildClassifier:
    def test_builds_efficientnet_b0(self):
        cfg = ClassifierConfig(
            model_name="efficientnet_b0",
            pretrained=False,
            num_classes=12,
        )
        model = build_classifier(cfg)
        assert isinstance(model, EdibleClassifier)
        assert model.num_classes == 12

    def test_output_shape_matches_num_classes(self):
        cfg = ClassifierConfig(
            model_name="efficientnet_b0",
            pretrained=False,
            num_classes=7,
        )
        model = build_classifier(cfg)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, 7)

    def test_head_is_linear(self):
        cfg = ClassifierConfig(pretrained=False, num_classes=12)
        model = build_classifier(cfg)
        assert isinstance(model.head, nn.Linear)
        assert model.head.out_features == 12


# ---------------------------------------------------------------------------
# build_loss
# ---------------------------------------------------------------------------

class TestBuildLoss:
    def test_default_uniform_weights(self):
        loss_fn = build_loss(num_classes=4)
        assert loss_fn.weight is not None
        # All weights equal
        assert torch.allclose(loss_fn.weight, loss_fn.weight[0].expand(4))

    def test_toxic_indices_upweighted(self):
        loss_fn = build_loss(
            num_classes=4,
            toxic_indices={0, 1},
            toxic_multiplier=3.0,
        )
        # Toxic classes (0, 1) should have 3× the weight of edible (2, 3)
        w = loss_fn.weight
        assert float(w[0]) == pytest.approx(float(w[2]) * 3.0)
        assert float(w[1]) == pytest.approx(float(w[3]) * 3.0)

    def test_class_weights_respected(self):
        freq_weights = torch.tensor([1.0, 2.0, 1.0, 2.0])
        loss_fn = build_loss(class_weights=freq_weights, num_classes=4)
        assert torch.allclose(loss_fn.weight, freq_weights)

    def test_loss_is_computable(self):
        loss_fn = build_loss(
            num_classes=4,
            toxic_indices={0, 1},
            toxic_multiplier=2.0,
        )
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        loss_val = loss_fn(logits, labels)
        assert loss_val.item() > 0

    def test_no_toxic_indices(self):
        loss_fn = build_loss(num_classes=3, toxic_indices=set())
        # No upweighting — weights from class_weights or uniform
        assert loss_fn.weight is not None

    def test_empty_toxic_indices_no_upweighting(self):
        uniform = build_loss(num_classes=4, toxic_indices=None)
        none_toxic = build_loss(num_classes=4, toxic_indices=set())
        # Both should be uniform
        assert torch.allclose(uniform.weight, none_toxic.weight)
