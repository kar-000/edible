"""
Tests for the PlantGate (Layer 1 inference gate).

Uses a stub model so no pretrained weights are downloaded in CI.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from edible.data.pipeline import GateDecision, GateResult
from edible.model.gate import (
    CONSERVATIVE_PLANT_INDICES,
    PlantGate,
)

# ---------------------------------------------------------------------------
# Stub ImageNet model
# ---------------------------------------------------------------------------

class _FixedOutputModel(nn.Module):
    """Returns a fixed probability distribution over 1000 ImageNet classes."""

    def __init__(self, high_classes: list[int], n_classes: int = 1000):
        super().__init__()
        self.high_classes = high_classes
        self.n_classes = n_classes
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        logits = torch.full((b, self.n_classes), -10.0)
        for cls in self.high_classes:
            logits[:, cls] = 10.0  # very high logit → high probability after softmax
        return logits


def _make_gate(high_classes: list[int], **kwargs) -> PlantGate:
    """Build a PlantGate backed by a stub model."""
    model = _FixedOutputModel(high_classes)
    return PlantGate(model=model, plant_indices=CONSERVATIVE_PLANT_INDICES, **kwargs)


# ---------------------------------------------------------------------------
# GateResult returned for plant images
# ---------------------------------------------------------------------------

class TestPlantGatePlantImage:
    def test_plant_class_passes_gate(self):
        # Pick a known plant class from CONSERVATIVE_PLANT_INDICES
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert isinstance(result, GateResult)
        assert result.decision == GateDecision.PASS

    def test_plant_score_is_high_for_plant(self):
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.plant_score > 0.5

    def test_top_imagenet_classes_not_in_result(self):
        # GateResult doesn't expose top_imagenet_classes — decision + reason is enough
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert not hasattr(result, "top_imagenet_classes")

    def test_gate_type_is_imagenet_category(self):
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.gate_type == "imagenet_category"

    def test_reason_non_empty_for_pass(self):
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.reason.strip()


# ---------------------------------------------------------------------------
# GateResult returned for non-plant images
# ---------------------------------------------------------------------------

class TestPlantGateNonPlantImage:
    def test_non_plant_class_rejects(self):
        # Class 0 (tench, a fish) is NOT in CONSERVATIVE_PLANT_INDICES
        gate = _make_gate([0, 1, 2, 3])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.decision == GateDecision.REJECT

    def test_non_plant_score_is_low(self):
        gate = _make_gate([0, 1, 2])  # non-plant classes
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.plant_score < 0.5

    def test_gate_result_not_plant_has_reject_decision(self):
        gate = _make_gate([0])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.decision == GateDecision.REJECT

    def test_reason_non_empty_for_reject(self):
        gate = _make_gate([0])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.reason.strip()


# ---------------------------------------------------------------------------
# Fail-safe: plant_score < 0.5 must never produce PASS
# ---------------------------------------------------------------------------

class TestGateFailSafe:
    def test_low_plant_score_forces_reject(self):
        # Mix of non-plant classes → plant_score < 0.5
        gate = _make_gate([0, 1, 2, 3, 4])
        x = torch.randn(3, 224, 224)
        result = gate(x)
        # plant_score < 0.5 → must be REJECT (fail-safe enforced by schema)
        if result.plant_score < 0.5:
            assert result.decision == GateDecision.REJECT

    def test_gate_result_schema_rejects_pass_with_low_score(self):
        # Directly creating an invalid GateResult should raise
        with pytest.raises(Exception):
            GateResult(
                decision=GateDecision.PASS,
                plant_score=0.3,
                reason="bad",
                gate_type="test",
            )


# ---------------------------------------------------------------------------
# Threshold enforcement
# ---------------------------------------------------------------------------

class TestGateThreshold:
    def test_high_threshold_forces_reject(self):
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))

        class _WeakModel(nn.Module):
            def forward(self, x):
                b = x.shape[0]
                logits = torch.zeros(b, 1000)
                logits[:, plant_cls] = 0.01  # very weak signal
                return logits

        gate = PlantGate(
            model=_WeakModel(),
            plant_indices=CONSERVATIVE_PLANT_INDICES,
            threshold=0.5,
            top_k=5,
        )
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.decision == GateDecision.REJECT

    def test_zero_threshold_allows_plant_to_pass(self):
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls], threshold=0.0)
        x = torch.randn(3, 224, 224)
        result = gate(x)
        assert result.decision == GateDecision.PASS


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

class TestPlantGateBatch:
    def test_batch_returns_list_of_results(self):
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls])
        batch = torch.randn(4, 3, 224, 224)
        results = gate.batch_call(batch)
        assert len(results) == 4
        assert all(isinstance(r, GateResult) for r in results)

    def test_batch_single_image_returns_one_result(self):
        gate = _make_gate([0])
        batch = torch.randn(1, 3, 224, 224)
        results = gate.batch_call(batch)
        assert len(results) == 1

    def test_batch_all_plant_returns_all_pass(self):
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls])
        batch = torch.randn(3, 3, 224, 224)
        results = gate.batch_call(batch)
        assert all(r.decision == GateDecision.PASS for r in results)

    def test_batch_no_plant_returns_all_reject(self):
        gate = _make_gate([0, 1, 2])  # non-plant
        batch = torch.randn(3, 3, 224, 224)
        results = gate.batch_call(batch)
        assert all(r.decision == GateDecision.REJECT for r in results)

    def test_batch_3d_input_promoted(self):
        plant_cls = next(iter(CONSERVATIVE_PLANT_INDICES))
        gate = _make_gate([plant_cls])
        single = torch.randn(3, 224, 224)
        results = gate.batch_call(single)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------

class TestPlantGateDevice:
    def test_default_device_is_cpu(self):
        gate = _make_gate([0])
        assert gate.device == torch.device("cpu")

    def test_explicit_device_accepted(self):
        gate = _make_gate([0], device=torch.device("cpu"))
        assert gate.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Conservative plant indices sanity
# ---------------------------------------------------------------------------

class TestConservativePlantIndices:
    def test_indices_are_valid_imagenet_range(self):
        assert all(0 <= i < 1000 for i in CONSERVATIVE_PLANT_INDICES)

    def test_indices_non_empty(self):
        assert len(CONSERVATIVE_PLANT_INDICES) > 10

    def test_known_fruit_class_included(self):
        # Class 948 is "Granny Smith" (apple) — should be plant-like
        assert 948 in CONSERVATIVE_PLANT_INDICES
