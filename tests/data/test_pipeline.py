"""
Adversarial tests for the inference pipeline schemas (Addendum A).

Focused on safety invariants:
- Gate fail-safe: uncertain → REJECT, never PASS
- Layer 2 fail-safe: confidence < threshold must not pass
- Rejected results must never carry an edibility verdict
- Accepted results must carry all required fields
- do-not-eat banner logic covers toxic, low-confidence, and rejected
- LookAlikeWarning surfaced correctly from pair data
"""

import pytest
from pydantic import ValidationError

from edible.data.pipeline import (
    CONFIDENCE_FLOOR,
    ClassifierResult,
    ConfidenceCheckResult,
    GateDecision,
    GateResult,
    InferenceResult,
    LookAlikeWarning,
)
from edible.data.schemas import Edibility, LookAlikePair, Severity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate(decision: GateDecision, plant_score: float) -> dict:
    return {
        "decision": decision,
        "plant_score": plant_score,
        "reason": "test",
        "gate_type": "imagenet_category",
    }


def _prediction(species_id: str, confidence: float, edibility: str = "edible_raw") -> dict:
    return {"species_id": species_id, "confidence": confidence, "edibility": edibility}


def _accepted_result(**kwargs) -> dict:
    base = {
        "accepted": True,
        "species_id": "rubus_trivialis",
        "species_common": "Dewberry",
        "edibility": "edible_raw",
        "confidence": 0.88,
    }
    return {**base, **kwargs}


def _rejected_result(**kwargs) -> dict:
    base = {
        "accepted": False,
        "rejection_reason": "not_a_plant",
        "rejection_message": "This does not appear to be a plant.",
    }
    return {**base, **kwargs}


# ---------------------------------------------------------------------------
# Layer 1 gate — fail-safe invariant
# ---------------------------------------------------------------------------

class TestGateResult:
    def test_high_plant_score_passes(self):
        g = GateResult.model_validate(_gate(GateDecision.PASS, 0.92))
        assert g.decision == GateDecision.PASS

    def test_zero_plant_score_rejects(self):
        g = GateResult.model_validate(_gate(GateDecision.REJECT, 0.0))
        assert g.decision == GateDecision.REJECT

    def test_exactly_half_plant_score_may_reject(self):
        g = GateResult.model_validate(_gate(GateDecision.REJECT, 0.5))
        assert g.decision == GateDecision.REJECT

    def test_barely_below_half_with_pass_is_rejected_by_validator(self):
        """Fail-safe: plant_score < 0.5 with PASS decision must raise."""
        with pytest.raises(ValidationError, match="fail-safe"):
            GateResult.model_validate(_gate(GateDecision.PASS, 0.49))

    def test_zero_score_with_pass_decision_raises(self):
        with pytest.raises(ValidationError, match="fail-safe"):
            GateResult.model_validate(_gate(GateDecision.PASS, 0.0))

    def test_plant_score_above_range_raises(self):
        with pytest.raises(ValidationError):
            GateResult.model_validate(_gate(GateDecision.PASS, 1.01))

    def test_plant_score_below_range_raises(self):
        with pytest.raises(ValidationError):
            GateResult.model_validate(_gate(GateDecision.PASS, -0.01))

    def test_plant_score_exactly_one_accepted(self):
        g = GateResult.model_validate(_gate(GateDecision.PASS, 1.0))
        assert g.plant_score == 1.0

    def test_plant_score_exactly_zero_with_reject_accepted(self):
        g = GateResult.model_validate(_gate(GateDecision.REJECT, 0.0))
        assert g.decision == GateDecision.REJECT

    def test_high_score_can_still_reject(self):
        """The gate is allowed to be conservative — high score + REJECT is valid."""
        g = GateResult.model_validate(_gate(GateDecision.REJECT, 0.85))
        assert g.decision == GateDecision.REJECT

    def test_both_gate_types_accepted(self):
        for gt in ("imagenet_category", "clip"):
            g = GateResult.model_validate({**_gate(GateDecision.PASS, 0.9), "gate_type": gt})
            assert g.gate_type == gt

    def test_missing_required_fields_raise(self):
        for field in ("decision", "plant_score", "reason", "gate_type"):
            d = {k: v for k, v in _gate(GateDecision.PASS, 0.9).items() if k != field}
            with pytest.raises(ValidationError):
                GateResult.model_validate(d)


# ---------------------------------------------------------------------------
# Classifier result
# ---------------------------------------------------------------------------

class TestClassifierResult:
    def _two_class_result(self, a_conf: float, b_conf: float) -> dict:
        return {
            "predictions": [
                _prediction("species_a", a_conf),
                _prediction("species_b", b_conf),
            ],
            "top_prediction": _prediction("species_a", a_conf),
        }

    def test_valid_result_accepted(self):
        r = ClassifierResult.model_validate(self._two_class_result(0.8, 0.2))
        assert r.top_prediction.species_id == "species_a"

    def test_confidences_sum_to_one(self):
        r = ClassifierResult.model_validate(self._two_class_result(0.7, 0.3))
        assert abs(sum(p.confidence for p in r.predictions) - 1.0) < 0.01

    def test_confidences_not_summing_to_one_raises(self):
        with pytest.raises(ValidationError, match="sum"):
            ClassifierResult.model_validate(self._two_class_result(0.8, 0.8))

    def test_top_prediction_not_in_predictions_raises(self):
        with pytest.raises(ValidationError, match="not in predictions"):
            ClassifierResult.model_validate({
                "predictions": [_prediction("species_a", 1.0)],
                "top_prediction": _prediction("species_z", 1.0),
            })

    def test_empty_predictions_raises(self):
        with pytest.raises(ValidationError):
            ClassifierResult.model_validate({
                "predictions": [],
                "top_prediction": _prediction("species_a", 1.0),
            })

    def test_toxic_top_prediction_accepted(self):
        r = ClassifierResult.model_validate({
            "predictions": [
                _prediction("phytolacca_americana", 0.9, "toxic"),
                _prediction("sambucus_canadensis", 0.1, "edible_cooked"),
            ],
            "top_prediction": _prediction("phytolacca_americana", 0.9, "toxic"),
        })
        assert r.top_prediction.edibility == Edibility.TOXIC


# ---------------------------------------------------------------------------
# Layer 2 — confidence floor
# ---------------------------------------------------------------------------

class TestConfidenceCheckResult:
    def test_above_floor_passes(self):
        r = ConfidenceCheckResult.model_validate({
            "passes": True, "confidence": 0.80, "reason": "above threshold"
        })
        assert r.passes

    def test_exactly_at_floor_passes(self):
        r = ConfidenceCheckResult.model_validate({
            "passes": True, "confidence": CONFIDENCE_FLOOR, "reason": "at threshold"
        })
        assert r.passes

    def test_below_floor_must_not_pass(self):
        """Fail-safe: passes=True with confidence < threshold must raise."""
        with pytest.raises(ValidationError, match="fail-safe"):
            ConfidenceCheckResult.model_validate({
                "passes": True, "confidence": 0.74, "reason": "wrong"
            })

    def test_zero_confidence_fails(self):
        r = ConfidenceCheckResult.model_validate({
            "passes": False, "confidence": 0.0, "reason": "no signal"
        })
        assert not r.passes

    def test_just_below_floor_fails(self):
        r = ConfidenceCheckResult.model_validate({
            "passes": False, "confidence": 0.749, "reason": "below floor"
        })
        assert not r.passes

    def test_custom_threshold_respected(self):
        """If a non-default threshold is used, validator must respect it."""
        with pytest.raises(ValidationError, match="fail-safe"):
            ConfidenceCheckResult.model_validate({
                "passes": True, "confidence": 0.85, "threshold_used": 0.90, "reason": ""
            })

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceCheckResult.model_validate({
                "passes": False, "confidence": 1.5, "reason": ""
            })


# ---------------------------------------------------------------------------
# InferenceResult — safety invariants
# ---------------------------------------------------------------------------

class TestInferenceResultAccepted:
    def test_valid_accepted_result(self):
        r = InferenceResult.model_validate(_accepted_result())
        assert r.accepted
        assert r.edibility == Edibility.EDIBLE_RAW

    def test_accepted_toxic_result_valid(self):
        r = InferenceResult.model_validate(_accepted_result(edibility="toxic"))
        assert r.edibility == Edibility.TOXIC

    def test_accepted_missing_species_id_raises(self):
        with pytest.raises(ValidationError, match="missing fields"):
            InferenceResult.model_validate(_accepted_result(species_id=None))

    def test_accepted_missing_confidence_raises(self):
        with pytest.raises(ValidationError, match="missing fields"):
            InferenceResult.model_validate(_accepted_result(confidence=None))

    def test_accepted_missing_edibility_raises(self):
        with pytest.raises(ValidationError, match="missing fields"):
            InferenceResult.model_validate(_accepted_result(edibility=None))

    def test_accepted_with_rejection_reason_raises(self):
        with pytest.raises(ValidationError, match="rejection_reason"):
            InferenceResult.model_validate(_accepted_result(rejection_reason="not_a_plant"))

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            InferenceResult.model_validate(_accepted_result(confidence=1.5))


class TestInferenceResultRejected:
    def test_valid_rejected_result(self):
        r = InferenceResult.model_validate(_rejected_result())
        assert not r.accepted

    def test_rejected_must_not_have_edibility(self):
        """SAFETY: a rejected result must never carry an edibility verdict."""
        with pytest.raises(ValidationError, match="SAFETY VIOLATION"):
            InferenceResult.model_validate(_rejected_result(edibility="edible_raw"))

    def test_rejected_toxic_edibility_also_blocked(self):
        """Even setting edibility=toxic on a rejected result is disallowed."""
        with pytest.raises(ValidationError, match="SAFETY VIOLATION"):
            InferenceResult.model_validate(_rejected_result(edibility="toxic"))

    def test_rejected_must_not_have_confidence(self):
        with pytest.raises(ValidationError, match="confidence"):
            InferenceResult.model_validate(_rejected_result(confidence=0.5))

    def test_rejected_requires_rejection_message(self):
        with pytest.raises(ValidationError, match="rejection_message"):
            InferenceResult.model_validate(_rejected_result(rejection_message=""))

    def test_rejected_requires_rejection_message_not_whitespace(self):
        with pytest.raises(ValidationError, match="rejection_message"):
            InferenceResult.model_validate(_rejected_result(rejection_message="   "))

    def test_rejected_requires_rejection_reason(self):
        with pytest.raises(ValidationError, match="rejection_reason"):
            r = _rejected_result()
            del r["rejection_reason"]
            InferenceResult.model_validate(r)

    def test_all_rejection_reasons_accepted(self):
        for reason in ("not_a_plant", "low_confidence", "image_invalid"):
            r = InferenceResult.model_validate(_rejected_result(rejection_reason=reason))
            assert r.rejection_reason.value == reason

    def test_disclaimer_always_present(self):
        r = InferenceResult.model_validate(_rejected_result())
        assert r.disclaimer.strip()


class TestDoNotEatBanner:
    """The 'do not eat' banner must appear in all dangerous situations."""

    def test_rejected_result_requires_banner(self):
        r = InferenceResult.model_validate(_rejected_result())
        assert r.requires_do_not_eat_banner

    def test_toxic_accepted_requires_banner(self):
        r = InferenceResult.model_validate(_accepted_result(edibility="toxic", confidence=0.95))
        assert r.requires_do_not_eat_banner

    def test_low_confidence_accepted_requires_banner(self):
        r = InferenceResult.model_validate(_accepted_result(confidence=0.70))
        assert r.requires_do_not_eat_banner

    def test_at_confidence_floor_does_not_require_banner(self):
        r = InferenceResult.model_validate(_accepted_result(confidence=CONFIDENCE_FLOOR))
        assert not r.requires_do_not_eat_banner

    def test_high_confidence_edible_raw_no_banner(self):
        r = InferenceResult.model_validate(_accepted_result(
            edibility="edible_raw", confidence=0.92
        ))
        assert not r.requires_do_not_eat_banner

    def test_edible_cooked_high_confidence_no_banner(self):
        r = InferenceResult.model_validate(_accepted_result(
            edibility="edible_cooked", confidence=0.88
        ))
        assert not r.requires_do_not_eat_banner

    def test_uncertain_edibility_does_not_trigger_banner_by_edibility_alone(self):
        # uncertain edibility above floor — banner is controlled by confidence, not edibility
        r = InferenceResult.model_validate(_accepted_result(
            edibility="uncertain", confidence=0.80
        ))
        assert not r.requires_do_not_eat_banner


class TestHighSeverityWarnings:
    def _warning(self, severity: str) -> dict:
        return {
            "pair_id": "test_pair",
            "lookalike_common": "Pokeweed",
            "lookalike_species_id": "phytolacca_americana",
            "severity": severity,
            "warning_message": "Danger!",
            "distinguishing_features": ["Feature A"],
        }

    def test_has_high_severity_warnings_true(self):
        r = InferenceResult.model_validate(_accepted_result(
            lookalike_warnings=[self._warning("high")]
        ))
        assert r.has_high_severity_warnings

    def test_has_high_severity_warnings_false_for_medium_only(self):
        r = InferenceResult.model_validate(_accepted_result(
            lookalike_warnings=[self._warning("medium")]
        ))
        assert not r.has_high_severity_warnings

    def test_no_warnings_is_false(self):
        r = InferenceResult.model_validate(_accepted_result())
        assert not r.has_high_severity_warnings


# ---------------------------------------------------------------------------
# LookAlikeWarning.from_pair
# ---------------------------------------------------------------------------

class TestLookAlikeWarning:
    def _pair(self) -> LookAlikePair:
        return LookAlikePair.model_validate({
            "id": "elderberry_pokeweed_fruiting",
            "edible_species_id": "sambucus_canadensis",
            "lookalike_species_id": "phytolacca_americana",
            "edible_common": "American Elderberry",
            "lookalike_common": "Pokeweed",
            "confusion_stage": "fruiting",
            "severity": "high",
            "warning_message": "Pokeweed is toxic.",
            "distinguishing_features": ["Stem color", "Leaf type", "Cluster shape"],
            "source": "Test source",
        })

    def test_from_pair_creates_warning(self):
        w = LookAlikeWarning.from_pair(self._pair())
        assert w.pair_id == "elderberry_pokeweed_fruiting"
        assert w.severity == Severity.HIGH

    def test_from_pair_copies_warning_message(self):
        w = LookAlikeWarning.from_pair(self._pair())
        assert "toxic" in w.warning_message.lower()

    def test_from_pair_copies_distinguishing_features(self):
        w = LookAlikeWarning.from_pair(self._pair())
        assert len(w.distinguishing_features) == 3

    def test_from_pair_defaults_poison_control(self):
        w = LookAlikeWarning.from_pair(self._pair())
        assert w.poison_control == "1-800-222-1222"
