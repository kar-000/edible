"""
Inference pipeline types (Addendum A).

Defines the typed contracts for every stage of the identification pipeline:
  Layer 1  → GateResult      (plant / not-plant gate)
  Classifier → SpeciesResult  (per-class prediction)
  Layer 2  → ConfidenceResult (75% floor check)
  Final    → InferenceResult  (complete response to the caller)

No ML code lives here — these are data contracts only. Actual gate implementations
live in src/edible/model/ and are wired in Phase 3.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from edible.data.schemas import Edibility, LookAlikePair, Severity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_FLOOR = 0.75  # below this → Layer 2 rejection


# ---------------------------------------------------------------------------
# Layer 1 — Coarse plant gate
# ---------------------------------------------------------------------------

class GateDecision(str, Enum):
    PASS = "pass"
    REJECT = "reject"


class GateResult(BaseModel):
    """Output of the Layer 1 coarse plant/not-plant gate."""

    decision: GateDecision
    plant_score: float = Field(..., ge=0.0, le=1.0)
    reason: str

    # Gate implementation that produced this result
    gate_type: str  # e.g. "imagenet_category" | "clip"

    @model_validator(mode="after")
    def reject_wins_on_uncertainty(self) -> GateResult:
        """
        Fail-safe invariant: if the plant_score is below 0.5 the decision
        must be REJECT, never PASS.  The gate is allowed to be more
        conservative (reject at higher thresholds) but it may never pass
        an image it scores below 50% as plant.
        """
        if self.plant_score < 0.5 and self.decision == GateDecision.PASS:
            raise ValueError(
                f"Gate fail-safe violated: plant_score={self.plant_score:.3f} < 0.5 "
                f"but decision is PASS. Gate must REJECT when uncertain."
            )
        return self


# ---------------------------------------------------------------------------
# Classifier output — per-species prediction
# ---------------------------------------------------------------------------

class SpeciesPrediction(BaseModel):
    """Single class output from the species classifier."""

    species_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    edibility: Edibility


class ClassifierResult(BaseModel):
    """Raw output of the species classifier (before Layer 2 floor check)."""

    predictions: list[SpeciesPrediction] = Field(..., min_length=1)
    top_prediction: SpeciesPrediction

    @model_validator(mode="after")
    def top_must_be_in_predictions(self) -> ClassifierResult:
        ids = {p.species_id for p in self.predictions}
        if self.top_prediction.species_id not in ids:
            raise ValueError(
                f"top_prediction '{self.top_prediction.species_id}' is not in predictions list"
            )
        return self

    @model_validator(mode="after")
    def confidences_sum_to_one(self) -> ClassifierResult:
        total = sum(p.confidence for p in self.predictions)
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Classifier confidences must sum to ~1.0, got {total:.4f}"
            )
        return self


# ---------------------------------------------------------------------------
# Layer 2 — Confidence floor
# ---------------------------------------------------------------------------

class ConfidenceCheckResult(BaseModel):
    """Output of the Layer 2 confidence floor check."""

    passes: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    threshold_used: float = Field(default=CONFIDENCE_FLOOR, ge=0.0, le=1.0)
    reason: str = ""

    @model_validator(mode="after")
    def passes_requires_confidence_above_floor(self) -> ConfidenceCheckResult:
        if self.passes and self.confidence < self.threshold_used:
            raise ValueError(
                f"ConfidenceCheckResult.passes=True but confidence={self.confidence:.3f} "
                f"< threshold={self.threshold_used:.3f}. Layer 2 fail-safe violated."
            )
        return self


# ---------------------------------------------------------------------------
# Look-alike warning — surfaced regardless of confidence for high severity
# ---------------------------------------------------------------------------

class LookAlikeWarning(BaseModel):
    """A look-alike warning to surface in the result."""

    pair_id: str
    lookalike_common: str
    lookalike_species_id: str
    severity: Severity
    warning_message: str
    distinguishing_features: list[str]
    poison_control: str = "1-800-222-1222"

    @classmethod
    def from_pair(cls, pair: LookAlikePair) -> LookAlikeWarning:
        return cls(
            pair_id=pair.id,
            lookalike_common=pair.lookalike_common,
            lookalike_species_id=pair.lookalike_species_id,
            severity=pair.severity,
            warning_message=pair.warning_message,
            distinguishing_features=pair.distinguishing_features,
            poison_control=pair.poison_control,
        )


# ---------------------------------------------------------------------------
# Final inference result
# ---------------------------------------------------------------------------

class RejectionReason(str, Enum):
    NOT_A_PLANT = "not_a_plant"
    LOW_CONFIDENCE = "low_confidence"
    IMAGE_INVALID = "image_invalid"


class InferenceResult(BaseModel):
    """
    Complete response from the identification pipeline.

    If accepted=False the caller must display the rejection message and
    must NOT display any edibility verdict.  This invariant is enforced
    here at construction time.
    """

    accepted: bool

    # Present only when accepted=True
    species_id: Optional[str] = None
    species_common: Optional[str] = None
    edibility: Optional[Edibility] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_out_of_range: bool = False
    lookalike_warnings: list[LookAlikeWarning] = Field(default_factory=list)

    # Present only when accepted=False
    rejection_reason: Optional[RejectionReason] = None
    rejection_message: str = ""

    # Always present
    disclaimer: str = (
        "Educational tool only. Always verify with an expert before consuming wild plants. "
        "When in doubt, do not eat it."
    )

    @model_validator(mode="after")
    def accepted_fields_consistent(self) -> InferenceResult:
        if self.accepted:
            missing = [
                f for f in ("species_id", "species_common", "edibility", "confidence")
                if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(
                    f"InferenceResult accepted=True but missing fields: {missing}"
                )
            if self.rejection_reason is not None:
                raise ValueError(
                    "InferenceResult accepted=True must not have a rejection_reason"
                )
        else:
            # Rejected result must NOT expose any edibility verdict
            if self.edibility is not None:
                raise ValueError(
                    "SAFETY VIOLATION: InferenceResult accepted=False must not set edibility. "
                    "A rejected image must never receive an edibility verdict."
                )
            if self.confidence is not None:
                raise ValueError(
                    "InferenceResult accepted=False must not set confidence"
                )
            if not self.rejection_message.strip():
                raise ValueError(
                    "InferenceResult accepted=False must include a rejection_message"
                )
            if self.rejection_reason is None:
                raise ValueError(
                    "InferenceResult accepted=False must include a rejection_reason"
                )
        return self

    @property
    def requires_do_not_eat_banner(self) -> bool:
        """True if the UI must display the 'do not eat' safety banner."""
        if not self.accepted:
            return True
        if self.edibility == Edibility.TOXIC:
            return True
        if self.confidence is not None and self.confidence < CONFIDENCE_FLOOR:
            return True
        return False

    @property
    def has_high_severity_warnings(self) -> bool:
        return any(w.severity == Severity.HIGH for w in self.lookalike_warnings)
