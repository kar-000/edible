"""
InferencePipeline — orchestrates all identification layers and returns InferenceResult.

Intentionally decoupled from FastAPI so it can be used in scripts,
tests, and future mobile runtimes.

Pipeline order
--------------
1. Layer 1  : PlantGate    — reject if no plant detected
2. Layer 1b : FruitGate   — reject if no fruit/berries visible
3. Classifier              — EfficientNet-B0 species prediction
4. Layer 2  : confidence floor + temperature scaling + per-class thresholds
5. Post     : look-alike warnings
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from edible.data.geocoding import NominatimGeocoder
from edible.data.pipeline import (
    FruitGateResult,
    GateDecision,
    GateResult,
    InferenceResult,
    LookAlikeWarning,
    RejectionReason,
)
from edible.data.range_check import CountyRangeChecker
from edible.data.schemas import Edibility, LookAlikeDatabase, SpeciesDatabase

DEFAULT_CONFIDENCE_FLOOR = 0.75
DEFAULT_LOCATION_PENALTY = 0.5  # multiply confidence by this when species is out of range


class InferencePipeline:
    """
    Parameters
    ----------
    plant_gate:
        Callable ``(tensor) → GateResult`` (Layer 1).
    fruit_gate:
        Callable ``(tensor) → FruitGateResult`` (Layer 1b).
    classifier:
        ``EdibleClassifier`` (or any ``nn.Module``) whose ``forward``
        returns raw logits of shape ``(B, num_classes)``.
    classifier_transform:
        Transform applied to a PIL Image before the species classifier.
    plant_gate_transform:
        Transform applied to a PIL Image before the plant gate.
    fruit_gate_transform:
        Transform applied to a PIL Image before the fruit gate.
    species_ids:
        Ordered list of species IDs matching classifier class indices 0…N.
    edibility_map:
        Maps species_id → ``Edibility``.
    species_db:
        Full ``SpeciesDatabase`` for common-name lookup.
    lookalike_db:
        ``LookAlikeDatabase`` for look-alike warning generation.
    temperature:
        Calibration temperature (Run F: T≈0.97). Applied as
        ``softmax(logits / T)`` before threshold comparison.
    per_class_thresholds:
        Maps species_id → minimum calibrated confidence to accept that
        prediction.  Species not listed fall back to DEFAULT_CONFIDENCE_FLOOR.
    geocoder:
        Optional Nominatim geocoder for lat/lon → county resolution.
        If None, location re-ranking is skipped.
    county_range_checker:
        Optional static county-level presence checker.
        If None, location re-ranking is skipped.
    location_penalty:
        Confidence multiplier applied when the predicted species is not
        recorded in the user's county.  Default 0.5 (halves confidence,
        which typically pushes borderline predictions below threshold).
    """

    def __init__(
        self,
        plant_gate: Callable,
        fruit_gate: Callable,
        classifier: torch.nn.Module,
        classifier_transform: Callable,
        plant_gate_transform: Callable,
        fruit_gate_transform: Callable,
        species_ids: list[str],
        edibility_map: dict[str, Edibility],
        species_db: SpeciesDatabase,
        lookalike_db: LookAlikeDatabase,
        temperature: float = 1.0,
        per_class_thresholds: Optional[dict[str, float]] = None,
        geocoder: Optional[NominatimGeocoder] = None,
        county_range_checker: Optional[CountyRangeChecker] = None,
        location_penalty: float = DEFAULT_LOCATION_PENALTY,
    ) -> None:
        self.plant_gate = plant_gate
        self.fruit_gate = fruit_gate
        self.classifier = classifier
        self.classifier_transform = classifier_transform
        self.plant_gate_transform = plant_gate_transform
        self.fruit_gate_transform = fruit_gate_transform
        self.species_ids = species_ids
        self.edibility_map = edibility_map
        self.species_db = species_db
        self.lookalike_db = lookalike_db
        self.temperature = temperature
        self.per_class_thresholds = per_class_thresholds or {}
        self.geocoder = geocoder
        self.county_range_checker = county_range_checker
        self.location_penalty = location_penalty

    def run(
        self,
        image: PILImage.Image,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> InferenceResult:
        """
        Run the full identification pipeline on a PIL Image.

        Parameters
        ----------
        image:
            Raw RGB image from the user.
        lat, lon:
            Optional GPS coordinates.  When both are provided and a
            ``geocoder`` + ``county_range_checker`` are wired in, the
            predicted species is checked against county-level range data.
            If it is not recorded in that county, confidence is multiplied
            by ``location_penalty`` (default 0.5), which typically pushes
            borderline predictions below the Layer 2 threshold.
        """
        rgb = image.convert("RGB")

        # Layer 1: plant gate
        plant_tensor = self.plant_gate_transform(rgb)
        gate_result: GateResult = self.plant_gate(plant_tensor)
        if gate_result.decision == GateDecision.REJECT:
            return InferenceResult(
                accepted=False,
                rejection_reason=RejectionReason.NOT_A_PLANT,
                rejection_message=(
                    "I don't recognize this as a plant. "
                    "Try a clearer photo focused on the plant."
                ),
            )

        # Layer 1b: fruit-presence gate
        fruit_tensor = self.fruit_gate_transform(rgb)
        fruit_result: FruitGateResult = self.fruit_gate(fruit_tensor)
        if fruit_result.decision == GateDecision.REJECT:
            return InferenceResult(
                accepted=False,
                rejection_reason=RejectionReason.NO_FRUIT_VISIBLE,
                rejection_message=(
                    "Show me the berries! I need a clear photo of the fruit "
                    "to make a safe identification."
                ),
            )

        # Species classifier + temperature scaling
        clf_tensor = self.classifier_transform(rgb)
        top_species_id, scaled_conf, top_edibility = self._classify(clf_tensor)

        # Location re-ranking: soft confidence penalty for out-of-range species.
        # Fail-safe: any geocoding error → skip penalty (never reject valid input).
        is_out_of_range = False
        if (
            lat is not None
            and lon is not None
            and self.geocoder is not None
            and self.county_range_checker is not None
        ):
            try:
                geo = self.geocoder.reverse_geocode(lat, lon)
                if geo.county:
                    in_range = self.county_range_checker.is_in_county(
                        top_species_id, geo.county, geo.state_code
                    )
                    if in_range is False:
                        scaled_conf *= self.location_penalty
                        is_out_of_range = True
            except Exception:
                pass  # geocoding failure → no penalty

        # Layer 2: confidence floor (per-class threshold or default 75%)
        threshold = self.per_class_thresholds.get(top_species_id, DEFAULT_CONFIDENCE_FLOOR)
        if scaled_conf < threshold:
            return InferenceResult(
                accepted=False,
                rejection_reason=RejectionReason.LOW_CONFIDENCE,
                rejection_message=(
                    f"I'm not confident enough to identify this safely "
                    f"({scaled_conf:.0%} confidence). Do not eat it."
                ),
                is_out_of_range=is_out_of_range,
            )

        # Look-alike warnings
        warnings = [
            LookAlikeWarning.from_pair(pair)
            for pair in self.lookalike_db.get_pairs_for_species(top_species_id)
        ]

        species_obj = self.species_db.get_by_id(top_species_id)
        common_name = species_obj.common_name if species_obj else top_species_id

        return InferenceResult(
            accepted=True,
            species_id=top_species_id,
            species_common=common_name,
            edibility=top_edibility,
            confidence=scaled_conf,
            is_out_of_range=is_out_of_range,
            lookalike_warnings=warnings,
        )

    def _classify(
        self, clf_tensor: torch.Tensor
    ) -> tuple[str, float, Edibility]:
        """Run classifier with temperature scaling; return (species_id, confidence, edibility)."""
        if clf_tensor.dim() == 3:
            clf_tensor = clf_tensor.unsqueeze(0)
        device = next(self.classifier.parameters()).device
        clf_tensor = clf_tensor.to(device)
        with torch.no_grad():
            logits = self.classifier(clf_tensor)  # (1, num_classes)
            probs = F.softmax(logits / self.temperature, dim=1)[0]  # (num_classes,)
        top_idx = int(probs.argmax())
        top_species_id = self.species_ids[top_idx]
        top_conf = float(probs[top_idx])
        top_edibility = self.edibility_map[top_species_id]
        return top_species_id, top_conf, top_edibility
