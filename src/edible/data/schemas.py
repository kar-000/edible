"""
Pydantic schemas for species and look-alike reference data.

These schemas validate the static reference files (species.json, lookalikes.json)
and are also used as the canonical data types throughout the pipeline.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Edibility(str, Enum):
    EDIBLE_RAW = "edible_raw"
    EDIBLE_COOKED = "edible_cooked"
    TOXIC = "toxic"
    UNCERTAIN = "uncertain"


class ConfusionStage(str, Enum):
    FRUITING = "fruiting"
    FLOWERING_FOLIAGE = "flowering_foliage"
    SEEDLING = "seedling"
    FOLIAGE = "foliage"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LabelConfidence(str, Enum):
    RESEARCH_GRADE = "research_grade"
    EXPERT_REVIEWED = "expert_reviewed"
    UNVERIFIED = "unverified"


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class Species(BaseModel):
    id: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    common_name: str
    scientific_name: str
    family: str
    edibility: Edibility
    is_target_class: bool
    inat_taxon_id: Optional[int] = None
    inat_taxon_id_verified: bool = False
    priority_reason: str
    distinguishing_features: list[str] = Field(default_factory=list)
    habitat: str = ""
    season_fruiting: str = ""
    texas_counties_confirmed: bool = False
    notes: str = ""

    @field_validator("distinguishing_features")
    @classmethod
    def features_not_empty_strings(cls, v: list[str]) -> list[str]:
        for item in v:
            if not item.strip():
                raise ValueError("distinguishing_features must not contain blank strings")
        return v

    @field_validator("scientific_name")
    @classmethod
    def scientific_name_two_words(cls, v: str) -> str:
        parts = v.strip().split()
        if len(parts) < 2:
            raise ValueError(
                f"scientific_name '{v}' must include at least genus and species epithet"
            )
        return v


class SpeciesDatabase(BaseModel):
    version: str
    scope: str
    notes: str = ""
    species: list[Species]

    @model_validator(mode="after")
    def ids_are_unique(self) -> SpeciesDatabase:
        ids = [s.id for s in self.species]
        if len(ids) != len(set(ids)):
            dupes = [i for i in ids if ids.count(i) > 1]
            raise ValueError(f"Duplicate species IDs found: {set(dupes)}")
        return self

    def get_by_id(self, species_id: str) -> Optional[Species]:
        return next((s for s in self.species if s.id == species_id), None)

    def get_toxic(self) -> list[Species]:
        return [s for s in self.species if s.edibility == Edibility.TOXIC]

    def get_edible(self) -> list[Species]:
        return [
            s for s in self.species
            if s.edibility in (Edibility.EDIBLE_RAW, Edibility.EDIBLE_COOKED)
        ]

    def get_target_classes(self) -> list[Species]:
        return [s for s in self.species if s.is_target_class]


# ---------------------------------------------------------------------------
# Look-alike pairs
# ---------------------------------------------------------------------------

class LookAlikePair(BaseModel):
    id: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    edible_species_id: str
    lookalike_species_id: str
    edible_common: str
    lookalike_common: str
    confusion_stage: ConfusionStage
    severity: Severity
    warning_message: str
    distinguishing_features: list[str] = Field(..., min_length=1)
    source: str
    poison_control: str = "1-800-222-1222"
    lookalike_note: str = ""

    @field_validator("warning_message")
    @classmethod
    def warning_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("warning_message must not be empty")
        return v

    @field_validator("distinguishing_features")
    @classmethod
    def features_not_empty_strings(cls, v: list[str]) -> list[str]:
        for item in v:
            if not item.strip():
                raise ValueError("distinguishing_features must not contain blank strings")
        return v


class LookAlikeDatabase(BaseModel):
    version: str
    scope: str
    notes: str = ""
    pairs: list[LookAlikePair]

    @model_validator(mode="after")
    def ids_are_unique(self) -> LookAlikeDatabase:
        ids = [p.id for p in self.pairs]
        if len(ids) != len(set(ids)):
            dupes = [i for i in ids if ids.count(i) > 1]
            raise ValueError(f"Duplicate lookalike pair IDs found: {set(dupes)}")
        return self

    def get_high_severity(self) -> list[LookAlikePair]:
        return [p for p in self.pairs if p.severity == Severity.HIGH]

    def get_pairs_for_species(self, species_id: str) -> list[LookAlikePair]:
        """Return all pairs where species_id is either the edible or the look-alike."""
        return [
            p for p in self.pairs
            if p.edible_species_id == species_id or p.lookalike_species_id == species_id
        ]

    def has_high_severity_pair(self, species_id: str) -> bool:
        return any(
            p.severity == Severity.HIGH for p in self.get_pairs_for_species(species_id)
        )


# ---------------------------------------------------------------------------
# Image metadata sidecar
# ---------------------------------------------------------------------------

class ImageMetadata(BaseModel):
    species_scientific: str
    species_common: str
    edibility: Edibility
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    lng: Optional[float] = Field(None, ge=-180.0, le=180.0)
    county: Optional[str] = None
    state: str = "TX"
    state_code: str = "TX"
    usda_confirmed_range: Optional[bool] = None
    confidence_in_label: LabelConfidence = LabelConfidence.UNVERIFIED
    date_observed: Optional[str] = None
    source: str
    image_quality_flag: Optional[str] = None
    inat_observation_id: Optional[int] = None

    @field_validator("lat")
    @classmethod
    def lat_required_with_lng(cls, v: Optional[float]) -> Optional[float]:
        return v

    @model_validator(mode="after")
    def lat_lng_both_or_neither(self) -> ImageMetadata:
        if (self.lat is None) != (self.lng is None):
            raise ValueError("lat and lng must both be provided or both be None")
        return self


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_species_db(path: Path) -> SpeciesDatabase:
    """Load and validate species.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return SpeciesDatabase.model_validate(data)


def load_lookalike_db(path: Path) -> LookAlikeDatabase:
    """Load and validate lookalikes.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return LookAlikeDatabase.model_validate(data)
