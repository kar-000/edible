"""
Adversarial tests for data schemas and reference file integrity.

Covers:
- Valid reference files load and validate
- Toxic species are never miscategorized
- Look-alike pairs reference valid species IDs
- High-severity pairs have warning messages
- Malformed / edge-case inputs are rejected
- Safety-critical fields are always present for toxic species
- Boundary conditions on lat/lng, IDs, enumerations
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from edible.data.schemas import (
    Edibility,
    ImageMetadata,
    LookAlikePair,
    Species,
    SpeciesDatabase,
    load_lookalike_db,
    load_species_db,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SPECIES_FILE = DATA_DIR / "species.json"
LOOKALIKES_FILE = DATA_DIR / "lookalikes.json"


# ---------------------------------------------------------------------------
# Reference file integration tests
# ---------------------------------------------------------------------------

class TestReferenceFilesLoad:
    def test_species_file_loads(self):
        db = load_species_db(SPECIES_FILE)
        assert db.version == "1.0"
        assert len(db.species) == 12

    def test_lookalikes_file_loads(self):
        db = load_lookalike_db(LOOKALIKES_FILE)
        assert db.version == "1.0"
        assert len(db.pairs) >= 5

    def test_species_count_is_6_edible_6_toxic(self):
        db = load_species_db(SPECIES_FILE)
        assert len(db.get_toxic()) == 6
        assert len(db.get_edible()) == 6

    def test_all_species_are_target_classes(self):
        db = load_species_db(SPECIES_FILE)
        assert all(s.is_target_class for s in db.species)

    def test_all_species_have_scientific_names(self):
        db = load_species_db(SPECIES_FILE)
        for s in db.species:
            parts = s.scientific_name.split()
            assert len(parts) >= 2, f"{s.id} has malformed scientific name"

    def test_all_species_have_distinguishing_features(self):
        db = load_species_db(SPECIES_FILE)
        for s in db.species:
            assert len(s.distinguishing_features) >= 3, (
                f"{s.id} has fewer than 3 distinguishing features"
            )

    def test_all_species_ids_unique(self):
        db = load_species_db(SPECIES_FILE)
        ids = [s.id for s in db.species]
        assert len(ids) == len(set(ids))

    def test_all_lookalike_ids_unique(self):
        db = load_lookalike_db(LOOKALIKES_FILE)
        ids = [p.id for p in db.pairs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Safety-critical: toxic species validation
# ---------------------------------------------------------------------------

class TestToxicSpeciesSafety:
    """Adversarial: ensure toxic species can never be miscategorized."""

    def test_toxic_species_are_not_edible(self):
        db = load_species_db(SPECIES_FILE)
        edible_ids = {s.id for s in db.get_edible()}
        toxic_ids = {s.id for s in db.get_toxic()}
        assert edible_ids.isdisjoint(toxic_ids), "A species is in both edible and toxic sets"

    def test_known_toxic_species_are_toxic(self):
        db = load_species_db(SPECIES_FILE)
        known_toxic = [
            "phytolacca_americana",
            "menispermum_canadense",
            "solanum_nigrum",
            "ilex_decidua",
            "ilex_vomitoria",
            "melia_azedarach",
        ]
        for species_id in known_toxic:
            s = db.get_by_id(species_id)
            assert s is not None, f"Expected toxic species '{species_id}' not found"
            assert s.edibility == Edibility.TOXIC, (
                f"SAFETY FAILURE: {species_id} is listed as {s.edibility}, expected TOXIC"
            )

    def test_known_edible_species_are_edible(self):
        db = load_species_db(SPECIES_FILE)
        known_edible = [
            "sambucus_canadensis",
            "vitis_mustangensis",
            "rubus_trivialis",
            "callicarpa_americana",
            "mahonia_trifoliolata",
            "celtis_laevigata",
        ]
        for species_id in known_edible:
            s = db.get_by_id(species_id)
            assert s is not None, f"Expected edible species '{species_id}' not found"
            assert s.edibility != Edibility.TOXIC, (
                f"SAFETY FAILURE: {species_id} is marked TOXIC — should be edible"
            )

    def test_all_high_severity_pairs_have_warning_message(self):
        db = load_lookalike_db(LOOKALIKES_FILE)
        for pair in db.get_high_severity():
            assert pair.warning_message.strip(), (
                f"High-severity pair '{pair.id}' has empty warning_message"
            )

    def test_all_pairs_have_poison_control(self):
        db = load_lookalike_db(LOOKALIKES_FILE)
        for pair in db.pairs:
            assert pair.poison_control.strip(), f"Pair '{pair.id}' missing poison_control"

    def test_elderberry_has_high_severity_lookalike(self):
        db = load_lookalike_db(LOOKALIKES_FILE)
        assert db.has_high_severity_pair("sambucus_canadensis"), (
            "SAFETY FAILURE: American elderberry must have at least one high-severity pair"
        )

    def test_mustang_grape_has_high_severity_lookalike(self):
        db = load_lookalike_db(LOOKALIKES_FILE)
        assert db.has_high_severity_pair("vitis_mustangensis"), (
            "SAFETY FAILURE: Mustang grape must have at least one high-severity look-alike pair"
        )

    def test_all_pairs_have_distinguishing_features(self):
        db = load_lookalike_db(LOOKALIKES_FILE)
        for pair in db.pairs:
            assert len(pair.distinguishing_features) >= 3, (
                f"Pair '{pair.id}' has fewer than 3 distinguishing features — unsafe"
            )


# ---------------------------------------------------------------------------
# Species schema validation — adversarial inputs
# ---------------------------------------------------------------------------

class TestSpeciesSchemaValidation:
    BASE = {
        "id": "test_species",
        "common_name": "Test Berry",
        "scientific_name": "Testus berryus",
        "family": "Testaceae",
        "edibility": "edible_raw",
        "is_target_class": True,
        "priority_reason": "Testing",
    }

    def test_valid_species_accepted(self):
        s = Species.model_validate(self.BASE)
        assert s.id == "test_species"

    def test_invalid_edibility_rejected(self):
        with pytest.raises(ValidationError):
            Species.model_validate({**self.BASE, "edibility": "delicious"})

    def test_id_with_uppercase_rejected(self):
        with pytest.raises(ValidationError):
            Species.model_validate({**self.BASE, "id": "TestSpecies"})

    def test_id_starting_with_number_rejected(self):
        with pytest.raises(ValidationError):
            Species.model_validate({**self.BASE, "id": "1test"})

    def test_id_with_spaces_rejected(self):
        with pytest.raises(ValidationError):
            Species.model_validate({**self.BASE, "id": "test species"})

    def test_scientific_name_one_word_rejected(self):
        with pytest.raises(ValidationError):
            Species.model_validate({**self.BASE, "scientific_name": "Mononym"})

    def test_blank_distinguishing_feature_rejected(self):
        with pytest.raises(ValidationError):
            Species.model_validate({**self.BASE, "distinguishing_features": ["valid feature", ""]})

    def test_all_edibility_values_accepted(self):
        for val in ("edible_raw", "edible_cooked", "toxic", "uncertain"):
            s = Species.model_validate({**self.BASE, "edibility": val})
            assert s.edibility.value == val

    def test_missing_required_fields_rejected(self):
        required = ["id", "common_name", "scientific_name", "family", "edibility",
                    "is_target_class", "priority_reason"]
        for field in required:
            partial = {k: v for k, v in self.BASE.items() if k != field}
            with pytest.raises(ValidationError):
                Species.model_validate(partial)


# ---------------------------------------------------------------------------
# SpeciesDatabase validation
# ---------------------------------------------------------------------------

class TestSpeciesDatabaseValidation:
    def _make_db(self, species_list: list[dict]) -> dict:
        return {
            "version": "1.0",
            "scope": "test",
            "notes": "",
            "species": species_list,
        }

    BASE_SPECIES = {
        "id": "alpha",
        "common_name": "Alpha",
        "scientific_name": "Alpha berryi",
        "family": "Testaceae",
        "edibility": "edible_raw",
        "is_target_class": True,
        "priority_reason": "test",
    }

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValidationError):
            SpeciesDatabase.model_validate(self._make_db([
                self.BASE_SPECIES,
                self.BASE_SPECIES,
            ]))

    def test_get_by_id_returns_species(self):
        db = SpeciesDatabase.model_validate(self._make_db([self.BASE_SPECIES]))
        assert db.get_by_id("alpha") is not None

    def test_get_by_id_returns_none_for_missing(self):
        db = SpeciesDatabase.model_validate(self._make_db([self.BASE_SPECIES]))
        assert db.get_by_id("nonexistent") is None

    def test_get_toxic_filters_correctly(self):
        db = SpeciesDatabase.model_validate(self._make_db([
            {**self.BASE_SPECIES, "id": "a", "edibility": "toxic"},
            {**self.BASE_SPECIES, "id": "b", "edibility": "edible_raw"},
        ]))
        toxic = db.get_toxic()
        assert len(toxic) == 1
        assert toxic[0].id == "a"

    def test_get_edible_excludes_toxic(self):
        db = SpeciesDatabase.model_validate(self._make_db([
            {**self.BASE_SPECIES, "id": "a", "edibility": "toxic"},
            {**self.BASE_SPECIES, "id": "b", "edibility": "edible_raw"},
            {**self.BASE_SPECIES, "id": "c", "edibility": "edible_cooked"},
        ]))
        edible = db.get_edible()
        assert len(edible) == 2
        assert all(s.edibility != Edibility.TOXIC for s in edible)


# ---------------------------------------------------------------------------
# LookAlikePair schema validation — adversarial inputs
# ---------------------------------------------------------------------------

class TestLookAlikePairValidation:
    BASE = {
        "id": "test_pair",
        "edible_species_id": "rubus_trivialis",
        "lookalike_species_id": "solanum_nigrum",
        "edible_common": "Dewberry",
        "lookalike_common": "Nightshade",
        "confusion_stage": "fruiting",
        "severity": "high",
        "warning_message": "Do not eat.",
        "distinguishing_features": ["Feature A", "Feature B", "Feature C"],
        "source": "Test source",
    }

    def test_valid_pair_accepted(self):
        p = LookAlikePair.model_validate(self.BASE)
        assert p.id == "test_pair"

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            LookAlikePair.model_validate({**self.BASE, "severity": "critical"})

    def test_invalid_confusion_stage_rejected(self):
        with pytest.raises(ValidationError):
            LookAlikePair.model_validate({**self.BASE, "confusion_stage": "eating"})

    def test_empty_warning_message_rejected(self):
        with pytest.raises(ValidationError):
            LookAlikePair.model_validate({**self.BASE, "warning_message": "   "})

    def test_empty_distinguishing_features_list_rejected(self):
        with pytest.raises(ValidationError):
            LookAlikePair.model_validate({**self.BASE, "distinguishing_features": []})

    def test_blank_feature_in_list_rejected(self):
        with pytest.raises(ValidationError):
            LookAlikePair.model_validate({
                **self.BASE,
                "distinguishing_features": ["Valid feature", ""]
            })

    def test_id_with_spaces_rejected(self):
        with pytest.raises(ValidationError):
            LookAlikePair.model_validate({**self.BASE, "id": "test pair"})


# ---------------------------------------------------------------------------
# ImageMetadata schema validation — adversarial inputs
# ---------------------------------------------------------------------------

class TestImageMetadataValidation:
    BASE = {
        "species_scientific": "Sambucus canadensis",
        "species_common": "American Elderberry",
        "edibility": "edible_cooked",
        "source": "iNaturalist",
    }

    def test_valid_metadata_accepted(self):
        m = ImageMetadata.model_validate(self.BASE)
        assert m.edibility == Edibility.EDIBLE_COOKED

    def test_lat_without_lng_rejected(self):
        with pytest.raises(ValidationError):
            ImageMetadata.model_validate({**self.BASE, "lat": 30.267})

    def test_lng_without_lat_rejected(self):
        with pytest.raises(ValidationError):
            ImageMetadata.model_validate({**self.BASE, "lng": -97.743})

    def test_lat_lng_together_accepted(self):
        m = ImageMetadata.model_validate({**self.BASE, "lat": 30.267, "lng": -97.743})
        assert m.lat == 30.267
        assert m.lng == -97.743

    def test_lat_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ImageMetadata.model_validate({**self.BASE, "lat": 91.0, "lng": -97.743})

    def test_lat_negative_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ImageMetadata.model_validate({**self.BASE, "lat": -91.0, "lng": -97.743})

    def test_lng_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ImageMetadata.model_validate({**self.BASE, "lat": 30.267, "lng": 181.0})

    def test_lng_negative_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ImageMetadata.model_validate({**self.BASE, "lat": 30.267, "lng": -181.0})

    def test_lat_lng_boundary_values_accepted(self):
        for lat, lng in [(90.0, 180.0), (-90.0, -180.0), (0.0, 0.0)]:
            m = ImageMetadata.model_validate({**self.BASE, "lat": lat, "lng": lng})
            assert m.lat == lat

    def test_invalid_edibility_rejected(self):
        with pytest.raises(ValidationError):
            ImageMetadata.model_validate({**self.BASE, "edibility": "maybe"})

    def test_invalid_label_confidence_rejected(self):
        with pytest.raises(ValidationError):
            ImageMetadata.model_validate({**self.BASE, "confidence_in_label": "pretty_sure"})

    def test_all_label_confidence_values_accepted(self):
        for val in ("research_grade", "expert_reviewed", "unverified"):
            m = ImageMetadata.model_validate({**self.BASE, "confidence_in_label": val})
            assert m.confidence_in_label.value == val

    def test_toxic_metadata_accepted(self):
        m = ImageMetadata.model_validate({**self.BASE, "edibility": "toxic"})
        assert m.edibility == Edibility.TOXIC

    def test_none_lat_lng_accepted(self):
        m = ImageMetadata.model_validate(self.BASE)
        assert m.lat is None
        assert m.lng is None


# ---------------------------------------------------------------------------
# load_species_db / load_lookalike_db edge cases
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_load_species_db_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_species_db(tmp_path / "nope.json")

    def test_load_species_db_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("this is not json")
        with pytest.raises(Exception):
            load_species_db(bad)

    def test_load_species_db_wrong_schema_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not": "a species db"}))
        with pytest.raises(ValidationError):
            load_species_db(bad)

    def test_load_lookalike_db_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lookalike_db(tmp_path / "nope.json")

    def test_load_lookalike_db_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{bad json}")
        with pytest.raises(Exception):
            load_lookalike_db(bad)
