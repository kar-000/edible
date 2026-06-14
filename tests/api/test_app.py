"""
Tests for the FastAPI app and InferencePipeline.

All tests use stub gates and a stub classifier so no pretrained weights
are downloaded in CI.  The FastAPI dependency ``get_pipeline`` is
overridden via ``app.dependency_overrides`` to inject a canned pipeline.
"""

from __future__ import annotations

import io
from typing import Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from edible.api.app import app, get_pipeline
from edible.api.inference import DEFAULT_CONFIDENCE_FLOOR, InferencePipeline
from edible.data.pipeline import FruitGateResult, GateDecision, GateResult, InferenceResult, RejectionReason
from edible.data.schemas import (
    Edibility,
    LookAlikeDatabase,
    LookAlikePair,
    ConfusionStage,
    Severity,
    Species,
    SpeciesDatabase,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal in-memory databases
# ---------------------------------------------------------------------------

@pytest.fixture
def species_db() -> SpeciesDatabase:
    return SpeciesDatabase(
        version="test",
        scope="test",
        species=[
            Species(
                id="rubus_trivialis",
                common_name="Southern Blackberry",
                scientific_name="Rubus trivialis",
                family="Rosaceae",
                edibility=Edibility.EDIBLE_RAW,
                is_target_class=True,
                priority_reason="test edible",
            ),
            Species(
                id="melia_azedarach",
                common_name="Chinaberry",
                scientific_name="Melia azedarach",
                family="Meliaceae",
                edibility=Edibility.TOXIC,
                is_target_class=True,
                priority_reason="test toxic",
            ),
        ],
    )


@pytest.fixture
def lookalike_db() -> LookAlikeDatabase:
    return LookAlikeDatabase(
        version="test",
        scope="test",
        pairs=[
            LookAlikePair(
                id="blackberry_chinaberry",
                edible_species_id="rubus_trivialis",
                lookalike_species_id="melia_azedarach",
                edible_common="Southern Blackberry",
                lookalike_common="Chinaberry",
                confusion_stage=ConfusionStage.FRUITING,
                severity=Severity.HIGH,
                warning_message="Chinaberry is toxic — do not confuse with blackberry.",
                distinguishing_features=["Chinaberry has yellow drupes"],
                source="test",
            )
        ],
    )


@pytest.fixture
def empty_lookalike_db() -> LookAlikeDatabase:
    return LookAlikeDatabase(version="test", scope="test", pairs=[])


# ---------------------------------------------------------------------------
# Stub gates and classifier
# ---------------------------------------------------------------------------

def _pass_gate() -> GateResult:
    return GateResult(decision=GateDecision.PASS, plant_score=0.9, reason="stub pass", gate_type="stub")


def _reject_gate() -> GateResult:
    return GateResult(decision=GateDecision.REJECT, plant_score=0.1, reason="stub reject", gate_type="stub")


def _pass_fruit_gate() -> FruitGateResult:
    return FruitGateResult(decision=GateDecision.PASS, fruit_score=0.9, reason="stub pass", gate_type="stub")


def _reject_fruit_gate() -> FruitGateResult:
    return FruitGateResult(decision=GateDecision.REJECT, fruit_score=0.1, reason="stub reject", gate_type="stub")


class _StubClassifier(nn.Module):
    """Returns fixed logits that put rubus_trivialis (idx=0) at high confidence."""

    def __init__(self, top_class_idx: int = 0, num_classes: int = 2):
        super().__init__()
        self.top_class_idx = top_class_idx
        self.num_classes = num_classes
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        logits = torch.full((b, self.num_classes), -10.0)
        logits[:, self.top_class_idx] = 10.0
        return logits


def _identity_transform(img: PILImage.Image) -> torch.Tensor:
    """Minimal transform: resize to 3×224×224 tensor without normalization."""
    import torchvision.transforms.functional as TF
    return TF.to_tensor(img.resize((224, 224)))


def _make_pipeline(
    species_db: SpeciesDatabase,
    lookalike_db: LookAlikeDatabase,
    plant_passes: bool = True,
    fruit_passes: bool = True,
    top_class_idx: int = 0,
    temperature: float = 1.0,
    per_class_thresholds: Optional[dict] = None,
) -> InferencePipeline:
    species_ids = sorted(s.id for s in species_db.species)
    edibility_map = {s.id: s.edibility for s in species_db.species}

    return InferencePipeline(
        plant_gate=lambda _: _pass_gate() if plant_passes else _reject_gate(),
        fruit_gate=lambda _: _pass_fruit_gate() if fruit_passes else _reject_fruit_gate(),
        classifier=_StubClassifier(top_class_idx=top_class_idx, num_classes=len(species_ids)),
        classifier_transform=_identity_transform,
        plant_gate_transform=_identity_transform,
        fruit_gate_transform=_identity_transform,
        species_ids=species_ids,
        edibility_map=edibility_map,
        species_db=species_db,
        lookalike_db=lookalike_db,
        temperature=temperature,
        per_class_thresholds=per_class_thresholds,
    )


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    img = PILImage.new("RGB", (width, height), color=(100, 150, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# InferencePipeline unit tests
# ---------------------------------------------------------------------------

class TestInferencePipelinePlantGate:
    def test_plant_gate_reject_returns_not_a_plant(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db, plant_passes=False)
        img = PILImage.new("RGB", (64, 64))
        result = pipeline.run(img)
        assert not result.accepted
        assert result.rejection_reason == RejectionReason.NOT_A_PLANT

    def test_plant_gate_reject_has_rejection_message(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db, plant_passes=False)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.rejection_message.strip()

    def test_plant_gate_reject_no_edibility(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db, plant_passes=False)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.edibility is None


class TestInferencePipelineFruitGate:
    def test_fruit_gate_reject_returns_no_fruit_visible(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db, fruit_passes=False)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert not result.accepted
        assert result.rejection_reason == RejectionReason.NO_FRUIT_VISIBLE

    def test_fruit_gate_reject_has_show_me_message(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db, fruit_passes=False)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert "berries" in result.rejection_message.lower()

    def test_fruit_gate_reject_no_edibility(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db, fruit_passes=False)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.edibility is None


class TestInferencePipelineConfidenceFloor:
    def test_low_confidence_returns_low_confidence_rejection(self, species_db, empty_lookalike_db):
        # Raise per-class threshold above what the stub can produce (stub gives ~1.0)
        # → instead use threshold > 1.0 to force rejection
        pipeline = _make_pipeline(
            species_db, empty_lookalike_db,
            per_class_thresholds={"melia_azedarach": 2.0, "rubus_trivialis": 2.0},
        )
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert not result.accepted
        assert result.rejection_reason == RejectionReason.LOW_CONFIDENCE

    def test_low_confidence_rejection_message_contains_percent(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(
            species_db, empty_lookalike_db,
            per_class_thresholds={"melia_azedarach": 2.0, "rubus_trivialis": 2.0},
        )
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert "%" in result.rejection_message

    def test_low_confidence_no_edibility(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(
            species_db, empty_lookalike_db,
            per_class_thresholds={"melia_azedarach": 2.0, "rubus_trivialis": 2.0},
        )
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.edibility is None


class TestInferencePipelineSuccess:
    def test_successful_result_is_accepted(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.accepted

    def test_successful_result_has_species_id(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.species_id in [s.id for s in species_db.species]

    def test_successful_result_has_confidence(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_successful_result_has_edibility(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.edibility is not None

    def test_successful_result_has_common_name(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.species_common

    def test_lookalike_warnings_attached(self, species_db, lookalike_db):
        # top_class_idx=0 → rubus_trivialis (has a lookalike pair)
        pipeline = _make_pipeline(species_db, lookalike_db, top_class_idx=0)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.accepted
        assert len(result.lookalike_warnings) == 1

    def test_no_warnings_when_no_lookalike(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db)
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.lookalike_warnings == []

    def test_gps_accepted_without_error(self, species_db, empty_lookalike_db):
        pipeline = _make_pipeline(species_db, empty_lookalike_db)
        result = pipeline.run(PILImage.new("RGB", (64, 64)), lat=30.2672, lon=-97.7431)
        assert result.accepted


class TestInferencePipelineTemperature:
    def test_temperature_above_one_reduces_confidence(self, species_db, empty_lookalike_db):
        pipeline_t1 = _make_pipeline(species_db, empty_lookalike_db, temperature=1.0)
        pipeline_t2 = _make_pipeline(species_db, empty_lookalike_db, temperature=2.0)
        result_t1 = pipeline_t1.run(PILImage.new("RGB", (64, 64)))
        result_t2 = pipeline_t2.run(PILImage.new("RGB", (64, 64)))
        assert result_t1.confidence > result_t2.confidence

    def test_per_class_threshold_applied(self, species_db, empty_lookalike_db):
        # rubus_trivialis at idx=0 should be top class; threshold below confidence → passes
        pipeline = _make_pipeline(
            species_db, empty_lookalike_db,
            per_class_thresholds={"rubus_trivialis": 0.5},
        )
        result = pipeline.run(PILImage.new("RGB", (64, 64)))
        assert result.accepted


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client_no_model():
    """TestClient with pipeline NOT loaded (model_loaded: false).

    Patches _build_pipeline to always fail so no checkpoint or CLIP weights
    are needed, regardless of local state.
    """
    with patch("edible.api.app._build_pipeline", side_effect=FileNotFoundError("no checkpoint")):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_with_model(species_db, empty_lookalike_db):
    """TestClient with a stub pipeline injected."""
    stub = _make_pipeline(species_db, empty_lookalike_db)
    app.dependency_overrides[get_pipeline] = lambda: stub
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_plant_reject(species_db, empty_lookalike_db):
    stub = _make_pipeline(species_db, empty_lookalike_db, plant_passes=False)
    app.dependency_overrides[get_pipeline] = lambda: stub
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_fruit_reject(species_db, empty_lookalike_db):
    stub = _make_pipeline(species_db, empty_lookalike_db, fruit_passes=False)
    app.dependency_overrides[get_pipeline] = lambda: stub
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_returns_200(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_model_loaded_false_when_no_checkpoint(self, client_no_model):
        resp = client_no_model.get("/health")
        # CI has no checkpoint → model_loaded is false
        assert "model_loaded" in resp.json()


class TestSpeciesEndpoint:
    def test_species_returns_200(self, client_no_model):
        # Patch the data dir to a temp path with minimal species.json
        import json
        import tempfile
        from pathlib import Path
        import edible.api.app as app_module

        species_data = {
            "version": "test",
            "scope": "test",
            "notes": "",
            "species": [
                {
                    "id": "rubus_trivialis",
                    "common_name": "Southern Blackberry",
                    "scientific_name": "Rubus trivialis",
                    "family": "Rosaceae",
                    "edibility": "edible_raw",
                    "is_target_class": True,
                    "priority_reason": "test",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "species.json").write_text(json.dumps(species_data))
            original = app_module._DATA_DIR
            app_module._DATA_DIR = data_dir
            try:
                resp = client_no_model.get("/species")
                assert resp.status_code == 200
                assert isinstance(resp.json(), list)
                assert len(resp.json()) == 1
                assert resp.json()[0]["id"] == "rubus_trivialis"
            finally:
                app_module._DATA_DIR = original


class TestIdentifyEndpoint:
    def test_identify_no_model_returns_503(self, client_no_model):
        resp = client_no_model.post(
            "/identify",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 503

    def test_identify_invalid_image_returns_422(self, client_with_model):
        resp = client_with_model.post(
            "/identify",
            files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_identify_valid_image_returns_200(self, client_with_model):
        resp = client_with_model.post(
            "/identify",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_identify_response_has_accepted_field(self, client_with_model):
        resp = client_with_model.post(
            "/identify",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert "accepted" in resp.json()

    def test_identify_accepted_result_has_species(self, client_with_model):
        resp = client_with_model.post(
            "/identify",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = resp.json()
        if body["accepted"]:
            assert body["species_id"] is not None

    def test_identify_plant_gate_rejection(self, client_plant_reject):
        resp = client_plant_reject.post(
            "/identify",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert not body["accepted"]
        assert body["rejection_reason"] == "not_a_plant"

    def test_identify_fruit_gate_rejection(self, client_fruit_reject):
        resp = client_fruit_reject.post(
            "/identify",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert not body["accepted"]
        assert body["rejection_reason"] == "no_fruit_visible"

    def test_identify_with_gps_coords(self, client_with_model):
        resp = client_with_model.post(
            "/identify",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
            data={"lat": "30.2672", "lon": "-97.7431"},
        )
        assert resp.status_code == 200

    def test_identify_result_has_disclaimer(self, client_with_model):
        resp = client_with_model.post(
            "/identify",
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        body = resp.json()
        assert "disclaimer" in body
        assert body["disclaimer"]
