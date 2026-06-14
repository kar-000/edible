"""
FastAPI application — Edible foraging identifier.

Endpoints
---------
GET  /health      Liveness + model-loaded status.
GET  /species     List all 12 v1 species with edibility.
POST /identify    Upload an image (+ optional GPS) → InferenceResult JSON.

Model loading
-------------
Models are loaded once at startup via the FastAPI ``lifespan`` context.
If the checkpoint is missing the app still starts; ``/health`` reports
``model_loaded: false`` and ``/identify`` returns 503.

Set ``EDIBLE_CHECKPOINT`` env var to override the checkpoint path.
The API also looks for ``<checkpoint_dir>/calibration.json`` (written by
``scripts/calibrate.py --save``) to apply temperature scaling and
per-class confidence thresholds from the latest calibration run.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image as PILImage

from edible.api.inference import InferencePipeline
from edible.data.pipeline import InferenceResult
from edible.data.schemas import load_lookalike_db, load_species_db

logger = logging.getLogger(__name__)

# Repository root relative to this file:
# src/edible/api/app.py → ../../../../
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_CHECKPOINT = _REPO_ROOT / "checkpoints" / "run_f" / "best_accuracy.pt"

_pipeline: Optional[InferencePipeline] = None


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_pipeline() -> InferencePipeline:
    """FastAPI dependency — raises 503 if models aren't loaded yet."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _pipeline


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def _build_pipeline() -> InferencePipeline:
    """Load all models and construct the inference pipeline."""
    from edible.model.classifier import ClassifierConfig, build_classifier
    from edible.model.dataset import DEFAULT_EVAL_TRANSFORM
    from edible.model.gate import FruitPresenceGate, PlantGate

    ckpt_path = Path(os.environ.get("EDIBLE_CHECKPOINT", str(_DEFAULT_CHECKPOINT)))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    species_db = load_species_db(_DATA_DIR / "species.json")
    lookalike_db = load_lookalike_db(_DATA_DIR / "lookalikes.json")

    sorted_ids = sorted(s.id for s in species_db.species)
    edibility_map = {s.id: s.edibility for s in species_db.species}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Species classifier (EfficientNet-B0)
    cfg = ClassifierConfig(num_classes=len(sorted_ids))
    model = build_classifier(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Gates
    plant_gate = PlantGate(device=device)
    fruit_gate = FruitPresenceGate(device=device)

    # Calibration (written by scripts/calibrate.py --save)
    cal_path = ckpt_path.parent / "calibration.json"
    temperature = 1.0
    per_class_thresholds: dict[str, float] = {}
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        temperature = float(cal.get("temperature", 1.0))
        per_class_thresholds = {k: float(v) for k, v in cal.get("thresholds", {}).items()}
        logger.info(
            "Loaded calibration: T=%.4f, %d thresholds", temperature, len(per_class_thresholds)
        )

    return InferencePipeline(
        plant_gate=plant_gate,
        fruit_gate=fruit_gate,
        classifier=model,
        classifier_transform=DEFAULT_EVAL_TRANSFORM,
        plant_gate_transform=DEFAULT_EVAL_TRANSFORM,
        fruit_gate_transform=fruit_gate.preprocess or DEFAULT_EVAL_TRANSFORM,
        species_ids=sorted_ids,
        edibility_map=edibility_map,
        species_db=species_db,
        lookalike_db=lookalike_db,
        temperature=temperature,
        per_class_thresholds=per_class_thresholds,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    try:
        _pipeline = _build_pipeline()
        logger.info("Edible model loaded successfully")
    except Exception as exc:
        logger.warning("Could not load model at startup: %s", exc)
        _pipeline = None
    yield
    _pipeline = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Edible — Foraging Identifier",
    description="AI-powered wild berry identification. Educational use only.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    """Liveness probe. ``model_loaded`` is false until checkpoint is found at startup."""
    return JSONResponse({"status": "ok", "model_loaded": _pipeline is not None})


@app.get("/species")
def list_species() -> JSONResponse:
    """Return the v1 species list with edibility classification."""
    species_db = load_species_db(_DATA_DIR / "species.json")
    return JSONResponse([
        {
            "id": s.id,
            "common_name": s.common_name,
            "scientific_name": s.scientific_name,
            "edibility": s.edibility.value,
            "family": s.family,
        }
        for s in sorted(species_db.species, key=lambda s: s.common_name)
    ])


@app.post("/identify", response_model=InferenceResult)
async def identify(
    image: UploadFile = File(..., description="JPEG or PNG photo of the plant"),
    lat: Optional[float] = Form(None, description="GPS latitude (optional)"),
    lon: Optional[float] = Form(None, description="GPS longitude (optional)"),
    pipeline: InferencePipeline = Depends(get_pipeline),
) -> InferenceResult:
    """
    Identify a plant from an image and return edibility verdict.

    The response ``accepted`` field is the primary gate: if ``false``,
    the caller must display ``rejection_message`` and suppress any
    edibility verdict. Never eat a plant with ``accepted: false``.
    """
    contents = await image.read()
    try:
        pil_image = PILImage.open(BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image file — could not decode")

    return pipeline.run(pil_image, lat=lat, lon=lon)
