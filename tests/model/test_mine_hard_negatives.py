"""
Tests for scripts/mine_hard_negatives.py.

Uses stub models so no pretrained weights are downloaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import mine_hard_negatives as mhn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(path: Path) -> None:
    Image.new("RGB", (32, 32), color=(100, 150, 80)).save(path)


def _make_dataset(tmp_path: Path, species: list[dict], n_train: int = 16) -> tuple[Path, Path]:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    db_path = tmp_path / "species.json"
    db_path.write_text(json.dumps({
        "version": "1.0", "scope": "test", "notes": "", "species": species,
    }))
    for sp in species:
        sid = sp["id"]
        (images_dir / sid).mkdir()
        for j in range(n_train):
            _make_rgb_image(images_dir / sid / f"img_{j:04d}.jpg")
    return images_dir, db_path


def _make_checkpoint(tmp_path: Path, num_classes: int = 2) -> Path:
    from edible.model.classifier import ClassifierConfig, build_classifier
    cfg = ClassifierConfig(pretrained=False, num_classes=num_classes)
    model = build_classifier(cfg)
    ckpt_path = tmp_path / "best_safety.pt"
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "val_accuracy": 0.8,
        "toxic_fp_rate": 0.05,
    }, ckpt_path)
    return ckpt_path


def _make_calibration(tmp_path: Path, temperature: float = 1.0,
                      thresholds: dict | None = None) -> Path:
    cal = {"checkpoint": "test", "temperature": temperature, "thresholds": thresholds or {}}
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(json.dumps(cal))
    return cal_path


_TWO_SPECIES = [
    {"id": "alpha", "common_name": "Alpha", "scientific_name": "Alpha sp.",
     "family": "T", "edibility": "toxic", "is_target_class": True, "priority_reason": "t"},
    {"id": "beta", "common_name": "Beta", "scientific_name": "Beta sp.",
     "family": "T", "edibility": "edible_raw", "is_target_class": True, "priority_reason": "t"},
]


# ---------------------------------------------------------------------------
# _load_calibration
# ---------------------------------------------------------------------------

class TestLoadCalibration:
    def test_parses_temperature(self, tmp_path):
        cal = _make_calibration(tmp_path, temperature=1.23)
        T, _ = mhn._load_calibration(cal)
        assert T == pytest.approx(1.23)

    def test_parses_thresholds(self, tmp_path):
        cal = _make_calibration(tmp_path, thresholds={"alpha": 0.75})
        _, thresholds = mhn._load_calibration(cal)
        assert thresholds["alpha"] == pytest.approx(0.75)

    def test_missing_thresholds_returns_empty_dict(self, tmp_path):
        # Write minimal calibration without "thresholds" key
        cal_path = tmp_path / "calibration.json"
        cal_path.write_text(json.dumps({"checkpoint": "x", "temperature": 1.0}))
        _, thresholds = mhn._load_calibration(cal_path)
        assert thresholds == {}


# ---------------------------------------------------------------------------
# Main logic via argparse entrypoint
# ---------------------------------------------------------------------------

class TestMineHardNegativesMain:
    def _run(self, tmp_path: Path, fp_boost: float = 5.0,
             near_miss_boost: float = 2.0, thresholds: dict | None = None) -> dict:
        """Run the script and return the parsed output JSON."""
        images_dir, db_path = _make_dataset(tmp_path, _TWO_SPECIES, n_train=20)
        ckpt_path = _make_checkpoint(tmp_path)
        cal_path = _make_calibration(tmp_path, temperature=1.0, thresholds=thresholds or {})
        out_path = tmp_path / "out.json"

        # Patch DATA_DIR to point at our tmp dataset
        orig_data = mhn.DATA_DIR
        orig_repo = mhn.REPO_ROOT
        mhn.DATA_DIR = tmp_path
        mhn.REPO_ROOT = tmp_path
        try:
            sys.argv = [
                "mine_hard_negatives.py",
                str(ckpt_path),
                "--calibration", str(cal_path),
                "--out", str(out_path),
                "--fp-boost", str(fp_boost),
                "--near-miss-boost", str(near_miss_boost),
            ]
            mhn.main()
        finally:
            mhn.DATA_DIR = orig_data
            mhn.REPO_ROOT = orig_repo

        return json.loads(out_path.read_text())

    def test_output_has_required_keys(self, tmp_path):
        result = self._run(tmp_path)
        assert "checkpoint" in result
        assert "fp_boost" in result
        assert "near_miss_boost" in result
        assert "counts" in result
        assert "samples" in result

    def test_counts_key_has_fps_and_near_misses(self, tmp_path):
        result = self._run(tmp_path)
        assert "fps" in result["counts"]
        assert "near_misses" in result["counts"]

    def test_fp_boost_stored_in_output(self, tmp_path):
        result = self._run(tmp_path, fp_boost=7.0)
        assert result["fp_boost"] == pytest.approx(7.0)

    def test_near_miss_boost_stored_in_output(self, tmp_path):
        result = self._run(tmp_path, near_miss_boost=3.0)
        assert result["near_miss_boost"] == pytest.approx(3.0)

    def test_samples_dict_values_are_boosts(self, tmp_path):
        result = self._run(tmp_path, fp_boost=5.0, near_miss_boost=2.0)
        for path, boost in result["samples"].items():
            assert boost == pytest.approx(5.0) or boost == pytest.approx(2.0), (
                f"Unexpected boost {boost} for {path}"
            )

    def test_samples_only_contain_toxic_images(self, tmp_path):
        result = self._run(tmp_path)
        # All mined paths must come from the toxic species directory
        for rel_path in result["samples"]:
            assert "alpha" in rel_path, f"Expected toxic species 'alpha' in path: {rel_path}"

    def test_output_file_written(self, tmp_path):
        images_dir, db_path = _make_dataset(tmp_path, _TWO_SPECIES, n_train=8)
        ckpt_path = _make_checkpoint(tmp_path)
        cal_path = _make_calibration(tmp_path)
        out_path = tmp_path / "subdir" / "hn.json"

        orig_data, orig_repo = mhn.DATA_DIR, mhn.REPO_ROOT
        mhn.DATA_DIR = tmp_path
        mhn.REPO_ROOT = tmp_path
        try:
            sys.argv = [
                "mine_hard_negatives.py", str(ckpt_path),
                "--calibration", str(cal_path),
                "--out", str(out_path),
            ]
            mhn.main()
        finally:
            mhn.DATA_DIR = orig_data
            mhn.REPO_ROOT = orig_repo

        assert out_path.exists()

    def test_no_fps_when_threshold_above_max_confidence(self, tmp_path):
        # Threshold=1.0 means nothing can qualify as FP (confidence always < 1.0)
        result = self._run(tmp_path, thresholds={"beta": 1.0})
        assert result["counts"]["fps"] == 0

    def test_all_edible_predictions_are_fps_when_threshold_zero(self, tmp_path):
        # Threshold=0.0 means every toxic→edible prediction qualifies as FP
        result = self._run(tmp_path, thresholds={"beta": 0.0})
        fps = result["counts"]["fps"]
        near_misses = result["counts"]["near_misses"]
        # With threshold=0.0, all toxic→edible preds are FPs, none are near-misses
        assert near_misses == 0
        assert fps + near_misses == fps

    def test_checkpoint_path_stored_in_output(self, tmp_path):
        result = self._run(tmp_path)
        assert result["checkpoint"]  # non-empty


# ---------------------------------------------------------------------------
# CLI error handling
# ---------------------------------------------------------------------------

class TestMineHardNegativesCLIErrors:
    def test_missing_checkpoint_exits(self, tmp_path):
        sys.argv = [
            "mine_hard_negatives.py",
            str(tmp_path / "nonexistent.pt"),
        ]
        with pytest.raises(SystemExit):
            mhn.main()

    def test_missing_calibration_exits(self, tmp_path):
        ckpt_path = _make_checkpoint(tmp_path)
        sys.argv = [
            "mine_hard_negatives.py", str(ckpt_path),
            "--calibration", str(tmp_path / "no_cal.json"),
        ]
        with pytest.raises(SystemExit):
            mhn.main()
