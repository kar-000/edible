"""
Tests for the training loop.

Uses a tiny synthetic dataset (2 species × 8 images) and a stub backbone
so tests run on CPU in a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from edible.model.classifier import ClassifierConfig
from edible.model.train import (
    TOXIC_FP_ALARM_THRESHOLD,
    EpochResult,
    TrainConfig,
    _get_device,
    _intra_class_cutmix,
    train,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(path: Path) -> None:
    Image.new("RGB", (32, 32), color=(120, 80, 60)).save(path)


def _make_train_fixture(
    tmp_path: Path,
    species: list[dict],
    n_images: int = 8,
) -> TrainConfig:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    db_path = tmp_path / "species.json"
    db_path.write_text(json.dumps({
        "version": "1.0",
        "scope": "test",
        "notes": "",
        "species": species,
    }))

    for sp in species:
        sid = sp["id"]
        species_dir = images_dir / sid
        species_dir.mkdir()
        for j in range(n_images):
            _make_rgb_image(species_dir / f"img_{j:04d}.jpg")

    # Use a minimal config with a tiny non-pretrained model
    tiny_cfg = ClassifierConfig(
        model_name="efficientnet_b0",
        pretrained=False,
        num_classes=len(species),
        dropout=0.0,
    )

    return TrainConfig(
        images_dir=images_dir,
        species_db_path=db_path,
        classifier_config=tiny_cfg,
        epochs=2,
        batch_size=4,
        num_workers=0,
        checkpoint_dir=tmp_path / "checkpoints",
        save_best_accuracy=True,
        save_best_toxic_fp=True,
        toxic_fp_patience=10,  # don't early-stop in tests
    )


_TWO_SPECIES = [
    {
        "id": "alpha",
        "common_name": "Alpha",
        "scientific_name": "Alpha berryi",
        "family": "Testaceae",
        "edibility": "toxic",
        "is_target_class": True,
        "priority_reason": "test",
    },
    {
        "id": "beta",
        "common_name": "Beta",
        "scientific_name": "Beta safus",
        "family": "Testaceae",
        "edibility": "edible_raw",
        "is_target_class": True,
        "priority_reason": "test",
    },
]


# ---------------------------------------------------------------------------
# EpochResult
# ---------------------------------------------------------------------------

class TestEpochResult:
    def _make_metrics(self, correct: int, total: int, toxic_fp: int, toxic_total: int):
        from edible.model.evaluate import compute_metrics
        # Build minimal metrics via compute_metrics
        n = total
        preds = [0] * n
        labels = [0] * (total - toxic_fp) + [0] * toxic_fp
        return compute_metrics(
            preds, labels, 2,
            toxic_indices={0},
            species_ids=["toxic", "edible"],
        )

    def test_val_accuracy_delegates(self, tmp_path):
        from edible.model.evaluate import compute_metrics
        m = compute_metrics([0, 1], [0, 1], 2, toxic_indices={0})
        result = EpochResult(epoch=1, train_loss=0.5, val_metrics=m, elapsed_seconds=1.0)
        assert result.val_accuracy == m.overall_accuracy

    def test_toxic_fp_rate_delegates(self, tmp_path):
        from edible.model.evaluate import compute_metrics
        m = compute_metrics([1, 0], [0, 0], 2, toxic_indices={0})
        result = EpochResult(epoch=1, train_loss=0.5, val_metrics=m, elapsed_seconds=1.0)
        assert result.toxic_fp_rate == m.toxic_fp_rate


# ---------------------------------------------------------------------------
# _get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_explicit_device_respected(self):
        cfg = TrainConfig(device=torch.device("cpu"))
        assert _get_device(cfg) == torch.device("cpu")

    def test_none_returns_valid_device(self):
        cfg = TrainConfig(device=None)
        dev = _get_device(cfg)
        assert dev.type in ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# train() — happy path
# ---------------------------------------------------------------------------

class TestTrainHappyPath:
    def test_returns_epoch_results_list(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES)
        history = train(cfg)
        assert isinstance(history, list)
        assert len(history) == cfg.epochs

    def test_epoch_numbers_sequential(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES)
        history = train(cfg)
        assert [r.epoch for r in history] == list(range(1, cfg.epochs + 1))

    def test_train_loss_is_positive(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES)
        history = train(cfg)
        for r in history:
            assert r.train_loss > 0

    def test_checkpoint_dir_created(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES)
        train(cfg)
        assert cfg.checkpoint_dir.exists()

    def test_best_accuracy_checkpoint_saved(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES)
        train(cfg)
        assert (cfg.checkpoint_dir / "best_accuracy.pt").exists()

    def test_best_safety_checkpoint_saved(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES)
        train(cfg)
        assert (cfg.checkpoint_dir / "best_safety.pt").exists()

    def test_checkpoint_contains_expected_keys(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES)
        train(cfg)
        ckpt = torch.load(cfg.checkpoint_dir / "best_accuracy.pt", weights_only=True)
        assert "epoch" in ckpt
        assert "model_state_dict" in ckpt
        assert "val_accuracy" in ckpt
        assert "toxic_fp_rate" in ckpt

    def test_elapsed_seconds_positive(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES)
        history = train(cfg)
        assert all(r.elapsed_seconds > 0 for r in history)


# ---------------------------------------------------------------------------
# train() — error conditions
# ---------------------------------------------------------------------------

class TestTrainErrors:
    def test_empty_images_dir_raises_runtime_error(self, tmp_path):
        species = _TWO_SPECIES
        db_path = tmp_path / "species.json"
        db_path.write_text(json.dumps({
            "version": "1.0", "scope": "test", "notes": "",
            "species": species,
        }))
        cfg = TrainConfig(
            images_dir=tmp_path / "empty_images",
            species_db_path=db_path,
            classifier_config=ClassifierConfig(pretrained=False, num_classes=2),
            epochs=1,
            checkpoint_dir=tmp_path / "ckpt",
        )
        # images dir doesn't exist → empty dataset → RuntimeError
        with pytest.raises(RuntimeError, match="empty"):
            train(cfg)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_early_stop_respected(self, tmp_path):
        """If toxic_fp_patience=1 and val set has toxic samples, training
        may stop before all epochs complete."""
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES, n_images=16)
        cfg.epochs = 10
        cfg.toxic_fp_patience = 1
        cfg.toxic_fp_min_delta = 1.0  # impossible to improve → always trigger
        history = train(cfg)
        # Should have stopped before epoch 10
        assert len(history) < 10


# ---------------------------------------------------------------------------
# _intra_class_cutmix
# ---------------------------------------------------------------------------

def _make_batch(B: int = 4, C: int = 3, H: int = 8, W: int = 8):
    images = torch.zeros(B, C, H, W)
    for i in range(B):
        images[i] = float(i)  # each image has a distinct fill value
    return images


class TestIntraClassCutmix:
    TOXIC = {0}

    def test_edible_images_never_modified(self):
        images = _make_batch(4)
        # labels: all edible (class 1)
        labels = torch.tensor([1, 1, 1, 1])
        original = images.clone()
        result = _intra_class_cutmix(images, labels, self.TOXIC, prob=1.0, alpha=1.0)
        assert torch.equal(result, original)

    def test_prob_zero_leaves_images_unchanged(self):
        images = _make_batch(4)
        labels = torch.tensor([0, 0, 1, 1])
        original = images.clone()
        result = _intra_class_cutmix(images, labels, self.TOXIC, prob=0.0, alpha=1.0)
        assert torch.equal(result, original)

    def test_single_toxic_image_in_batch_skipped(self):
        images = _make_batch(4)
        # Only one toxic sample — no candidate to pair with
        labels = torch.tensor([0, 1, 1, 1])
        original = images.clone()
        result = _intra_class_cutmix(images, labels, self.TOXIC, prob=1.0, alpha=1.0)
        assert torch.equal(result, original)

    def test_does_not_modify_input_tensor(self):
        images = _make_batch(4)
        labels = torch.tensor([0, 0, 1, 1])
        original = images.clone()
        _intra_class_cutmix(images, labels, self.TOXIC, prob=1.0, alpha=1.0)
        assert torch.equal(images, original)  # input is not mutated

    def test_output_shape_unchanged(self):
        images = _make_batch(4)
        labels = torch.tensor([0, 0, 1, 1])
        result = _intra_class_cutmix(images, labels, self.TOXIC, prob=1.0, alpha=1.0)
        assert result.shape == images.shape

    def test_toxic_image_modified_when_pair_exists(self):
        from unittest.mock import patch
        images = _make_batch(4)  # images[0]=0.0 (toxic), images[1]=1.0 (toxic)
        labels = torch.tensor([0, 0, 1, 1])
        # Force lam=0.0 → cut_h=H, cut_w=W → guaranteed non-empty crop from the paired image
        with patch.object(torch.distributions.Beta, "sample", return_value=torch.tensor(0.0)), \
             patch("edible.model.train.torch.rand", return_value=torch.tensor(0.0)):
            result = _intra_class_cutmix(images, labels, self.TOXIC, prob=1.0, alpha=1.0)
        assert not torch.equal(result[0], images[0]) or not torch.equal(result[1], images[1])

    def test_edible_images_untouched_when_toxic_mixed(self):
        torch.manual_seed(0)
        images = _make_batch(4)
        labels = torch.tensor([0, 0, 1, 1])
        original = images.clone()
        result = _intra_class_cutmix(images, labels, self.TOXIC, prob=1.0, alpha=1.0)
        assert torch.equal(result[2], original[2])
        assert torch.equal(result[3], original[3])

    def test_pasted_values_come_from_same_class(self):
        # Image 0 filled with 0.0, image 1 filled with 1.0 — both toxic
        images = torch.zeros(4, 3, 16, 16)
        images[1] = 1.0
        images[2] = 2.0  # edible
        images[3] = 3.0  # edible
        labels = torch.tensor([0, 0, 1, 1])
        torch.manual_seed(7)
        result = _intra_class_cutmix(images, labels, self.TOXIC, prob=1.0, alpha=0.5)
        # Toxic images (indices 0 and 1) must only contain values from the toxic pair {0.0, 1.0}
        assert result[0].max().item() <= 1.0 + 1e-6
        assert result[1].max().item() <= 1.0 + 1e-6
        # Edible images must be untouched
        assert torch.equal(result[2], images[2])
        assert torch.equal(result[3], images[3])

    def test_train_loop_accepts_cutmix_config(self, tmp_path):
        cfg = _make_train_fixture(tmp_path, _TWO_SPECIES, n_images=16)
        cfg.cutmix_prob = 0.5
        cfg.cutmix_alpha = 1.0
        history = train(cfg)
        assert len(history) == cfg.epochs


# ---------------------------------------------------------------------------
# Constant: TOXIC_FP_ALARM_THRESHOLD
# ---------------------------------------------------------------------------

class TestConstants:
    def test_alarm_threshold_is_five_percent(self):
        assert TOXIC_FP_ALARM_THRESHOLD == pytest.approx(0.05)
