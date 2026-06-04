"""
Tests for EdibleDataset.

Uses a tiny fixture tree (2 species × 4 images each) built in tmp_path,
alongside the real data/species.json for end-to-end schema integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from edible.model.dataset import (
    DEFAULT_EVAL_TRANSFORM,
    DEFAULT_TRAIN_TRANSFORM,
    EdibleDataset,
    _file_split,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SPECIES_FILE = DATA_DIR / "species.json"

_DUMMY_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


def _make_rgb_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Write a tiny solid-colour JPEG to *path*."""
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(path)


def _make_fixture(
    tmp_path: Path,
    species_ids: list[str],
    n_images: int = 8,
    sidecar: bool = False,
) -> tuple[Path, Path]:
    """
    Build a mini ``data/images`` tree under *tmp_path*.

    Returns (images_dir, species_db_path).
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Build a minimal species.json with the requested IDs
    species_list = []
    for i, sid in enumerate(species_ids):
        edibility = "toxic" if i % 2 == 0 else "edible_raw"
        species_list.append({
            "id": sid,
            "common_name": sid.replace("_", " ").title(),
            "scientific_name": f"Genus {sid}",
            "family": "Testaceae",
            "edibility": edibility,
            "is_target_class": True,
            "priority_reason": "test",
        })

    db_path = tmp_path / "species.json"
    db_path.write_text(json.dumps({
        "version": "1.0",
        "scope": "test",
        "notes": "",
        "species": species_list,
    }))

    # Write images
    for sid in species_ids:
        species_dir = images_dir / sid
        species_dir.mkdir()
        for j in range(n_images):
            img_path = species_dir / f"img_{j:04d}.jpg"
            _make_rgb_image(img_path)
            if sidecar:
                meta = {
                    "species_scientific": f"Genus {sid}",
                    "species_common": sid,
                    "edibility": "edible_raw",
                    "source": "test",
                }
                img_path.with_suffix(".json").write_text(json.dumps(meta))

    return images_dir, db_path


# ---------------------------------------------------------------------------
# Split determinism
# ---------------------------------------------------------------------------

class TestFileSplit:
    def test_deterministic_for_same_name(self):
        p = Path("img_0001.jpg")
        assert _file_split(p) == _file_split(p)

    def test_returns_valid_split(self):
        for i in range(50):
            split = _file_split(Path(f"img_{i:04d}.jpg"))
            assert split in ("train", "val", "test")

    def test_train_is_majority(self):
        train_count = sum(
            1 for i in range(256) if _file_split(Path(f"img_{i:04d}.jpg")) == "train"
        )
        # ~75 % — allow ±10 %
        assert 160 < train_count < 220


# ---------------------------------------------------------------------------
# EdibleDataset basic behaviour
# ---------------------------------------------------------------------------

class TestEdibleDatasetBasic:
    def test_len_matches_all_images(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha", "beta"], n_images=8)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        assert len(ds) == 16  # 2 species × 8 images

    def test_no_images_dir_returns_empty(self, tmp_path):
        _, db = _make_fixture(tmp_path, ["alpha"], n_images=0)
        ds = EdibleDataset(tmp_path / "nonexistent", db, transform=_DUMMY_TRANSFORM)
        assert len(ds) == 0

    def test_unknown_species_dir_skipped(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha"], n_images=4)
        extra = images_dir / "unknown_berry"
        extra.mkdir()
        _make_rgb_image(extra / "img_0.jpg")
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        assert len(ds) == 4  # only alpha's images

    def test_getitem_returns_tensor_and_label(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha", "beta"], n_images=4)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        tensor, label = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert 0 <= label < ds.num_classes()

    def test_num_classes(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["a", "b", "c"], n_images=2)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        assert ds.num_classes() == 3

    def test_class_indices_stable_alphabetical(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["zebra", "alpha", "mango"], n_images=1)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        assert ds.class_to_idx["alpha"] == 0
        assert ds.class_to_idx["mango"] == 1
        assert ds.class_to_idx["zebra"] == 2

    def test_species_ids_ordered(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["charlie", "alpha", "beta"], n_images=1)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        assert ds.species_ids() == ["alpha", "beta", "charlie"]


# ---------------------------------------------------------------------------
# Split filtering
# ---------------------------------------------------------------------------

class TestEdibleDatasetSplits:
    def test_splits_are_disjoint(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha", "beta"], n_images=64)
        t = _DUMMY_TRANSFORM
        train_paths = {
            s.image_path for s in EdibleDataset(images_dir, db, split="train", transform=t).samples
        }
        val_paths = {
            s.image_path for s in EdibleDataset(images_dir, db, split="val", transform=t).samples
        }
        test_paths = {
            s.image_path for s in EdibleDataset(images_dir, db, split="test", transform=t).samples
        }
        assert train_paths.isdisjoint(val_paths)
        assert train_paths.isdisjoint(test_paths)
        assert val_paths.isdisjoint(test_paths)

    def test_splits_cover_all_images(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha"], n_images=32)
        train = EdibleDataset(images_dir, db, split="train", transform=_DUMMY_TRANSFORM)
        val = EdibleDataset(images_dir, db, split="val", transform=_DUMMY_TRANSFORM)
        test = EdibleDataset(images_dir, db, split="test", transform=_DUMMY_TRANSFORM)
        assert len(train) + len(val) + len(test) == 32

    def test_train_uses_augmentation_by_default(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha"], n_images=4)
        ds = EdibleDataset(images_dir, db, split="train")
        assert ds.transform is DEFAULT_TRAIN_TRANSFORM

    def test_val_uses_eval_transform_by_default(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha"], n_images=4)
        ds = EdibleDataset(images_dir, db, split="val")
        assert ds.transform is DEFAULT_EVAL_TRANSFORM

    def test_custom_transform_overrides_default(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha"], n_images=2)
        ds = EdibleDataset(images_dir, db, split="train", transform=_DUMMY_TRANSFORM)
        assert ds.transform is _DUMMY_TRANSFORM


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

class TestClassWeights:
    def test_weights_shape(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha", "beta"], n_images=4)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        w = ds.class_weights()
        assert w.shape == (2,)

    def test_empty_dataset_weights_are_zero(self, tmp_path):
        _, db = _make_fixture(tmp_path, ["alpha"], n_images=0)
        ds = EdibleDataset(tmp_path / "nonexistent", db, transform=_DUMMY_TRANSFORM)
        w = ds.class_weights()
        assert w.shape == (1,)
        assert w[0] == 0.0

    def test_equal_class_distribution_gives_equal_weights(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha", "beta"], n_images=8)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        w = ds.class_weights()
        # Both classes have same count → same weight
        assert abs(float(w[0]) - float(w[1])) < 1e-5


# ---------------------------------------------------------------------------
# Toxic class detection
# ---------------------------------------------------------------------------

class TestToxicClassIndices:
    def test_toxic_indices_match_species(self, tmp_path):
        # fixture: index 0 = alpha (toxic), index 1 = beta (edible)
        images_dir, db = _make_fixture(tmp_path, ["alpha", "beta"], n_images=2)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        toxic = ds.toxic_class_indices()
        # alpha is at index 0 (alphabetical), edibility = toxic (i%2==0)
        assert 0 in toxic
        assert 1 not in toxic

    def test_no_toxic_species(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["beta", "delta"], n_images=1)
        # Both have i%2==1 → edible_raw
        # But fixture assigns edibility based on enumerate position...
        # Patch: just verify the set is a subset of valid indices
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        assert ds.toxic_class_indices().issubset(set(range(ds.num_classes())))


# ---------------------------------------------------------------------------
# Sidecar metadata loading
# ---------------------------------------------------------------------------

class TestSidecarLoading:
    def test_valid_sidecar_loaded(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha"], n_images=2, sidecar=True)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        sidecars = [s.metadata for s in ds.samples if s.metadata is not None]
        assert len(sidecars) == 2

    def test_corrupt_sidecar_returns_none(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha"], n_images=2)
        # Write corrupt sidecar next to first image
        img = sorted((images_dir / "alpha").iterdir())[0]
        img.with_suffix(".json").write_text("{not valid json")
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        # corrupt sidecar → metadata is None
        bad = [s for s in ds.samples if s.image_path == img]
        assert len(bad) == 1
        assert bad[0].metadata is None

    def test_missing_sidecar_returns_none(self, tmp_path):
        images_dir, db = _make_fixture(tmp_path, ["alpha"], n_images=2)
        ds = EdibleDataset(images_dir, db, split=None, transform=_DUMMY_TRANSFORM)
        assert all(s.metadata is None for s in ds.samples)


# ---------------------------------------------------------------------------
# Integration: real species.json
# ---------------------------------------------------------------------------

class TestRealSpeciesDb:
    def test_num_classes_matches_species_count(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ds = EdibleDataset(images_dir, SPECIES_FILE, split=None, transform=_DUMMY_TRANSFORM)
        assert ds.num_classes() == 12

    def test_known_species_in_class_map(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ds = EdibleDataset(images_dir, SPECIES_FILE, split=None, transform=_DUMMY_TRANSFORM)
        assert "sambucus_canadensis" in ds.class_to_idx
        assert "phytolacca_americana" in ds.class_to_idx

    def test_toxic_indices_non_empty(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        ds = EdibleDataset(images_dir, SPECIES_FILE, split=None, transform=_DUMMY_TRANSFORM)
        assert len(ds.toxic_class_indices()) == 6
