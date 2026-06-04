"""
EdibleDataset — PyTorch Dataset for the Edible fine-tuning pipeline.

Directory layout expected::

    data/images/
        <species_id>/
            <filename>.jpg
            <filename>.json   ← ImageMetadata sidecar (optional)

Splits are deterministic: a stable hash of the filename drives
train/val/test assignment, so the same image always lands in the
same split regardless of insertion order.

Class indices are assigned alphabetically by species_id so they are
stable across runs and machines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from edible.data.schemas import Edibility, ImageMetadata, load_species_db

Split = Literal["train", "val", "test"]

# Default ImageNet normalisation — fine-tuning on pretrained backbone
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

DEFAULT_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# Split boundary hashes (hex digits 0-f)
# train: 0-b  (12/16 ≈ 75 %)
# val:   c-d  ( 2/16 ≈ 12.5 %)
# test:  e-f  ( 2/16 ≈ 12.5 %)
_SPLIT_BOUNDARIES: dict[Split, tuple[str, str]] = {
    "train": ("0", "b"),
    "val": ("c", "d"),
    "test": ("e", "f"),
}


def _file_split(path: Path) -> Split:
    """Deterministically assign *path* to a split using its SHA-256."""
    digest = hashlib.sha256(path.name.encode()).hexdigest()
    first = digest[0]
    if first <= "b":
        return "train"
    if first <= "d":
        return "val"
    return "test"


class SampleRecord:
    """Lightweight descriptor for one image in the dataset."""

    __slots__ = ("image_path", "species_id", "class_idx", "edibility", "metadata")

    def __init__(
        self,
        image_path: Path,
        species_id: str,
        class_idx: int,
        edibility: Edibility,
        metadata: Optional[ImageMetadata],
    ) -> None:
        self.image_path = image_path
        self.species_id = species_id
        self.class_idx = class_idx
        self.edibility = edibility
        self.metadata = metadata


class EdibleDataset(Dataset):
    """
    Parameters
    ----------
    images_dir:
        Root of the ``data/images/`` tree.
    species_db_path:
        Path to ``data/species.json``.
    split:
        One of ``"train"``, ``"val"``, ``"test"``, or ``None`` for all images.
    transform:
        Torchvision transform applied to each PIL image.  Defaults to the
        standard train or eval pipeline depending on *split*.
    image_extensions:
        File suffixes to treat as images.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        images_dir: Path,
        species_db_path: Path,
        split: Optional[Split] = None,
        transform: Optional[Callable] = None,
        image_extensions: Optional[set[str]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.split = split
        self.extensions = image_extensions or self.IMAGE_EXTENSIONS

        species_db = load_species_db(species_db_path)
        # Stable alphabetical ordering → stable class indices
        sorted_ids = sorted(s.id for s in species_db.species)
        self.class_to_idx: dict[str, int] = {sid: i for i, sid in enumerate(sorted_ids)}
        self.idx_to_class: dict[int, str] = {i: sid for sid, i in self.class_to_idx.items()}
        self._species_edibility: dict[str, Edibility] = {
            s.id: s.edibility for s in species_db.species
        }

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = DEFAULT_TRAIN_TRANSFORM
        else:
            self.transform = DEFAULT_EVAL_TRANSFORM

        self.samples: list[SampleRecord] = self._discover_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_samples(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        if not self.images_dir.exists():
            return records

        for species_dir in sorted(self.images_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            species_id = species_dir.name
            if species_id not in self.class_to_idx:
                continue  # unknown species — skip silently
            class_idx = self.class_to_idx[species_id]
            edibility = self._species_edibility[species_id]

            for img_path in sorted(species_dir.iterdir()):
                if img_path.suffix.lower() not in self.extensions:
                    continue
                if self.split and _file_split(img_path) != self.split:
                    continue
                metadata = self._load_sidecar(img_path)
                records.append(SampleRecord(img_path, species_id, class_idx, edibility, metadata))

        return records

    @staticmethod
    def _load_sidecar(img_path: Path) -> Optional[ImageMetadata]:
        sidecar = img_path.with_suffix(".json")
        if not sidecar.exists():
            return None
        try:
            data = json.loads(sidecar.read_text())
            return ImageMetadata.model_validate(data)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, sample.class_idx

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def class_weights(self) -> torch.Tensor:
        """
        Return inverse-frequency weights per class, shape ``(num_classes,)``.

        Useful for ``torch.nn.CrossEntropyLoss(weight=...)``.
        Classes not present in this split receive weight 0.
        """
        counts = torch.zeros(self.num_classes())
        for s in self.samples:
            counts[s.class_idx] += 1
        weights = torch.zeros(self.num_classes())
        for i, c in enumerate(counts):
            if c > 0:
                weights[i] = 1.0 / c
        # Normalise so weights sum to num_classes (conventional)
        total = weights.sum()
        if total > 0:
            weights = weights * self.num_classes() / total
        return weights

    def toxic_class_indices(self) -> set[int]:
        """Return the set of class indices that correspond to toxic species."""
        return {
            idx
            for sid, idx in self.class_to_idx.items()
            if self._species_edibility.get(sid) == Edibility.TOXIC
        }

    def species_ids(self) -> list[str]:
        """Ordered list of species IDs matching class index 0, 1, 2, …"""
        return [self.idx_to_class[i] for i in range(self.num_classes())]
