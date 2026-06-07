"""Tests for image_quality blur detection and phenology constants."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from edible.data.image_quality import (
    DEFAULT_BLUR_THRESHOLD,
    INAT_FRUITING_VALUE_ID,
    INAT_PHENOLOGY_TERM_ID,
    blur_score,
    is_sharp,
)


def _make_image_bytes(mode: str = "sharp", size: int = 64) -> bytes:
    """
    Create a synthetic JPEG.

    mode='sharp'  — high-frequency checkerboard pattern → high Laplacian variance
    mode='blurry' — uniform grey → near-zero Laplacian variance
    """
    if mode == "sharp":
        arr = np.indices((size, size)).sum(axis=0) % 2 * 255
        img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    else:
        img = Image.new("RGB", (size, size), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class TestBlurScore:
    def test_sharp_image_has_high_score(self):
        score = blur_score(_make_image_bytes("sharp"))
        assert score > DEFAULT_BLUR_THRESHOLD

    def test_uniform_image_has_low_score(self):
        score = blur_score(_make_image_bytes("blurry"))
        assert score < 1.0

    def test_score_is_non_negative(self):
        assert blur_score(_make_image_bytes("sharp")) >= 0
        assert blur_score(_make_image_bytes("blurry")) >= 0

    def test_sharp_scores_higher_than_blurry(self):
        assert blur_score(_make_image_bytes("sharp")) > blur_score(_make_image_bytes("blurry"))

    def test_accepts_png_bytes(self):
        img = Image.new("RGB", (32, 32), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        score = blur_score(buf.getvalue())
        assert score >= 0


class TestIsSharp:
    def test_sharp_image_passes(self):
        assert is_sharp(_make_image_bytes("sharp")) is True

    def test_blurry_image_fails(self):
        assert is_sharp(_make_image_bytes("blurry")) is False

    def test_custom_threshold_zero_always_passes(self):
        assert is_sharp(_make_image_bytes("blurry"), threshold=0.0) is True

    def test_very_high_threshold_rejects_sharp(self):
        assert is_sharp(_make_image_bytes("sharp"), threshold=1e9) is False

    def test_threshold_boundary(self):
        score = blur_score(_make_image_bytes("sharp"))
        assert is_sharp(_make_image_bytes("sharp"), threshold=score - 1) is True
        assert is_sharp(_make_image_bytes("sharp"), threshold=score + 1) is False


class TestInatConstants:
    def test_phenology_term_id(self):
        assert INAT_PHENOLOGY_TERM_ID == 12

    def test_fruiting_value_id(self):
        assert INAT_FRUITING_VALUE_ID == 14
