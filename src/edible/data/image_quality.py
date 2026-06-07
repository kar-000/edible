"""
Image quality gates for the Edible scraping pipeline.

Provides a sharpness (blur) detector based on Laplacian variance —
a standard technique: a blurry image has low high-frequency content,
so the variance of the Laplacian kernel response is low.

Calibrated for iNaturalist "medium" size photos (~500px wide).
Threshold of 100.0 was chosen to reject clearly blurry images while
keeping mildly soft ones. Tune downward to be more permissive.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# iNat annotation term IDs for phenology filtering.
# Verified via GET /v1/controlled_terms on 2026-06-07.
INAT_PHENOLOGY_TERM_ID = 12       # "Flowers and Fruits"
INAT_FRUITING_VALUE_ID = 14       # "Fruits or Seeds"

# Default sharpness threshold for iNat medium images (~500px).
DEFAULT_BLUR_THRESHOLD = 100.0

# 3×3 Laplacian kernel
def blur_score(image_bytes: bytes) -> float:
    """
    Return the Laplacian variance of an image loaded from raw bytes.

    Higher = sharper. A value below ``DEFAULT_BLUR_THRESHOLD`` indicates
    a blurry image that should be rejected from the training set.

    Laplacian approximation (no scipy needed):
      L[i,j] = up + down + left + right - 4*center
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32)
    lap = (
        arr[2:, 1:-1] + arr[:-2, 1:-1]
        + arr[1:-1, 2:] + arr[1:-1, :-2]
        - 4 * arr[1:-1, 1:-1]
    )
    return float(lap.var())


def is_sharp(image_bytes: bytes, threshold: float = DEFAULT_BLUR_THRESHOLD) -> bool:
    """Return True if the image is sharp enough to use for training."""
    return blur_score(image_bytes) >= threshold
