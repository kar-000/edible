"""
Layer 1 — Plant / Not-Plant gate.

Uses the ImageNet class hierarchy: a pretrained model's top-k predictions
are checked against a curated whitelist of plant-related synsets.  If none
of the top predictions are plant-like, the image is rejected before the
species classifier ever runs.

This is a *heuristic* gate — it is not fine-tuned.  False negatives (plant
images rejected) are acceptable; false positives (non-plant passing through)
are handled by the confidence floor in Layer 2.

Design
------
* Stateless callable: ``PlantGate(image_tensor) → GateResult``
* The ImageNet synset list is loaded lazily on first call.
* Threshold and top-k are constructor-injectable for testing.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import models

from edible.data.pipeline import GateDecision, GateResult

# ---------------------------------------------------------------------------
# ImageNet plant synset indices
# Taken from the standard ImageNet 1k label set; covers common plants,
# berries, leaves, trees, herbs visible in foraging photographs.
# ---------------------------------------------------------------------------

# Source: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a (ILSVRC synsets)
# Classes that represent plants, berries, or foraging-relevant imagery.
PLANT_SYNSET_INDICES: frozenset[int] = frozenset({
    # fruits / berries
    949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959,  # various fruits
    # flowers / plants
    985,  # daisy
    986,  # yellow lady's slipper
    987,  # corn
    988,  # acorn
    989,  # hip, rose hip
    990,  # buckeye, horse chestnut
    991,  # coral fungus
    992,  # agaric
    993,  # gyromitra
    994,  # stinkhorn mushroom
    995,  # earth star
    996,  # hen-of-the-woods
    997,  # bolete
    998,  # ear, spike, capitulum
    999,  # toilet tissue (last entry — excluded in practice)
    # leaves, vegetables, herbs
    937,  # broccoli
    938,  # cauliflower
    939,  # zucchini
    940,  # spaghetti squash
    941,  # acorn squash
    942,  # butternut squash
    943,  # cucumber
    944,  # artichoke
    945,  # bell pepper
    946,  # cardoon
    947,  # mushroom
    948,  # Granny Smith (apple)
    # trees / shrubs (bark, leaves)
    340,  # red fox — excluded conceptually but forest scenes pass
    # Additional plant-like classes
    311,  # leaf beetle (foliage context)
    # Generic outdoor / nature scenes that imply plant presence
    # (kept narrow to avoid passing true non-plants)
})

# A more curated, conservative set used at inference time
CONSERVATIVE_PLANT_INDICES: frozenset[int] = frozenset({
    937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948,
    949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959,
    985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995,
    996, 997, 998,
})


class PlantGate:
    """
    Lightweight plant / not-plant gate using a pretrained ImageNet backbone.

    Parameters
    ----------
    model:
        A pretrained torchvision model.  Defaults to MobileNetV3-Small for
        speed.  Must output ``(B, 1000)`` logits.
    plant_indices:
        Set of ImageNet class indices considered "plant-like".
    top_k:
        Number of top predictions to check.
    threshold:
        Minimum softmax probability required for a top-k prediction to
        count.  Avoids counting near-zero predictions as "plant".
    device:
        Torch device to run inference on.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        plant_indices: Optional[frozenset[int]] = None,
        top_k: int = 5,
        threshold: float = 0.05,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        if model is None:
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
        self.model = model.to(self.device).eval()
        self.plant_indices = (
            plant_indices if plant_indices is not None else CONSERVATIVE_PLANT_INDICES
        )
        self.top_k = top_k
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> GateResult:
        """
        Evaluate a single image tensor (C, H, W) or batch (B, C, H, W).

        Returns a ``GateResult`` for the *first* image if a batch is
        provided (the scraper pipeline calls this image-by-image).
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        logits = self.model(image)  # (B, 1000)
        probs = F.softmax(logits, dim=1)  # (B, 1000)

        # Use first image in batch
        probs_single = probs[0]
        plant_idx_list = list(self.plant_indices)
        plant_score = min(float(probs_single[plant_idx_list].sum()), 1.0)

        topk_probs, topk_indices = torch.topk(probs_single, self.top_k)
        is_plant = any(
            idx.item() in self.plant_indices and prob.item() >= self.threshold
            for idx, prob in zip(topk_indices, topk_probs)
        )

        decision = GateDecision.PASS if is_plant and plant_score >= 0.5 else GateDecision.REJECT
        reason = (
            "plant-like ImageNet classes detected"
            if decision == GateDecision.PASS
            else f"no plant class in top-{self.top_k} above threshold {self.threshold}"
        )
        return GateResult(
            decision=decision,
            plant_score=plant_score,
            reason=reason,
            gate_type="imagenet_category",
        )

    def batch_call(self, images: torch.Tensor) -> list[GateResult]:
        """Evaluate a batch of images, returning one GateResult per image."""
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)

        logits = self.model(images)
        probs = F.softmax(logits, dim=1)

        results = []
        plant_idx_list = list(self.plant_indices)
        for i in range(probs.shape[0]):
            p = probs[i]
            plant_score = min(float(p[plant_idx_list].sum()), 1.0)
            topk_probs, topk_indices = torch.topk(p, self.top_k)
            is_plant = any(
                idx.item() in self.plant_indices and prob.item() >= self.threshold
                for idx, prob in zip(topk_indices, topk_probs)
            )
            decision = GateDecision.PASS if is_plant and plant_score >= 0.5 else GateDecision.REJECT
            reason = (
                "plant-like ImageNet classes detected"
                if decision == GateDecision.PASS
                else f"no plant class in top-{self.top_k} above threshold {self.threshold}"
            )
            results.append(GateResult(
                decision=decision,
                plant_score=plant_score,
                reason=reason,
                gate_type="imagenet_category",
            ))
        return results
