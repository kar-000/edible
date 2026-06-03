"""
iNaturalist scraper for the Edible data pipeline.

Downloads research-grade observations with photos for each target species,
enriches each image with Nominatim geocoding + USDA range confirmation,
and writes a JSON metadata sidecar alongside every image.

Output layout:
  data/images/{species_id}/{observation_id}_{photo_id}.jpg
  data/images/{species_id}/{observation_id}_{photo_id}.json

Auth note:
  The iNaturalist v1 API is publicly accessible for read-only queries.
  Set INAT_API_TOKEN in your environment for authenticated requests
  (higher rate limits: 200 req/min vs 60 req/min unauthenticated).
  Get your token at: https://www.inaturalist.org/users/api_token
  (INAT_ACCT / INAT_PW are not used here — iNat does not support
  simple password auth on the API; the token must be obtained via
  the web UI or OAuth2 app registration.)

Rate limits:
  Unauthenticated: ~60 req/min   → rate_limit_s = 1.0
  Authenticated:   ~200 req/min  → rate_limit_s = 0.3
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import requests

from edible.data.geocoding import USER_AGENT, NominatimGeocoder
from edible.data.range_check import UsdaPlantsClient
from edible.data.schemas import ImageMetadata, LabelConfidence, Species

logger = logging.getLogger(__name__)

# iNaturalist place ID for Texas (US state, place_type=8).
# Confirmed via: GET /v1/places/autocomplete?q=Texas&geo=true → id=18
TEXAS_PLACE_ID = 18

_INAT_BASE = "https://api.inaturalist.org/v1"
_PHOTO_SIZE = "medium"  # options: square, small, medium, large, original

# iNat caps per_page at 200
_MAX_PER_PAGE = 200


@dataclass
class ScrapeSummary:
    species_id: str
    scientific_name: str
    requested: int
    downloaded: int
    skipped_existing: int
    failed: int
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.downloaded + self.failed
        return self.downloaded / total if total else 0.0


class InatClient:
    """
    Thin wrapper around the iNaturalist v1 REST API.

    Paginates through observations and yields raw observation dicts.
    Handles rate limiting and optional Bearer token auth.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        session: Optional[requests.Session] = None,
        rate_limit_s: float = 1.0,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        if api_token:
            self._session.headers.update({"Authorization": f"Bearer {api_token}"})
        self._rate_limit_s = rate_limit_s
        self._last_request_time: float = 0.0

    @classmethod
    def from_env(cls) -> InatClient:
        """
        Construct from environment variables.
        Uses INAT_API_TOKEN if set, otherwise unauthenticated.
        """
        token = os.environ.get("INAT_API_TOKEN")
        rate = 0.3 if token else 1.0
        return cls(api_token=token, rate_limit_s=rate)

    def iter_observations(
        self,
        taxon_name: str,
        place_id: int = TEXAS_PLACE_ID,
        quality_grade: str = "research",
        max_results: int = 1000,
    ) -> Iterator[dict]:
        """
        Yield raw observation dicts, paginating until max_results or exhausted.
        Only yields observations that have at least one photo.
        """
        yielded = 0
        page = 1

        while yielded < max_results:
            self._throttle()
            per_page = min(_MAX_PER_PAGE, max_results - yielded)
            resp = self._session.get(
                f"{_INAT_BASE}/observations",
                params={
                    "taxon_name": taxon_name,
                    "place_id": place_id,
                    "quality_grade": quality_grade,
                    "photos": "true",
                    "per_page": per_page,
                    "page": page,
                    "order": "desc",
                    "order_by": "created_at",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for obs in results:
                if obs.get("photos"):
                    yield obs
                    yielded += 1
                    if yielded >= max_results:
                        return

            if len(results) < per_page:
                break  # last page

            page += 1

    def download_bytes(self, url: str, timeout: int = 30) -> bytes:
        """Download raw bytes from a URL (photo or otherwise)."""
        self._throttle()
        resp = self._session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_s:
            time.sleep(self._rate_limit_s - elapsed)
        self._last_request_time = time.monotonic()


def _photo_url(url: str, size: str = _PHOTO_SIZE) -> str:
    """
    Build a sized photo URL from an iNaturalist photo URL string.
    iNat photo URLs have the form: .../photos/{id}/square.jpg
    We replace the size segment.
    """
    for candidate in ("square", "small", "medium", "large", "original"):
        if f"/{candidate}." in url:
            return url.replace(f"/{candidate}.", f"/{size}.")
    return url


def _parse_location(obs: dict) -> tuple[Optional[float], Optional[float]]:
    """Parse lat/lng from iNaturalist observation dict."""
    location = obs.get("location")
    if not location:
        return None, None
    try:
        lat_s, lng_s = location.split(",")
        return float(lat_s), float(lng_s)
    except (ValueError, AttributeError):
        return None, None


class SpeciesScraper:
    """
    Orchestrates scraping for a single species:
      1. Paginate iNaturalist for research-grade observations
      2. Download the first (best) photo per observation
      3. Reverse-geocode the location
      4. Confirm USDA range
      5. Write image + JSON metadata sidecar

    Designed to be called once per species so that multiple instances
    can run in parallel (one per species) via Dynamic Workflows.
    """

    def __init__(
        self,
        client: InatClient,
        geocoder: NominatimGeocoder,
        range_client: UsdaPlantsClient,
        output_dir: Path,
    ) -> None:
        self._client = client
        self._geocoder = geocoder
        self._range_client = range_client
        self._output_dir = output_dir

    def scrape(
        self,
        species: Species,
        max_images: int = 1000,
        place_id: int = TEXAS_PLACE_ID,
    ) -> ScrapeSummary:
        """
        Scrape up to max_images for the given species.
        Skips images that already exist on disk (resume-safe).
        """
        species_dir = self._output_dir / species.id
        species_dir.mkdir(parents=True, exist_ok=True)

        summary = ScrapeSummary(
            species_id=species.id,
            scientific_name=species.scientific_name,
            requested=max_images,
            downloaded=0,
            skipped_existing=0,
            failed=0,
        )

        for obs in self._client.iter_observations(
            taxon_name=species.scientific_name,
            place_id=place_id,
            max_results=max_images,
        ):
            obs_id = obs["id"]
            photos = obs.get("photos", [])
            if not photos:
                continue

            photo = photos[0]
            photo_id = photo.get("id", "unknown")
            stem = f"{obs_id}_{photo_id}"
            img_path = species_dir / f"{stem}.jpg"
            meta_path = species_dir / f"{stem}.json"

            if img_path.exists() and meta_path.exists():
                summary.skipped_existing += 1
                logger.debug("Skipping existing: %s", img_path.name)
                continue

            try:
                url = _photo_url(photo.get("url", ""))
                img_bytes = self._client.download_bytes(url)
                img_path.write_bytes(img_bytes)

                metadata = self._build_metadata(obs, species)
                meta_path.write_text(
                    json.dumps(metadata.model_dump(mode="json"), indent=2),
                    encoding="utf-8",
                )
                summary.downloaded += 1
                logger.info(
                    "[%s] %d/%d downloaded",
                    species.id,
                    summary.downloaded,
                    max_images,
                )
            except Exception as exc:
                summary.failed += 1
                msg = f"obs {obs_id}: {exc}"
                summary.errors.append(msg)
                logger.warning("Failed %s", msg)
                # Clean up partial files
                img_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)

        return summary

    def _build_metadata(self, obs: dict, species: Species) -> ImageMetadata:
        lat, lng = _parse_location(obs)
        county = None
        state = None
        state_code = None
        usda_confirmed: Optional[bool] = None

        if lat is not None and lng is not None:
            try:
                geo = self._geocoder.reverse_geocode(lat, lng)
                county = geo.county
                state = geo.state
                state_code = geo.state_code
            except Exception as exc:
                logger.warning("Geocoding failed for obs %s: %s", obs["id"], exc)

        if state_code and state_code.upper() == "TX":
            try:
                usda_confirmed = self._range_client.is_present_in_state(
                    species.scientific_name, "TX"
                )
            except Exception as exc:
                logger.warning("USDA check failed for %s: %s", species.scientific_name, exc)

        # iNaturalist research-grade = community-validated
        confidence = LabelConfidence.RESEARCH_GRADE

        return ImageMetadata(
            species_scientific=species.scientific_name,
            species_common=species.common_name,
            edibility=species.edibility,
            lat=lat,
            lng=lng,
            county=county,
            state=state or "TX",
            state_code=state_code or "TX",
            usda_confirmed_range=usda_confirmed,
            confidence_in_label=confidence,
            date_observed=obs.get("observed_on"),
            source="iNaturalist",
            image_quality_flag=None,
            inat_observation_id=obs["id"],
        )
