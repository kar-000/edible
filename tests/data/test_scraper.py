"""
Adversarial tests for the iNaturalist scraper.

Covers: observation pagination, photo download, metadata sidecar writing,
resume (skip existing), partial failure handling, location parsing,
metadata correctness for toxic vs edible species.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import responses as rsps_lib
from PIL import Image

from edible.data.geocoding import GeoResult, NominatimGeocoder
from edible.data.range_check import UsdaPlantsClient
from edible.data.schemas import Species
from edible.data.scraper import (
    TEXAS_PLACE_ID,
    InatClient,
    ScrapeSummary,
    SpeciesScraper,
    _parse_location,
    _photo_url,
)

INAT_OBS_URL = "https://api.inaturalist.org/v1/observations"
PHOTO_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/111/medium.jpg"


def _make_sharp_jpeg() -> bytes:
    """Return a real sharp JPEG (checkerboard) that passes the blur gate."""
    arr = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_species(edibility: str = "edible_raw", species_id: str = "rubus_trivialis") -> Species:
    return Species.model_validate({
        "id": species_id,
        "common_name": "Dewberry",
        "scientific_name": "Rubus trivialis",
        "family": "Rosaceae",
        "edibility": edibility,
        "is_target_class": True,
        "priority_reason": "test",
    })


def _make_observation(obs_id: int = 1, photo_id: int = 111, lat: str = "30.267",
                      lng: str = "-97.743") -> dict:
    return {
        "id": obs_id,
        "observed_on": "2024-08-01",
        "location": f"{lat},{lng}",
        "photos": [
            {
                "id": photo_id,
                "url": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/square.jpg",
            }
        ],
        "quality_grade": "research",
    }


def _inat_response(observations: list[dict], total: int | None = None) -> dict:
    return {
        "total_results": total if total is not None else len(observations),
        "page": 1,
        "per_page": 200,
        "results": observations,
    }


def _make_geocoder(county: str = "Travis", state: str = "Texas",
                   state_code: str = "TX") -> NominatimGeocoder:
    geocoder = MagicMock(spec=NominatimGeocoder)
    geocoder.reverse_geocode.return_value = GeoResult(
        lat=30.267, lng=-97.743,
        county=county, state=state, state_code=state_code,
    )
    return geocoder


def _make_range_client(present: bool = True) -> UsdaPlantsClient:
    client = MagicMock(spec=UsdaPlantsClient)
    client.is_present_in_state.return_value = present
    return client


# ---------------------------------------------------------------------------
# _photo_url helper
# ---------------------------------------------------------------------------

class TestPhotoUrl:
    def test_replaces_square_with_medium(self):
        url = "https://example.com/photos/1/square.jpg"
        assert _photo_url(url) == "https://example.com/photos/1/medium.jpg"

    def test_replaces_small_with_medium(self):
        url = "https://example.com/photos/1/small.jpg"
        assert _photo_url(url) == "https://example.com/photos/1/medium.jpg"

    def test_replaces_large_with_medium(self):
        url = "https://example.com/photos/1/large.jpg"
        assert _photo_url(url) == "https://example.com/photos/1/medium.jpg"

    def test_custom_target_size(self):
        url = "https://example.com/photos/1/square.jpg"
        assert _photo_url(url, size="original") == "https://example.com/photos/1/original.jpg"

    def test_url_with_no_known_size_returned_unchanged(self):
        url = "https://example.com/photos/1/image.jpg"
        assert _photo_url(url) == url


# ---------------------------------------------------------------------------
# _parse_location helper
# ---------------------------------------------------------------------------

class TestParseLocation:
    def test_valid_location_string(self):
        obs = {"location": "30.267,-97.743"}
        lat, lng = _parse_location(obs)
        assert lat == pytest.approx(30.267)
        assert lng == pytest.approx(-97.743)

    def test_missing_location_returns_none(self):
        lat, lng = _parse_location({})
        assert lat is None and lng is None

    def test_none_location_returns_none(self):
        lat, lng = _parse_location({"location": None})
        assert lat is None and lng is None

    def test_malformed_location_returns_none(self):
        lat, lng = _parse_location({"location": "not_a_coordinate"})
        assert lat is None and lng is None

    def test_negative_coordinates_parsed(self):
        lat, lng = _parse_location({"location": "-33.87,151.21"})
        assert lat == pytest.approx(-33.87)
        assert lng == pytest.approx(151.21)


# ---------------------------------------------------------------------------
# InatClient — observation pagination
# ---------------------------------------------------------------------------

class TestInatClientObservations:
    @rsps_lib.activate
    def test_yields_observations_with_photos(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([_make_observation(1), _make_observation(2)]),
                     status=200)
        client = InatClient(rate_limit_s=0)
        results = list(client.iter_observations("Rubus trivialis", max_results=10))
        assert len(results) == 2

    @rsps_lib.activate
    def test_skips_observations_without_photos(self):
        obs_no_photo = {**_make_observation(1), "photos": []}
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([obs_no_photo, _make_observation(2)]),
                     status=200)
        client = InatClient(rate_limit_s=0)
        results = list(client.iter_observations("Rubus trivialis", max_results=10))
        assert len(results) == 1
        assert results[0]["id"] == 2

    @rsps_lib.activate
    def test_respects_max_results(self):
        obs_list = [_make_observation(i, photo_id=i) for i in range(1, 6)]
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response(obs_list), status=200)
        client = InatClient(rate_limit_s=0)
        results = list(client.iter_observations("Rubus trivialis", max_results=3))
        assert len(results) == 3

    @rsps_lib.activate
    def test_stops_on_empty_page(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([]), status=200)
        client = InatClient(rate_limit_s=0)
        results = list(client.iter_observations("Rubus trivialis", max_results=100))
        assert results == []

    @rsps_lib.activate
    def test_uses_texas_place_id_by_default(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([_make_observation(1)]), status=200)
        client = InatClient(rate_limit_s=0)
        list(client.iter_observations("Rubus trivialis", max_results=10))
        assert str(TEXAS_PLACE_ID) in rsps_lib.calls[0].request.url

    @rsps_lib.activate
    def test_place_id_none_omits_place_filter(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([_make_observation(1)]), status=200)
        client = InatClient(rate_limit_s=0)
        list(client.iter_observations("Menispermum canadense", place_id=None, max_results=1))
        assert "place_id" not in rsps_lib.calls[0].request.url

    @rsps_lib.activate
    def test_fruiting_only_adds_term_params(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([_make_observation(1)]), status=200)
        client = InatClient(rate_limit_s=0)
        list(client.iter_observations("Rubus trivialis", fruiting_only=True, max_results=1))
        url = rsps_lib.calls[0].request.url
        assert "term_id=12" in url
        assert "term_value_id=14" in url

    @rsps_lib.activate
    def test_fruiting_only_false_omits_term_params(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([_make_observation(1)]), status=200)
        client = InatClient(rate_limit_s=0)
        list(client.iter_observations("Rubus trivialis", fruiting_only=False, max_results=1))
        url = rsps_lib.calls[0].request.url
        assert "term_id" not in url

    @rsps_lib.activate
    def test_http_error_raises(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL, status=429)
        client = InatClient(rate_limit_s=0)
        with pytest.raises(Exception):
            list(client.iter_observations("Rubus trivialis", max_results=10))

    @rsps_lib.activate
    def test_api_token_sets_auth_header(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([]), status=200)
        client = InatClient(api_token="test-token-abc", rate_limit_s=0)
        list(client.iter_observations("Rubus trivialis", max_results=1))
        assert "Bearer test-token-abc" in rsps_lib.calls[0].request.headers.get("Authorization", "")

    @rsps_lib.activate
    def test_no_token_has_no_auth_header(self):
        rsps_lib.add(rsps_lib.GET, INAT_OBS_URL,
                     json=_inat_response([]), status=200)
        client = InatClient(rate_limit_s=0)
        list(client.iter_observations("Rubus trivialis", max_results=1))
        assert "Authorization" not in rsps_lib.calls[0].request.headers


# ---------------------------------------------------------------------------
# SpeciesScraper — full pipeline
# ---------------------------------------------------------------------------

class TestSpeciesScraper:
    def _make_scraper(self, output_dir: Path, geocoder=None, range_client=None,
                      inat_client=None) -> SpeciesScraper:
        return SpeciesScraper(
            client=inat_client or MagicMock(spec=InatClient),
            geocoder=geocoder or _make_geocoder(),
            range_client=range_client or _make_range_client(),
            output_dir=output_dir,
        )

    @rsps_lib.activate
    def test_downloads_image_and_writes_sidecar(self, tmp_path):
        obs = _make_observation(obs_id=9001, photo_id=111)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = _make_sharp_jpeg()

        scraper = self._make_scraper(tmp_path, inat_client=client)
        species = _make_species()
        summary = scraper.scrape(species, max_images=1)

        img = tmp_path / "rubus_trivialis" / "9001_111.jpg"
        meta = tmp_path / "rubus_trivialis" / "9001_111.json"
        assert img.exists()
        assert meta.exists()
        assert summary.downloaded == 1
        assert summary.failed == 0

    @rsps_lib.activate
    def test_sidecar_metadata_is_valid(self, tmp_path):
        obs = _make_observation(obs_id=9001, photo_id=111)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = _make_sharp_jpeg()

        scraper = self._make_scraper(tmp_path, inat_client=client)
        scraper.scrape(_make_species(), max_images=1)

        meta_path = tmp_path / "rubus_trivialis" / "9001_111.json"
        data = json.loads(meta_path.read_text())
        assert data["species_scientific"] == "Rubus trivialis"
        assert data["edibility"] == "edible_raw"
        assert data["source"] == "iNaturalist"
        assert data["inat_observation_id"] == 9001
        assert data["confidence_in_label"] == "research_grade"

    def test_toxic_species_metadata_marks_toxic(self, tmp_path):
        obs = _make_observation(obs_id=9002, photo_id=222)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = _make_sharp_jpeg()

        scraper = self._make_scraper(tmp_path, inat_client=client)
        species = Species.model_validate({
            "id": "phytolacca_americana",
            "common_name": "Pokeweed",
            "scientific_name": "Phytolacca americana",
            "family": "Phytolaccaceae",
            "edibility": "toxic",
            "is_target_class": True,
            "priority_reason": "test",
        })
        scraper.scrape(species, max_images=1)

        meta_path = tmp_path / "phytolacca_americana" / "9002_222.json"
        data = json.loads(meta_path.read_text())
        assert data["edibility"] == "toxic"

    def test_skips_existing_files(self, tmp_path):
        obs = _make_observation(obs_id=9003, photo_id=333)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])

        species_dir = tmp_path / "rubus_trivialis"
        species_dir.mkdir()
        (species_dir / "9003_333.jpg").write_bytes(b"existing")
        (species_dir / "9003_333.json").write_text("{}")

        scraper = self._make_scraper(tmp_path, inat_client=client)
        summary = scraper.scrape(_make_species(), max_images=10)

        client.download_bytes.assert_not_called()
        assert summary.skipped_existing == 1
        assert summary.downloaded == 0

    def test_failed_download_counted_not_raised(self, tmp_path):
        obs = _make_observation(obs_id=9004, photo_id=444)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.side_effect = ConnectionError("network error")

        scraper = self._make_scraper(tmp_path, inat_client=client)
        summary = scraper.scrape(_make_species(), max_images=1)

        assert summary.failed == 1
        assert summary.downloaded == 0
        assert len(summary.errors) == 1
        # Partial files must be cleaned up
        assert not (tmp_path / "rubus_trivialis" / "9004_444.jpg").exists()

    def test_partial_image_file_cleaned_up_on_failure(self, tmp_path):
        obs = _make_observation(obs_id=9005, photo_id=555)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.side_effect = RuntimeError("halfway through")

        scraper = self._make_scraper(tmp_path, inat_client=client)
        scraper.scrape(_make_species(), max_images=1)

        assert not (tmp_path / "rubus_trivialis" / "9005_555.jpg").exists()
        assert not (tmp_path / "rubus_trivialis" / "9005_555.json").exists()

    def test_geocoding_failure_does_not_abort_scrape(self, tmp_path):
        obs = _make_observation(obs_id=9006, photo_id=666)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = _make_sharp_jpeg()

        bad_geocoder = MagicMock(spec=NominatimGeocoder)
        bad_geocoder.reverse_geocode.side_effect = RuntimeError("nominatim down")

        scraper = self._make_scraper(tmp_path, geocoder=bad_geocoder, inat_client=client)
        summary = scraper.scrape(_make_species(), max_images=1)

        assert summary.downloaded == 1  # download should still succeed
        meta = json.loads((tmp_path / "rubus_trivialis" / "9006_666.json").read_text())
        assert meta["county"] is None

    def test_usda_failure_does_not_abort_scrape(self, tmp_path):
        obs = _make_observation(obs_id=9007, photo_id=777)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = _make_sharp_jpeg()

        bad_range = MagicMock(spec=UsdaPlantsClient)
        bad_range.is_present_in_state.side_effect = RuntimeError("usda down")

        scraper = self._make_scraper(tmp_path, range_client=bad_range, inat_client=client)
        summary = scraper.scrape(_make_species(), max_images=1)

        assert summary.downloaded == 1

    def test_observation_without_location_has_none_lat_lng(self, tmp_path):
        obs = {**_make_observation(obs_id=9008, photo_id=888), "location": None}
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = _make_sharp_jpeg()

        scraper = self._make_scraper(tmp_path, inat_client=client)
        scraper.scrape(_make_species(), max_images=1)

        meta = json.loads((tmp_path / "rubus_trivialis" / "9008_888.json").read_text())
        assert meta["lat"] is None
        assert meta["lng"] is None

    def test_success_rate_calculation(self):
        s = ScrapeSummary(
            "test", "Test sp", 10,
            downloaded=8, skipped_existing=0, skipped_blurry=0,
            skipped_not_fruiting=0, failed=2,
        )
        assert s.success_rate == pytest.approx(0.8)

    def test_blurry_image_skipped(self, tmp_path):
        import io as _io

        from PIL import Image as _Image
        # Uniform grey → near-zero Laplacian variance → blurry
        buf = _io.BytesIO()
        _Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buf, format="JPEG")
        blurry_bytes = buf.getvalue()

        obs = _make_observation(obs_id=9900, photo_id=900)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = blurry_bytes

        scraper = self._make_scraper(tmp_path, inat_client=client)
        summary = scraper.scrape(_make_species(), max_images=1, blur_threshold=100.0)

        assert summary.skipped_blurry == 1
        assert summary.downloaded == 0
        assert not (tmp_path / "rubus_trivialis" / "9900_900.jpg").exists()

    def test_sharp_image_not_rejected(self, tmp_path):
        import io as _io

        import numpy as _np
        from PIL import Image as _Image
        # Checkerboard → high Laplacian variance → sharp
        arr = (_np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(_np.uint8)
        img = _Image.fromarray(arr, mode="L").convert("RGB")
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=95)

        obs = _make_observation(obs_id=9901, photo_id=901)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = buf.getvalue()

        scraper = self._make_scraper(tmp_path, inat_client=client)
        summary = scraper.scrape(_make_species(), max_images=1, blur_threshold=100.0)

        assert summary.skipped_blurry == 0
        assert summary.downloaded == 1

    def test_blur_threshold_zero_disables_filtering(self, tmp_path):
        import io as _io

        from PIL import Image as _Image
        buf = _io.BytesIO()
        _Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buf, format="JPEG")

        obs = _make_observation(obs_id=9902, photo_id=902)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = buf.getvalue()

        scraper = self._make_scraper(tmp_path, inat_client=client)
        summary = scraper.scrape(_make_species(), max_images=1, blur_threshold=0.0)

        assert summary.skipped_blurry == 0
        assert summary.downloaded == 1

    def test_fruiting_only_forwarded_to_iter_observations(self, tmp_path):
        obs = _make_observation(obs_id=9903, photo_id=903)
        client = MagicMock(spec=InatClient)
        client.iter_observations.return_value = iter([obs])
        client.download_bytes.return_value = _make_sharp_jpeg()

        scraper = self._make_scraper(tmp_path, inat_client=client)
        scraper.scrape(_make_species(), max_images=1, fruiting_only=True, blur_threshold=0.0)

        call_kwargs = client.iter_observations.call_args
        assert call_kwargs.kwargs.get("fruiting_only") is True

    def test_success_rate_zero_when_no_attempts(self):
        s = ScrapeSummary(
            "test", "Test sp", 0,
            downloaded=0, skipped_existing=0, skipped_blurry=0,
            skipped_not_fruiting=0, failed=0,
        )
        assert s.success_rate == 0.0


# ---------------------------------------------------------------------------
# InatClient.from_env
# ---------------------------------------------------------------------------

class TestFromEnv:
    def test_uses_api_token_from_env(self, monkeypatch):
        monkeypatch.setenv("INAT_API_TOKEN", "my-token-xyz")
        client = InatClient.from_env()
        assert "my-token-xyz" in client._session.headers.get("Authorization", "")

    def test_no_token_env_gives_slower_rate(self, monkeypatch):
        monkeypatch.delenv("INAT_API_TOKEN", raising=False)
        client = InatClient.from_env()
        assert client._rate_limit_s >= 1.0

    def test_token_env_gives_faster_rate(self, monkeypatch):
        monkeypatch.setenv("INAT_API_TOKEN", "some-token")
        client = InatClient.from_env()
        assert client._rate_limit_s < 1.0
