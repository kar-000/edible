"""
Adversarial tests for Nominatim reverse geocoder.

Covers: happy path, rate limiting, missing address fields,
non-TX locations, HTTP errors, malformed responses.
"""

from __future__ import annotations

import time

import pytest
import responses as rsps_lib

from edible.data.geocoding import _DEFAULT_RATE_LIMIT_S, NominatimGeocoder

REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"


def _nominatim_response(county="Travis County", state="Texas", state_code="US-TX") -> dict:
    return {
        "place_id": 12345,
        "address": {
            "county": county,
            "state": state,
            "ISO3166-2-lvl4": state_code,
            "country_code": "us",
        },
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestReverseGeocodeHappyPath:
    @rsps_lib.activate
    def test_returns_county_and_state(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=_nominatim_response(), status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(30.267, -97.743)
        assert result.county == "Travis"
        assert result.state == "Texas"
        assert result.state_code == "TX"

    @rsps_lib.activate
    def test_strips_county_suffix(self):
        resp = _nominatim_response(county="Hays County")
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=resp, status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(30.0, -98.0)
        assert result.county == "Hays"

    @rsps_lib.activate
    def test_is_texas_true_for_tx(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=_nominatim_response(), status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(30.267, -97.743)
        assert result.is_texas is True

    @rsps_lib.activate
    def test_is_texas_false_for_other_state(self):
        resp = _nominatim_response(state="Oklahoma", state_code="US-OK")
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=resp, status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(34.0, -97.0)
        assert result.is_texas is False

    @rsps_lib.activate
    def test_lat_lng_preserved_in_result(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=_nominatim_response(), status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(29.4241, -98.4936)
        assert result.lat == 29.4241
        assert result.lng == -98.4936

    @rsps_lib.activate
    def test_raw_response_stored(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=_nominatim_response(), status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(30.267, -97.743)
        assert result.raw is not None
        assert "address" in result.raw

    @rsps_lib.activate
    def test_state_code_parsed_from_iso_format(self):
        resp = _nominatim_response(state_code="US-TX")
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=resp, status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(30.267, -97.743)
        assert result.state_code == "TX"


# ---------------------------------------------------------------------------
# Missing / incomplete address fields
# ---------------------------------------------------------------------------

class TestReverseGeocodeMissingFields:
    @rsps_lib.activate
    def test_missing_county_returns_none(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json={
            "address": {"state": "Texas", "ISO3166-2-lvl4": "US-TX", "country_code": "us"}
        }, status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(30.0, -97.0)
        assert result.county is None

    @rsps_lib.activate
    def test_empty_address_block(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json={"address": {}}, status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(30.0, -97.0)
        assert result.county is None
        assert result.state is None
        assert result.state_code is None
        assert result.is_texas is False

    @rsps_lib.activate
    def test_city_fallback_when_county_missing(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json={
            "address": {
                "city": "Austin",
                "state": "Texas",
                "ISO3166-2-lvl4": "US-TX",
                "country_code": "us",
            }
        }, status=200)
        g = NominatimGeocoder(rate_limit_s=0)
        result = g.reverse_geocode(30.267, -97.743)
        assert result.county == "Austin"


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------

class TestReverseGeocodeHTTPErrors:
    @rsps_lib.activate
    def test_404_raises(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, status=404)
        g = NominatimGeocoder(rate_limit_s=0)
        with pytest.raises(Exception):
            g.reverse_geocode(30.0, -97.0)

    @rsps_lib.activate
    def test_500_raises(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, status=500)
        g = NominatimGeocoder(rate_limit_s=0)
        with pytest.raises(Exception):
            g.reverse_geocode(30.0, -97.0)

    @rsps_lib.activate
    def test_connection_error_raises(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, body=ConnectionError("refused"))
        g = NominatimGeocoder(rate_limit_s=0)
        with pytest.raises(Exception):
            g.reverse_geocode(30.0, -97.0)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    @rsps_lib.activate
    def test_two_calls_respect_rate_limit(self):
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=_nominatim_response(), status=200)
        rsps_lib.add(rsps_lib.GET, REVERSE_URL, json=_nominatim_response(), status=200)
        rate = 0.05  # 50ms for test speed
        g = NominatimGeocoder(rate_limit_s=rate)
        t0 = time.monotonic()
        g.reverse_geocode(30.0, -97.0)
        g.reverse_geocode(30.1, -97.1)
        elapsed = time.monotonic() - t0
        assert elapsed >= rate, f"Rate limit not respected: {elapsed:.3f}s < {rate}s"

    def test_default_rate_limit_is_above_one_second(self):
        assert _DEFAULT_RATE_LIMIT_S >= 1.0
