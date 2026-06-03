"""
Nominatim reverse geocoder.

Converts GPS coordinates to county + state via OpenStreetMap Nominatim.
Free, no API key required. Usage policy: max 1 req/sec, valid User-Agent.
https://nominatim.org/release-docs/latest/api/Reverse/
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

USER_AGENT = "Edible-App/0.1 (educational foraging identifier; contact via GitHub)"
_DEFAULT_RATE_LIMIT_S = 1.1  # nominatim policy: max 1 req/sec; add small buffer


@dataclass
class GeoResult:
    lat: float
    lng: float
    county: Optional[str]
    state: Optional[str]
    state_code: Optional[str]
    country_code: str = "us"
    raw: Optional[dict] = None

    @property
    def is_texas(self) -> bool:
        return self.state_code is not None and self.state_code.upper() == "TX"


class NominatimGeocoder:
    """
    Thin wrapper around the Nominatim reverse geocoding API.

    Enforces the 1 req/sec usage policy with a configurable sleep between calls.
    Inject a custom session for testing (pre-loaded with responses mock).
    """

    BASE_URL = "https://nominatim.openstreetmap.org/reverse"

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        rate_limit_s: float = _DEFAULT_RATE_LIMIT_S,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._rate_limit_s = rate_limit_s
        self._last_request_time: float = 0.0

    def reverse_geocode(self, lat: float, lng: float) -> GeoResult:
        """
        Reverse geocode a lat/lng pair to county + state.

        Returns a GeoResult with county=None if the location cannot be resolved
        to a named county (e.g. offshore, border areas).
        """
        self._throttle()
        resp = self._session.get(
            self.BASE_URL,
            params={"lat": lat, "lon": lng, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        address = data.get("address", {})
        county = (
            address.get("county")
            or address.get("city")  # some TX cities report without county key
            or None
        )
        # Strip " County" suffix if present (Nominatim sometimes includes it)
        if county and county.lower().endswith(" county"):
            county = county[: -len(" county")]

        state_code = address.get("ISO3166-2-lvl4", "")
        if "-" in state_code:
            state_code = state_code.split("-")[-1]  # "US-TX" → "TX"

        return GeoResult(
            lat=lat,
            lng=lng,
            county=county or None,
            state=address.get("state"),
            state_code=state_code or None,
            country_code=address.get("country_code", "us"),
            raw=data,
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_s:
            time.sleep(self._rate_limit_s - elapsed)
        self._last_request_time = time.monotonic()
