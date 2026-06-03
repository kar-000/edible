"""
USDA PLANTS state-level range check.

Confirms whether a species is recorded in Texas using the USDA PLANTS
Web Services API. Used to set `usda_confirmed_range` on image metadata.

County-level data (BONAP) is a Phase 2+ enhancement — state-level is
sufficient for v1 metadata quality tracking.

USDA PLANTS Web Services: https://plantsservices.sc.egov.usda.gov/
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from edible.data.geocoding import USER_AGENT

_PLANTS_BASE = "https://plantsservices.sc.egov.usda.gov"
_DEFAULT_RATE_LIMIT_S = 0.5


class UsdaPlantsClient:
    """
    Queries the USDA PLANTS Web Services for state-level species distribution.

    The symbol lookup endpoint resolves a scientific name to its USDA symbol
    (e.g. "Sambucus canadensis" → "SACA5"), then the state list endpoint
    confirms whether that symbol is present in a given state.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        rate_limit_s: float = _DEFAULT_RATE_LIMIT_S,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._rate_limit_s = rate_limit_s
        self._last_request_time: float = 0.0
        self._symbol_cache: dict[str, Optional[str]] = {}

    def get_symbol(self, scientific_name: str) -> Optional[str]:
        """
        Resolve a scientific name to its USDA PLANTS symbol.
        Returns None if the name is not found.
        """
        if scientific_name in self._symbol_cache:
            return self._symbol_cache[scientific_name]

        self._throttle()
        try:
            resp = self._session.get(
                f"{_PLANTS_BASE}/api/PlantList/TaxonNameSearch/{scientific_name}",
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json()
            symbol = results[0].get("Symbol") if results else None
        except (requests.RequestException, ConnectionError, IndexError, KeyError, ValueError):
            symbol = None

        self._symbol_cache[scientific_name] = symbol
        return symbol

    def is_present_in_state(self, scientific_name: str, state_code: str = "TX") -> Optional[bool]:
        """
        Return True if USDA PLANTS records the species in the given state.
        Returns None if the symbol cannot be resolved (treat as unknown, not absent).
        """
        symbol = self.get_symbol(scientific_name)
        if symbol is None:
            return None

        self._throttle()
        try:
            resp = self._session.get(
                f"{_PLANTS_BASE}/api/StateList/{symbol}",
                timeout=10,
            )
            resp.raise_for_status()
            states = resp.json()
            # Response is a list of state abbreviations e.g. ["TX", "OK", "LA"]
            return state_code.upper() in [s.upper() for s in states]
        except requests.RequestException:
            return None

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_s:
            time.sleep(self._rate_limit_s - elapsed)
        self._last_request_time = time.monotonic()
