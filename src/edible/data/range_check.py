"""
USDA PLANTS range checks — state-level and county-level.

State-level: UsdaPlantsClient queries the live USDA PLANTS Web Services API.
County-level: CountyRangeChecker loads a pre-built static JSON produced by
  scripts/build_county_range.py (which also uses the USDA API, offline).
  Static lookup keeps inference latency near-zero.

USDA PLANTS Web Services: https://plantsservices.sc.egov.usda.gov/
"""

from __future__ import annotations

import json
import time
from pathlib import Path
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

    def county_fips_in_state(
        self, scientific_name: str, state_fips_prefix: str = "48"
    ) -> Optional[list[str]]:
        """
        Return a list of county FIPS codes where the species is present,
        filtered to the given state FIPS prefix (default "48" = Texas).

        Returns None if the symbol cannot be resolved.
        Used by build_county_range.py to build the static county_range.json.
        """
        symbol = self.get_symbol(scientific_name)
        if symbol is None:
            return None

        self._throttle()
        try:
            resp = self._session.get(
                f"{_PLANTS_BASE}/api/CountyList/{symbol}",
                timeout=15,
            )
            resp.raise_for_status()
            all_fips: list[str] = resp.json()
            return [f for f in all_fips if str(f).startswith(state_fips_prefix)]
        except (requests.RequestException, ValueError):
            return None

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_s:
            time.sleep(self._rate_limit_s - elapsed)
        self._last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# County-level static range checker
# ---------------------------------------------------------------------------

class CountyRangeChecker:
    """
    Fast county-level presence lookup from a pre-built static JSON.

    The JSON has the form::

        {
          "species_id": {
            "TX": ["Travis", "Hays", "Williamson", ...]
          },
          ...
        }

    Built offline by ``scripts/build_county_range.py``.
    County names are title-cased and stripped of " County" suffix to match
    the output of ``NominatimGeocoder.reverse_geocode()``.
    """

    def __init__(self, range_path: Path) -> None:
        self._data: dict[str, dict[str, set[str]]] = {}
        if range_path.exists():
            raw = json.loads(range_path.read_text())
            for species_id, state_map in raw.items():
                self._data[species_id] = {
                    state: {c.lower() for c in counties}
                    for state, counties in state_map.items()
                }

    def is_in_county(
        self,
        species_id: str,
        county_name: str,
        state_code: Optional[str] = None,
    ) -> Optional[bool]:
        """
        Return True if the species is recorded in this county, False if not,
        or None if we have no county data for this species/state combination
        (caller should treat None as unknown and not penalise).

        Parameters
        ----------
        species_id:
            Internal species identifier, e.g. ``"ilex_vomitoria"``.
        county_name:
            County name as returned by Nominatim (may or may not have
            " County" suffix — both forms are handled).
        state_code:
            Two-letter state code (e.g. ``"TX"``).  If omitted, checks
            across all states.
        """
        if species_id not in self._data:
            return None

        state_map = self._data[species_id]
        states_to_check = (
            [state_code.upper()] if state_code else list(state_map.keys())
        )

        clean = county_name.lower().removesuffix(" county").strip()
        for state in states_to_check:
            if state not in state_map:
                return None  # no data for this state
            if clean in state_map[state]:
                return True

        return False

    def __len__(self) -> int:
        return len(self._data)
