"""
Adversarial tests for USDA PLANTS range check client and CountyRangeChecker.

Covers: symbol lookup, state presence check, county FIPS lookup, caching,
HTTP errors, unknown species, malformed responses, county checker edge cases.
"""

from __future__ import annotations

import json

import pytest
import responses as rsps_lib

from edible.data.range_check import CountyRangeChecker, _PLANTS_BASE, UsdaPlantsClient

SYMBOL_URL = f"{_PLANTS_BASE}/api/PlantList/TaxonNameSearch/Sambucus canadensis"
SYMBOL_URL_POKEWEED = f"{_PLANTS_BASE}/api/PlantList/TaxonNameSearch/Phytolacca americana"
STATELIST_URL = f"{_PLANTS_BASE}/api/StateList/SACA5"
STATELIST_POKEWEED_URL = f"{_PLANTS_BASE}/api/StateList/PHAME"


# ---------------------------------------------------------------------------
# Symbol lookup
# ---------------------------------------------------------------------------

class TestGetSymbol:
    @rsps_lib.activate
    def test_returns_symbol_for_known_species(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.get_symbol("Sambucus canadensis") == "SACA5"

    @rsps_lib.activate
    def test_returns_none_for_empty_results(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.get_symbol("Sambucus canadensis") is None

    @rsps_lib.activate
    def test_returns_none_on_http_error(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, status=500)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.get_symbol("Sambucus canadensis") is None

    @rsps_lib.activate
    def test_returns_none_on_connection_error(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, body=ConnectionError("refused"))
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.get_symbol("Sambucus canadensis") is None

    @rsps_lib.activate
    def test_caches_symbol_on_second_call(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        client.get_symbol("Sambucus canadensis")
        # Second call should NOT make another HTTP request (only one registered)
        result = client.get_symbol("Sambucus canadensis")
        assert result == "SACA5"
        assert len(rsps_lib.calls) == 1

    @rsps_lib.activate
    def test_caches_none_result(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        client.get_symbol("Sambucus canadensis")
        result = client.get_symbol("Sambucus canadensis")
        assert result is None
        assert len(rsps_lib.calls) == 1


# ---------------------------------------------------------------------------
# State presence check
# ---------------------------------------------------------------------------

class TestIsPresentInState:
    @rsps_lib.activate
    def test_returns_true_when_state_in_list(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        rsps_lib.add(rsps_lib.GET, STATELIST_URL, json=["TX", "OK", "LA", "AR"], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.is_present_in_state("Sambucus canadensis", "TX") is True

    @rsps_lib.activate
    def test_returns_false_when_state_not_in_list(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        rsps_lib.add(rsps_lib.GET, STATELIST_URL, json=["NY", "PA", "OH"], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.is_present_in_state("Sambucus canadensis", "TX") is False

    @rsps_lib.activate
    def test_returns_none_when_symbol_not_found(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.is_present_in_state("Sambucus canadensis", "TX") is None

    @rsps_lib.activate
    def test_returns_none_on_statelist_http_error(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        rsps_lib.add(rsps_lib.GET, STATELIST_URL, status=503)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.is_present_in_state("Sambucus canadensis", "TX") is None

    @rsps_lib.activate
    def test_state_comparison_case_insensitive(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        rsps_lib.add(rsps_lib.GET, STATELIST_URL, json=["tx", "ok"], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.is_present_in_state("Sambucus canadensis", "TX") is True

    @rsps_lib.activate
    def test_empty_state_list_returns_false(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        rsps_lib.add(rsps_lib.GET, STATELIST_URL, json=[], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.is_present_in_state("Sambucus canadensis", "TX") is False

    @rsps_lib.activate
    def test_toxic_species_checked_same_as_edible(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL_POKEWEED, json=[{"Symbol": "PHAME"}], status=200)
        rsps_lib.add(rsps_lib.GET, STATELIST_POKEWEED_URL, json=["TX", "OK"], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.is_present_in_state("Phytolacca americana", "TX") is True


# ---------------------------------------------------------------------------
# County FIPS lookup
# ---------------------------------------------------------------------------

COUNTYLIST_URL = f"{_PLANTS_BASE}/api/CountyList/SACA5"


class TestCountyFipsInState:
    @rsps_lib.activate
    def test_returns_texas_fips_only(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        rsps_lib.add(
            rsps_lib.GET, COUNTYLIST_URL,
            json=["48453", "48491", "12086"],  # Travis, Williamson (TX), Miami-Dade (FL)
            status=200,
        )
        client = UsdaPlantsClient(rate_limit_s=0)
        result = client.county_fips_in_state("Sambucus canadensis", state_fips_prefix="48")
        assert result is not None
        assert "48453" in result
        assert "48491" in result
        assert "12086" not in result  # Florida county excluded

    @rsps_lib.activate
    def test_returns_none_when_symbol_not_found(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.county_fips_in_state("Sambucus canadensis") is None

    @rsps_lib.activate
    def test_returns_none_on_http_error(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        rsps_lib.add(rsps_lib.GET, COUNTYLIST_URL, status=500)
        client = UsdaPlantsClient(rate_limit_s=0)
        assert client.county_fips_in_state("Sambucus canadensis") is None

    @rsps_lib.activate
    def test_empty_list_returns_empty(self):
        rsps_lib.add(rsps_lib.GET, SYMBOL_URL, json=[{"Symbol": "SACA5"}], status=200)
        rsps_lib.add(rsps_lib.GET, COUNTYLIST_URL, json=[], status=200)
        client = UsdaPlantsClient(rate_limit_s=0)
        result = client.county_fips_in_state("Sambucus canadensis")
        assert result == []


# ---------------------------------------------------------------------------
# CountyRangeChecker
# ---------------------------------------------------------------------------

def _make_range_file(tmp_path, data: dict) -> "Path":
    p = tmp_path / "county_range.json"
    p.write_text(json.dumps(data))
    return p


class TestCountyRangeCheckerLoad:
    def test_loads_empty_when_file_absent(self, tmp_path):
        checker = CountyRangeChecker(tmp_path / "nonexistent.json")
        assert len(checker) == 0

    def test_loads_species_from_file(self, tmp_path):
        p = _make_range_file(tmp_path, {"sambucus_canadensis": {"TX": ["Travis", "Hays"]}})
        checker = CountyRangeChecker(p)
        assert len(checker) == 1

    def test_len_returns_species_count(self, tmp_path):
        p = _make_range_file(tmp_path, {
            "sambucus_canadensis": {"TX": ["Travis"]},
            "ilex_vomitoria": {"TX": ["Travis", "Harris"]},
        })
        assert len(CountyRangeChecker(p)) == 2


class TestCountyRangeCheckerIsInCounty:
    def _checker(self, tmp_path) -> CountyRangeChecker:
        p = _make_range_file(tmp_path, {
            "sambucus_canadensis": {"TX": ["Travis", "Hays", "Williamson"]},
            "menispermum_canadense": {"TX": ["Anderson", "Cherokee"]},
        })
        return CountyRangeChecker(p)

    def test_present_county_returns_true(self, tmp_path):
        c = self._checker(tmp_path)
        assert c.is_in_county("sambucus_canadensis", "Travis", "TX") is True

    def test_absent_county_returns_false(self, tmp_path):
        c = self._checker(tmp_path)
        assert c.is_in_county("sambucus_canadensis", "El Paso", "TX") is False

    def test_unknown_species_returns_none(self, tmp_path):
        c = self._checker(tmp_path)
        assert c.is_in_county("unknown_species", "Travis", "TX") is None

    def test_unknown_state_returns_none(self, tmp_path):
        c = self._checker(tmp_path)
        # sambucus_canadensis only has TX data — asking for CA returns None
        assert c.is_in_county("sambucus_canadensis", "Sacramento", "CA") is None

    def test_county_name_case_insensitive(self, tmp_path):
        c = self._checker(tmp_path)
        assert c.is_in_county("sambucus_canadensis", "travis", "TX") is True
        assert c.is_in_county("sambucus_canadensis", "TRAVIS", "TX") is True

    def test_county_name_strips_county_suffix(self, tmp_path):
        c = self._checker(tmp_path)
        assert c.is_in_county("sambucus_canadensis", "Travis County", "TX") is True

    def test_no_state_code_checks_all_states(self, tmp_path):
        c = self._checker(tmp_path)
        assert c.is_in_county("sambucus_canadensis", "Travis") is True

    def test_false_not_confused_with_none(self, tmp_path):
        c = self._checker(tmp_path)
        result = c.is_in_county("sambucus_canadensis", "Loving", "TX")
        assert result is False
        assert result is not None
