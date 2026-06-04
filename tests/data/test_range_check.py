"""
Adversarial tests for USDA PLANTS range check client.

Covers: symbol lookup, state presence check, caching, HTTP errors,
unknown species, malformed responses.
"""

from __future__ import annotations

import responses as rsps_lib

from edible.data.range_check import _PLANTS_BASE, UsdaPlantsClient

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
