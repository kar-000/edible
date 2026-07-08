"""
Build data/county_range.json — static county-level range lookup for GPS re-ranking.

Queries the USDA PLANTS Web Services for each species in species.json, fetches
county distribution via the plantsservices API, and writes the result.

Output format::

    {
      "ilex_vomitoria": {
        "TX": ["Anderson", "Angelina", "Aransas", ...]
      },
      ...
    }

County names are title-cased and stripped of any " County" suffix to match
the output of NominatimGeocoder.reverse_geocode().

Usage
-----
    uv run python scripts/build_county_range.py
    uv run python scripts/build_county_range.py --out data/county_range.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edible.data.schemas import load_species_db

DATA_DIR = Path(__file__).parent.parent / "data"

_PLANTS_API = "https://plantsservices.sc.egov.usda.gov/api"
_TX_STATE_FIP = "48"


def _get_master_id(session: requests.Session, scientific_name: str) -> int | None:
    """Return the accepted plant master ID for a scientific name."""
    resp = session.get(
        f"{_PLANTS_API}/PlantSearch",
        params={"searchText": scientific_name},
        timeout=20,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None

    # Use first result (most relevant match)
    plant_id: int = results[0]["Plant"]["Id"]

    # Resolve accepted synonym if needed
    profile_resp = session.get(f"{_PLANTS_API}/PlantProfile/{plant_id}", timeout=20)
    profile_resp.raise_for_status()
    profile = profile_resp.json()
    accepted_id: int = profile.get("AcceptedId", 0)
    return accepted_id if accepted_id else plant_id


def _get_tx_counties(session: requests.Session, master_id: int) -> list[str]:
    """Return sorted list of Texas county names for the given plant master ID."""
    resp = session.post(
        f"{_PLANTS_API}/PlantProfile/getDownloadDistributionDocumentation",
        json={"masterId": master_id, "offset": 0},
        timeout=30,
    )
    resp.raise_for_status()

    # Response is CSV with a leading "Distribution Data" title line:
    # Distribution Data
    # Symbol,Country,State,State FIP,County,County FIP
    # ...
    text = resp.text
    # Skip the "Distribution Data" title row; DictReader uses the next line as header
    lines = text.splitlines()
    csv_start = next((i for i, l in enumerate(lines) if l.startswith("Symbol,")), 1)
    reader = csv.DictReader(io.StringIO("\n".join(lines[csv_start:])))
    counties: list[str] = []
    for row in reader:
        state_fip = row.get("State FIP", "").strip()
        county = row.get("County", "").strip()
        if state_fip == _TX_STATE_FIP and county:
            # Strip " County" suffix to match NominatimGeocoder output
            county = county.removesuffix(" County").strip()
            counties.append(county)
    return sorted(set(counties))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build county_range.json from USDA PLANTS API"
    )
    parser.add_argument("--out", type=Path, default=DATA_DIR / "county_range.json")
    parser.add_argument("--species-db", type=Path, default=DATA_DIR / "species.json")
    parser.add_argument("--rate-limit", type=float, default=0.5,
                        help="Seconds to sleep between API calls")
    args = parser.parse_args()

    species_db = load_species_db(args.species_db)
    session = requests.Session()
    session.headers["User-Agent"] = "edible-foraging-app/1.0 research"

    output: dict[str, dict[str, list[str]]] = {}
    failed: list[str] = []

    for species in sorted(species_db.species, key=lambda s: s.id):
        sid = species.id
        sci = species.scientific_name
        print(f"  {sid:<35} ({sci}) ...", end=" ", flush=True)

        try:
            master_id = _get_master_id(session, sci)
            if master_id is None:
                print("FAILED (not found)")
                failed.append(sid)
                continue

            time.sleep(args.rate_limit)
            tx_counties = _get_tx_counties(session, master_id)
            output[sid] = {"TX": tx_counties}
            print(f"{len(tx_counties)} TX counties (masterId={master_id})")

        except Exception as exc:
            print(f"FAILED ({exc})")
            failed.append(sid)

        time.sleep(args.rate_limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {args.out}  ({len(output)} species)")

    if failed:
        print(f"\nFailed to resolve: {', '.join(failed)}")
        print("These species will have no county data (GPS re-ranking skipped for them).")


if __name__ == "__main__":
    main()
