"""
Scrape iNaturalist observations for all (or specific) target species.

Usage:
    # Scrape all 12 species, 1000 images each (Texas only, default)
    uv run python scripts/scrape.py

    # Scrape a single species globally (useful for Texas-rare species)
    uv run python scripts/scrape.py --species menispermum_canadense --global

    # Scrape only fruiting observations (recommended for training data quality)
    uv run python scripts/scrape.py --fruiting-only

    # Combine: global + fruiting only + blur filter
    uv run python scripts/scrape.py --global --fruiting-only

    # Adjust blur rejection threshold (default 100; lower = more permissive)
    uv run python scripts/scrape.py --blur-threshold 50

    # Disable blur filtering entirely
    uv run python scripts/scrape.py --blur-threshold 0

    # Dry run — show what would be scraped without downloading
    uv run python scripts/scrape.py --dry-run

Output:
    data/images/{species_id}/{observation_id}_{photo_id}.jpg
    data/images/{species_id}/{observation_id}_{photo_id}.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edible.data.geocoding import NominatimGeocoder
from edible.data.image_quality import DEFAULT_BLUR_THRESHOLD
from edible.data.range_check import UsdaPlantsClient
from edible.data.scraper import TEXAS_PLACE_ID, InatClient, ScrapeSummary, SpeciesScraper
from edible.data.schemas import load_species_db

DATA_DIR = Path(__file__).parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
SPECIES_FILE = DATA_DIR / "species.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape iNaturalist images for Edible species")
    parser.add_argument(
        "--species",
        help="Species ID to scrape (e.g. sambucus_canadensis). Omit to scrape all.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Max images per species (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be scraped without downloading anything",
    )
    parser.add_argument(
        "--fruiting-only",
        action="store_true",
        help=(
            "Only download observations annotated 'Fruits or Seeds' on iNaturalist. "
            "Eliminates stem/leaf-only images that cause look-alike confusion."
        ),
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=DEFAULT_BLUR_THRESHOLD,
        help=(
            f"Laplacian variance floor for blur rejection (default: {DEFAULT_BLUR_THRESHOLD}). "
            "Set to 0 to disable."
        ),
    )
    place_group = parser.add_mutually_exclusive_group()
    place_group.add_argument(
        "--global",
        dest="global_scrape",
        action="store_true",
        help="Scrape globally (no place filter). Useful for Texas-rare species like moonseed.",
    )
    place_group.add_argument(
        "--place-id",
        type=int,
        default=None,
        help=f"iNaturalist place ID to filter by (default: {TEXAS_PLACE_ID} = Texas).",
    )
    args = parser.parse_args()

    if not os.environ.get("INAT_API_TOKEN"):
        logger.warning(
            "INAT_API_TOKEN not set — running unauthenticated (rate-limited to ~60 req/min). "
            "Get your token at https://www.inaturalist.org/users/api_token"
        )

    if args.global_scrape:
        place_id = None
        logger.info("Place filter: GLOBAL (no restriction)")
    elif args.place_id is not None:
        place_id = args.place_id
        logger.info("Place filter: place_id=%d", place_id)
    else:
        place_id = TEXAS_PLACE_ID
        logger.info("Place filter: Texas (place_id=%d)", place_id)

    if args.fruiting_only:
        logger.info("Phenology filter: Fruits or Seeds only")
    if args.blur_threshold > 0:
        logger.info("Blur threshold: %.1f (Laplacian variance)", args.blur_threshold)
    else:
        logger.info("Blur filtering: disabled")

    db = load_species_db(SPECIES_FILE)

    if args.species:
        species_list = [db.get_by_id(args.species)]
        if species_list[0] is None:
            valid = [s.id for s in db.species]
            logger.error("Unknown species ID '%s'. Valid IDs:\n  %s", args.species, "\n  ".join(valid))
            sys.exit(1)
    else:
        species_list = db.species

    logger.info(
        "Scraping %d species × up to %d images each → %s",
        len(species_list),
        args.max_images,
        IMAGES_DIR,
    )

    if args.dry_run:
        logger.info("DRY RUN — no downloads will occur")
        for s in species_list:
            existing = list((IMAGES_DIR / s.id).glob("*.jpg")) if (IMAGES_DIR / s.id).exists() else []
            logger.info(
                "  %-35s  edibility=%-15s  existing=%d",
                s.scientific_name, s.edibility.value, len(existing),
            )
        return

    client = InatClient.from_env()
    geocoder = NominatimGeocoder()
    range_client = UsdaPlantsClient()
    scraper = SpeciesScraper(client, geocoder, range_client, IMAGES_DIR)

    summaries: list[ScrapeSummary] = []
    for species in species_list:
        logger.info("── Scraping: %s (%s)", species.common_name, species.scientific_name)
        summary = scraper.scrape(
            species,
            max_images=args.max_images,
            place_id=place_id,
            fruiting_only=args.fruiting_only,
            blur_threshold=args.blur_threshold,
        )
        summaries.append(summary)
        logger.info(
            "   downloaded=%d  blurry=%d  not_fruiting=%d  skipped=%d  failed=%d",
            summary.downloaded,
            summary.skipped_blurry,
            summary.skipped_not_fruiting,
            summary.skipped_existing,
            summary.failed,
        )
        if summary.errors:
            for err in summary.errors[:5]:
                logger.warning("   error: %s", err)

    # Final summary table
    print("\n" + "─" * 80)
    print(f"{'Species':<35} {'DL':>6} {'Blurry':>7} {'!Fruit':>7} {'Skip':>6} {'Fail':>6}")
    print("─" * 80)
    for s in summaries:
        print(
            f"{s.species_id:<35} {s.downloaded:>6} {s.skipped_blurry:>7} "
            f"{s.skipped_not_fruiting:>7} {s.skipped_existing:>6} {s.failed:>6}"
        )
    print("─" * 80)
    total_dl = sum(s.downloaded for s in summaries)
    total_blur = sum(s.skipped_blurry for s in summaries)
    total_fail = sum(s.failed for s in summaries)
    print(f"{'TOTAL':<35} {total_dl:>6} {total_blur:>7} {'':>7} {'':>6} {total_fail:>6}")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
