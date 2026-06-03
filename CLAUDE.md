# Edible — Claude Code Project Context

**AI-powered wild berry & foraging identification app (educational tool)**

## Core Safety Constraint
This is a food-safety application. A false positive (calling toxic → edible) is a critical failure.
- Toxic species recall is the #1 model metric — never trade it for accuracy
- Every result screen must show confidence score + look-alike warnings
- Below 75% confidence: force "do not eat" message

## Branch Strategy
- All work in branches off `main` unless explicitly told otherwise
- Branch naming: `phase-N/description` or `feature/description`
- Commit early, commit often

## Test Requirements
- 90% coverage minimum — enforced via `pytest --cov=src/edible --cov-fail-under=90`
- Testing is adversarial: edge cases, boundary conditions, malformed inputs, toxic misclassification paths
- Run: `uv run pytest --cov=src/edible --cov-report=term-missing`

## Dynamic Workflows
We use Claude Code's agentic capabilities (sub-agents) for:
- Parallel species scraping (12 species concurrently)
- Parallel training experiments
- Test generation against edge cases

## Stack
| Layer | Technology |
|---|---|
| Language | Python 3.10 |
| Package manager | uv |
| ML | PyTorch + timm (EfficientNet-B0 / MobileNetV3) |
| Training | Google Colab (free GPU) |
| Data pipeline | Python + iNaturalist API |
| Label tooling | Label Studio (self-hosted) |
| Backend API | FastAPI |
| Frontend | React (web first), React Native later |
| Model hosting | Hugging Face (free tier) |
| Range data | USDA PLANTS + BONAP (county-level) |
| Geocoding | OpenStreetMap Nominatim |

## Project Structure
```
edible/
├── src/edible/          # core library
│   ├── data/            # scraping, metadata, pipeline
│   ├── model/           # training, inference, evaluation
│   └── api/             # FastAPI app
├── tests/               # mirrors src/edible/ structure
├── data/
│   ├── species.json     # species reference list (tracked)
│   ├── lookalikes.json  # look-alike pairs (tracked)
│   └── images/          # scraped images (gitignored, local only)
├── notebooks/           # Colab-ready training notebooks
└── scripts/             # one-off utilities
```

## Environment
- iNaturalist credentials: `INAT_ACCT` and `INAT_PW` in `~/.bashrc`
- Images stored locally at `data/images/` (gitignored)
- **IMPORTANT**: System `PYTHONPATH` is polluted with Python 3.9 paths. Always run:
  `unset PYTHONPATH && uv run pytest` — or just use `make test`

## Geographic Scope — Phase 1
- Texas statewide; Central Texas data density expected from iNaturalist
- iNaturalist Texas place ID: **10` (to verify)`**

## Species — v1 (12 species)
6 edible, 6 toxic — see `data/species.json`

## Edibility Taxonomy
`edible_raw` | `edible_cooked` | `toxic` | `uncertain`

## Key Risk: False Positives on Toxic Species
Weight loss functions and evaluation to penalize toxic→edible misclassification heavily.
Never suppress out-of-range predictions — only down-rank them.

## Label Tooling: Label Studio
Self-hosted, no data leaves machine, no tier limits.
Run: `uv run label-studio` (after install)
