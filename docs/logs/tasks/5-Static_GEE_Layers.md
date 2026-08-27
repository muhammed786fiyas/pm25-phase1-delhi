# Task Log: Static GEE Layers
_Last updated: 2026-08-27_

## Scope
Static (non-time-varying-in-source, or time-varying-but-not-overpass-matched) covariate extraction for Delhi Phase 1: ESA WorldCover land-use fractions, Sentinel-2 NDVI, SRTM elevation/slope. Follows the ERA5-Land meteorology module in the "on the horizon" sequence.

## Completed

**WorldCover (land-use % fractions)** — 2026-08-27
- Script 1: raw pixel-count extraction per station (1km buffer, 10m scale), all 42 KEEP stations, includes per-run extraction summary (min/max/median pixel counts, failed stations)
- Script 2: QC gate — missing station, coordinate mismatch, zero pixels, expected-pixel-count deviation (median-based), all-one-class warning, implausible-class warning (snow/mangrove/moss in Delhi)
- Script 3: converts raw histogram to % fractions, writes to processed
- Result: 42/42 stations, 0 hard fails, 0 warnings. Dominant class built-up (avg ~60%), tree cover ~38% at R K Puram test station.
- Tagged `delhi-phase1-v13`

**NDVI (Sentinel-2, 5-day composites)** — 2026-08-27
- Script 4: batch extraction via `reduceRegions` across 73 fixed 5-day periods spanning the study window (73 GEE calls total, not 15,330)
- Script 5: QC — missing stations, row-count sanity, overall/by-month/by-station null rates, implausible NDVI range
- Script 6: fill-distance analysis (nearest valid period per null, no fill applied yet)
- Script 7: gap-fill using cap=5 periods, backward-preferred on ties
- Result: 3066 station-period rows, 699 nulls (22.8%), all 699 filled (0 skipped, 0 unfillable), cap matched the true observed max distance exactly
- Tagged `delhi-phase1-v14`

**SRTM (elevation + slope)** — 2026-08-27
- Script 8: raw extraction, mean elevation + slope (via `ee.Terrain.slope()`) across 1km buffer, 30m scale
- Script 9: QC — missing station, elevation range [150,300]m, slope range [0,15]°
- Script 10: finalize (QC-passed raw → processed, straight copy)
- Result: 42/42 stations, 0 hard fails. Elevation 199–272m, slope 2.7–4.5°.
- Tagged `delhi-phase1-v15`

## Key decisions

- **Buffer radius: 1km for all three static covariates** (2026-08-27) — matches MAIAC AOD's native 1km grid so every static layer shares the same spatial footprint as the primary predictor; no literature precedent found for a different scale in this project's reading list, and mismatched buffer scales would need separate justification later.
- **WorldCover v200 (2021), not v100 (2020)** (2026-08-27) — v200 uses an improved algorithm. Confirmed via web search that no 2022+ version exists as of Aug 2026 — v200 remains the latest.
- **NDVI: nearest 5-day composite, not seasonal median** (2026-08-27) — Delhi Phase 1 is the best-case validation run for the whole modeling approach; collapsing NDVI to 4 values/year would discard temporal signal (dust/dry vegetation pre-monsoon vs. post-monsoon greening) that the daily AOD-PM2.5 relationship could benefit from.
- **NDVI extraction restructured to fixed 5-day periods + batch `reduceRegions`, not per-station-day loop** (2026-08-27) — naive per-station-day looping would require ~15,330 individual GEE calls (42 stations × 365 days); fixed periods + batch extraction reduces this to 73 calls total (one per period, each covering all 42 stations at once). "5-day composite" interpreted as periodic bucket (MODIS-style), not sliding nearest-in-time window.
- **NDVI gap-fill cap = 5 periods (25 days)** (2026-08-27) — set after running the fill-distance analysis script rather than guessed upfront; 5 periods is the actual observed max distance in the data (699/699 nulls fillable, 0 unfillable, 0 over-cap). Hard-fail gate pattern: cap lives in params.yaml, script refuses to run if unset.
- **params.yaml nested per-stage (`static_gee_layers.<subdataset>.<stage>`), not one flat section** (2026-08-27) — matches the `era5_blh` structure; prevents DVC from marking unrelated stages dirty when only one stage's params change.
- **Output paths moved from params.yaml to CLI args** (2026-08-27) — params.yaml holds only tunable config; dvc.yaml owns the actual file paths, per project convention.
- **Single `scripts/datasets/static_gee/` folder for all 3 sub-datasets, not split by sub-dataset** (2026-08-27) — matches MAIAC's precedent of one folder per module even across multiple pipeline stages.
- **SRTM kept as 3 scripts (extract/QC/finalize)**, not collapsed to 2, despite finalize being a near-no-op copy (2026-08-27) — chosen for structural consistency with WorldCover and NDVI over minimizing script count.

## Data notes & gotchas

- **WorldCover pixel counts run ~35,440 for a 1km buffer at 10m scale, not the naive area/pixel-area estimate of ~31,416.** `reduceRegion`'s reprojection at Delhi's latitude inflates the apparent count ~12–13% above flat theoretical math. QC's expected-pixel check had to be built from the run's own median, not a hardcoded formula — a hardcoded 31,416 would have failed every single station despite correct extraction.
- **NDVI nulls (22.8% overall) cluster by month, not by station.** July 76% null, August 67% null (monsoon cloud cover — same mechanism as MAIAC's AOD gaps). January 48% null — winter fog/haze gets flagged by the QA60 cloud mask too. Station-level null rates are evenly spread 16–29% with no outlier station, confirming the pattern is temporal/seasonal, not spatial/coordinate-related.
- **January NDVI gap coincides with peak PM2.5 season** — worth flagging as a real limitation for the eventual manuscript, not just a pipeline quirk: NDVI coverage is worst exactly when the AOD-PM2.5 relationship is most active and informative.
- **Negative NDVI values (min observed: -0.26) are expected, not errors** — indicate water bodies (e.g. near the Yamuna) inside a station's 1km buffer.
- **pandas `to_csv` does not auto-create missing parent directories.** Script 1's first run crashed on this for the new `data/raw/static_gee/worldcover/` path. Fixed with `os.makedirs(..., exist_ok=True)` before every `to_csv` call — apply this pattern to any new script writing into a not-yet-existing folder.

## Pending
- OSM features module (industrial land-use fraction, distance to nearest power plant; stretch: bus stop density, fuel station density) — separate tooling (Overpass API/Geofabrik), not GEE-based
- Zaid & Sahu (2025) airshed boundaries — need supplementary shapefile or digitize from published figure

## Ideas / under consideration
- Collapsing SRTM's QC + finalize into a single script — raised, rejected in favor of keeping 3 scripts for consistency across all static_gee sub-datasets.