# Task Log: MERRA-2
_Last updated: 2026-08-21_

## Scope
Extract MERRA-2 hourly aerosol optical thickness (TOTEXTTAU) as the calibrated gap-fill source for MAIAC AOD gaps, for the Delhi Phase 1 study window (2025-03-01 to 2026-02-28), matched to the 42 finalized CPCB stations.

## Completed

**Setup & validation** (2026-08-20)
- Confirmed GEE collection: `NASA/GSFC/MERRA/aer/2` (M2T1NXAER), band `TOTEXTTAU`, ~62km native resolution (0.5° x 0.625°).
- Sanity-checked in GEE Code Editor: 24 hourly images/day confirmed; TOTEXTTAU = 1.39 on a June test date (pre-monsoon dust season, plausible); TOTEXTTAU = 0.23 on a monsoon date (2025-07-15), consistent with wet scavenging.
- Saved the sanity-check script as `scripts/datasets/merra2_aod/gee_console_checks.js` (reference only, not a DVC stage).

**Station-to-cell mapping** (2026-08-20)
- Wrote and ran `scripts/datasets/merra2_aod/1-map_stations_to_cells.py`.
- All 42 finalized KEEP stations mapped to their true MERRA-2 grid cell center using `pixelLonLat().reproject(merra_projection)` — not `filterBounds` (no swath-geometry issue here, unlike MAIAC) and not manual grid math (avoids grid-origin misalignment risk).
- Result: 3 unique cells. Cell 1 (28.75, 77.3125): 38 stations. Cell 2 (28.25, 77.3125): 3 stations (NISE Gwal Pahari, Aya Nagar, Karni Singh Shooting Range — southern edge). Cell 3 (28.75, 76.6875): 1 station (Najafgarh — westernmost).
- Output: `data/raw/merra2_aod/station_cell_mapping.csv`, DVC stage `map_stations_to_cells`.

**Raw hourly extraction** (2026-08-20)
- Wrote and ran `scripts/datasets/merra2_aod/2-extract_gee_merra2.py` — one `.getRegion()` call per unique cell (3 calls total, not 42) pulling the full hourly time series for the study window.
- Got 8,616 hourly rows per cell (8,760 expected for a full 365-day year) — 144 hours short, identical across all 3 cells.
- Investigated: 6 fully-missing days, all September 2025 (09-04, 09-05, 09-20, 09-23, 09-26, 09-28), 24/24 hours missing each, identical across cells — a real MERRA-2 collection gap, not a script bug. No action needed; these days simply have zero rows going into the windowing step.
- Output: `data/raw/merra2_aod/merra2_aod_raw.csv`, DVC stage `extract_merra2`.

**Windowing, QC & daily aggregation** (2026-08-21)
- Added a coordinate-consistency check to `3-window_and_qc_merra2.py`: compares GEE's actually-sampled point (`longitude`/`latitude` in the raw file) against the intended cell center (`cell_lat`/`cell_lon` in the mapping file), per `cell_id`. Runs first, before windowing/duplicates/range filtering — designed as a hard-fail correctness gate, not a data-quality filter, since a mismatch implicates a whole cell's data, not just a few rows. Required adding a `--mapping` argument to script 3.
- Corrected DVC output folder structure to match the `raw/interim/processed` convention: `station_cell_mapping.csv` stays in `data/raw/merra2_aod/` (dependency of the raw extraction step, not itself a derived output); `merra2_aod_windowed_qc.csv` → `data/interim/merra2_aod/`; `merra2_aod_daily.csv` → `data/processed/merra2_aod/`.
- First `dvc repro` after adding the check failed: all 25,848 raw rows flagged as coordinate mismatches (~90–310m offsets, consistent within each cell). Diagnosed as `getRegion(scale=1000)` snapping the reported longitude/latitude to a 1km reprojection grid before echoing them back — not a real bug, and not affecting the actual `TOTEXTTAU` values (negligible relative to the ~62km native cell size).
- Fixed by loosening `coord_tolerance_degrees` from `1e-6` to `0.05` (~5.5km) — comfortably above the observed jitter, comfortably below the ~62km cell spacing, so it still catches a genuine wrong-cell bug. Params-only fix, no code change or re-extraction needed.
- Reran `dvc repro` — `window_qc_merra2` and `aggregate_merra2` both completed successfully (confirmed via `git status`: `dvc.lock` updated, `data/interim/merra2_aod/` and `data/processed/merra2_aod/` created).
- Reaffirmed `totexttau_max: 5.0` as a "catch broken data" backstop rather than a literal physical AOD ceiling — MERRA-2's ~62km spatial averaging dilutes even genuine extreme events well below that bound, so it shouldn't clip real data while still catching fill values or unit-scale bugs.
- Output: `data/processed/merra2_aod/merra2_aod_daily.csv` — station-level daily TOTEXTTAU for all 42 stations, DVC stages `window_qc_merra2` and `aggregate_merra2`.
- Tagged `delhi-phase1-v7`: MERRA-2 pipeline complete end-to-end (map → raw extract → window/QC → daily aggregation).

## Key decisions
- 2026-08-20: Group stations by shared MERRA-2 grid cell instead of extracting per station — Delhi's full station spread (~40km) is smaller than one MERRA-2 cell (~62km), so 42 point extractions would have been redundant. Cut GEE calls from 42 to 3.
- 2026-08-20: Determine cell membership via GEE's own `pixelLonLat()`, reprojected to MERRA-2's native projection — not manual lat/lon rounding against the documented grid spacing, since getting the grid origin wrong would silently misassign stations with no error thrown.
- 2026-08-20: Split MERRA-2 processing into 4 separate numbered scripts (map cells → raw extract → window+QC → aggregate) rather than one combined script, mirroring the CPCB pipeline's separation of QC from aggregation. Keeps the expensive/slow GEE call isolated from the cheap, easily-rerun-on-tweak pandas logic.
- 2026-08-20: Overpass window for MERRA-2 set to UTC hours 5, 6, 7 (05:00–08:00 UTC = 10:30–13:30 IST), matching the window already used for CPCB and MAIAC, and matching the Maheshwarkar & Sunder Raman (2021) precedent.
- 2026-08-20: Daily aggregation requires at least 2 of 3 overpass hours present (`min_hours_required=2`), mirroring CPCB's own "min 2 overpass hours/day" completeness rule.
- 2026-08-20: TOTEXTTAU sanity range set loosely to [0, 5] — meant to catch fill values/unit bugs, not real dust-storm spikes, so genuine extreme readings shouldn't be removed.
- 2026-08-20: All script configuration (GEE project ID, collection, band, dates, thresholds) now loads from `params.yaml` per-script, not hardcoded — standing rule going forward for all pipeline scripts, with a matching `dvc.yaml` stage provided alongside each one.
- 2026-08-21: Coordinate-consistency check treats any mismatch as a hard failure, not a warning or row-drop — a mismatch implicates the whole cell's data, not just individual rows.
- 2026-08-21: `station_cell_mapping.csv` kept in `data/raw/merra2_aod/` rather than moved to `interim` — it's a direct dependency of the raw extraction step (used to choose query points), not itself a processed/derived output.
- 2026-08-21: `coord_tolerance_degrees` set to 0.05° rather than a near-zero value — the check exists to catch genuine wrong-cell errors, not to flag expected sub-kilometer reprojection jitter from GEE's `getRegion(scale=...)` behavior.

## Data notes & gotchas
- Caught before it reached the real script: `ee.Image.pixelLonLat().reduceRegion(scale=1000)` without reprojecting first does NOT return the true MERRA-2 cell center — it silently returns something very close to the queried point's own coordinates instead, because `pixelLonLat()` has no fixed native resolution until reprojected onto the target grid. Fixed by calling `.reproject(merra_projection)` before reducing. Real TOTEXTTAU value sampling (not `pixelLonLat`) at scale=1000 was unaffected, since the actual MERRA-2 image already carries its correct native projection.
- pandas version difference: `pd.factorize()` on a plain `list(zip(...))` fails on newer pandas ("factorize requires a Series..."). Replaced with `drop_duplicates()` + `reset_index()` + `merge()` for the `cell_id` assignment instead — also cleaner, plain-pandas style.
- MERRA-2 image IDs are timestamped `YYYYMMDDHH` in UTC (e.g. `NASA/GSFC/MERRA/aer/2/2025060100`) — the HH segment maps directly to the overpass-window hours (05, 06, 07) needed for filtering.
- 6 fully-missing days in Sept 2025 across all cells (real MERRA-2 gap, listed above under Completed) — surfaced as skipped days in the windowing/aggregation step; no fix needed, was expected.
- `getRegion(geometry, scale)` does not echo back the exact query point's coordinates — at `scale=1000` it reprojects to a 1km grid first and reports that pixel's center, producing up to ~1km of drift in the returned `longitude`/`latitude` columns (but not in the sampled band value itself, which still reflects the correct native ~62km cell). Same underlying category of issue as the `pixelLonLat()` scale bug above, but shows up differently since `getRegion` samples a real image with its own native projection rather than a synthetic coordinate image.

## Pending
- Sanity-check `merra2_aod_daily.csv` output ranges and station coverage (row counts per station, TOTEXTTAU distribution, dropped-day counts) — not yet explicitly reviewed, only confirmed the pipeline ran without error.
- Feed into the MAIAC↔MERRA-2 calibration regression per airshed (`MAIAC ≈ a + b · MERRA2`) — next real milestone for this data, blocked on the temporal-alignment/gap-fill-ordering decision currently open in the AOD preprocessing task log.

## Ideas / under consideration
- None raised yet for this module.