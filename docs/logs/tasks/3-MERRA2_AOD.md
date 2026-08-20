# Task Log: MERRA-2
_Last updated: 2026-08-20_

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

## Key decisions
- 2026-08-20: Group stations by shared MERRA-2 grid cell instead of extracting per station — Delhi's full station spread (~40km) is smaller than one MERRA-2 cell (~62km), so 42 point extractions would have been redundant. Cut GEE calls from 42 to 3.
- 2026-08-20: Determine cell membership via GEE's own `pixelLonLat()`, reprojected to MERRA-2's native projection — not manual lat/lon rounding against the documented grid spacing, since getting the grid origin wrong would silently misassign stations with no error thrown.
- 2026-08-20: Split MERRA-2 processing into 4 separate numbered scripts (map cells → raw extract → window+QC → aggregate) rather than one combined script, mirroring the CPCB pipeline's separation of QC from aggregation. Keeps the expensive/slow GEE call isolated from the cheap, easily-rerun-on-tweak pandas logic.
- 2026-08-20: Overpass window for MERRA-2 set to UTC hours 5, 6, 7 (05:00–08:00 UTC = 10:30–13:30 IST), matching the window already used for CPCB and MAIAC, and matching the Maheshwarkar & Sunder Raman (2021) precedent.
- 2026-08-20: Daily aggregation requires at least 2 of 3 overpass hours present (`min_hours_required=2`), mirroring CPCB's own "min 2 overpass hours/day" completeness rule.
- 2026-08-20: TOTEXTTAU sanity range set loosely to [0, 5] — meant to catch fill values/unit bugs, not real dust-storm spikes, so genuine extreme readings shouldn't be removed.
- 2026-08-20: All script configuration (GEE project ID, collection, band, dates, thresholds) now loads from `params.yaml` per-script, not hardcoded — standing rule going forward for all pipeline scripts, with a matching `dvc.yaml` stage provided alongside each one.

## Data notes & gotchas
- Caught before it reached the real script: `ee.Image.pixelLonLat().reduceRegion(scale=1000)` without reprojecting first does NOT return the true MERRA-2 cell center — it silently returns something very close to the queried point's own coordinates instead, because `pixelLonLat()` has no fixed native resolution until reprojected onto the target grid. Fixed by calling `.reproject(merra_projection)` before reducing. Real TOTEXTTAU value sampling (not `pixelLonLat`) at scale=1000 was unaffected, since the actual MERRA-2 image already carries its correct native projection.
- pandas version difference: `pd.factorize()` on a plain `list(zip(...))` fails on newer pandas ("factorize requires a Series..."). Replaced with `drop_duplicates()` + `reset_index()` + `merge()` for the `cell_id` assignment instead — also cleaner, plain-pandas style.
- MERRA-2 image IDs are timestamped `YYYYMMDDHH` in UTC (e.g. `NASA/GSFC/MERRA/aer/2/2025060100`) — the HH segment maps directly to the overpass-window hours (05, 06, 07) needed for filtering.
- 6 fully-missing days in Sept 2025 across all cells (real MERRA-2 gap, listed above under Completed) — will surface as skipped days in the windowing/aggregation step; no fix needed, just expected.

## Pending
- Run `scripts/datasets/merra2_aod/3-window_and_qc_merra2.py` (overpass window filter, duplicate check, TOTEXTTAU range check) — written, not yet executed.
- Run `scripts/datasets/merra2_aod/4-aggregate_merra2_daily.py` (daily aggregation + broadcast to all 42 stations) — written, not yet executed.
- After both run: sanity-check `merra2_aod_daily.csv` output ranges and station coverage before moving to the calibration-regression step (`MAIAC ≈ a + b · MERRA2` per airshed).

## Ideas / under consideration
- None raised yet for this module.