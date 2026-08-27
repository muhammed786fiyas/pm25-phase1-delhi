# Task Log: ERA5 Meteorology
_Last updated: 2026-08-26_

## Scope
All four meteorology covariates (β_met) for the LME model: Temperature, RH, Wind Speed (ERA5-Land, ~11km) and Boundary Layer Height (full ERA5, ~31km).

## Completed

**ERA5-Land (Temp, RH, Wind Speed)**
- Station-to-cell mapping (2026-08-2x): 42 stations → 15 unique cells
- Raw hourly extraction (2026-08-2x): quarterly-chunked after hitting GEE memory limit on unchunked full-year pull; 15 cells × 8,760 hours = 131,400 rows
- filterDate exclusive-end bug fixed (2026-08-2x): `study_end` moved from `2026-02-28` to `2026-03-01`
- Window + QC + derive (2026-08-2x): RH via Magnus-Tetens, Wind Speed via sqrt(u²+v²), coordinate tolerance 0.01°
- Daily aggregation (2026-08-2x): 15,330 station-days
- Overpass window hour-8 fix (2026-08-26): corrected `[5,6,7]` → `[5,6,7,8]`; reran window_qc and daily_aggregation

**ERA5 BLH (full ERA5)**
- Console sanity check (2026-08-26): confirmed `boundary_layer_height` band exists, units in meters, no scale factor; seasonal direction confirmed (summer 1402.91m > monsoon 912.10m)
- Station-to-cell mapping (2026-08-26): 42 stations → 4 unique cells
- Raw extraction (2026-08-26): 4 cells × 8,760 hours = 35,040 rows, no memory error (quarterly chunking applied from the start)
- Window + QC (2026-08-26): window_hours [5,6,7,8] from the start, coordinate tolerance 0.05°, QC range 0–6000m; 61,320 rows, BLH range 72.18–5035.4m
- Daily aggregation (2026-08-26): 15,330 station-days, n_hours_used=4 for all

## Key decisions
- BLH deferred until ERA5-Land core was complete, then built as its own module once core meteorology landed (decided 2026-08-2x, executed 2026-08-26)
- MERRA-2's block-averaged fields need only 3 window hours [5,6,7]; ERA5-Land and ERA5 BLH's instantaneous fields need 4 [5,6,7,8], since the 13:30 IST boundary is a snapshot point, not covered by a preceding average block (2026-08-26)
- Coordinate tolerance scaled to grid spacing rather than reused across modules: 0.01° for ERA5-Land (~11km), 0.05° for ERA5 BLH/MERRA-2 (~31/62km)
- `era5_blh` kept as a separate script namespace from `era5_land`, despite both feeding the same LME meteorology covariate group, since they're genuinely different GEE collections and grids

## Data notes & gotchas
- ERA5-Land's `filterDate(start, end)` treats `end` as exclusive — silent missing-last-day bug; same root cause independently found in MAIAC's extraction script
- GEE's `getRegion()` over a full year of hourly data can hit "User memory limit exceeded" non-deterministically; fixed by chunking the date range (quarterly) rather than pulling the whole year in one call
- `boundary_layer_height` is not visible in ERA5 Hourly's catalog page band table (very long list, cuts off before reaching it) — confirmed to exist only via direct `bandNames()` console check
- BLH values need no scale factor, unlike MAIAC's MODIS-derived AOD bands (×0.001) — ECMWF reanalysis bands store direct physical values
- ERA5-Land cell centers land on clean 0.1° grid marks (~11.1km); ERA5 (full) cell centers land on clean 0.25° grid marks (~27.8km) — confirms correct grid snapping, not a bug

## Pending
None — module complete.

## Ideas / under consideration
None raised for this module.