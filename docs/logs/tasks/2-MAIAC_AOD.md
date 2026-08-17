# Task Log: MAIAC AOD
_Last updated: 2026-08-17_

## Scope
Google Earth Engine extraction of MAIAC (MCD19A2) AOD at 550nm for all Delhi/NCR stations,
raw overpass level, feeding the AOD side of the ST-LUR feature set.

## Completed
**GEE account setup**
- Signed up for Google Earth Engine, created Cloud Project `internship-pm2.5`
- Enabled the Google Earth Engine API for the project via Cloud Console
- Installed `earthengine-api` Python package, authenticated via `earthengine authenticate`
- Verified setup end to end with a test script (`ee.Initialize` + simple `getInfo()` call)

**Collection and band validation (Code Editor, JavaScript)**
- Confirmed correct collection ID: `MODIS/061/MCD19A2_GRANULES` — the `006` version is deprecated, with data ending 2023-02-17, before the study window
- Confirmed band names: `Optical_Depth_055` (550nm, primary AOD band), `Optical_Depth_047` (470nm), `AOD_Uncertainty`, `AOD_QA`
- Discovered `filterBounds` alone is unreliable — it returns false-positive matches from unrelated MODIS tiles (`h15v17`–`h20v17`) whose swath footprint diagonally overlaps Delhi's longitude, even though the actual valid tile is `h24v06`
- Validated the fix: per-image `reduceRegion` at the station point, filtering out nulls, correctly isolates real overpasses (confirmed only `h24v06` images survive)
- Confirmed `h24v06` is the correct MODIS sinusoidal grid tile for Delhi/NCR (consistent with the Central India paper's tile reference for Madhya Pradesh)

**Python extraction script**
- Wrote `extract_gee_covariates.py` (raw overpass level, single-pixel point extraction, no 3x3 buffer averaging yet)
- Columns: `location_id`, `date`, `image_id`, `aod_055`, `aod_047`, `aod_uncertainty`, `aod_qa`
- Tested on 3 known stations (13, 103, 236) — output sane, all overpasses correctly from `h24v06`, ~300 valid overpasses per station over the year, matching expected cloud-loss pattern
- Ran the full extraction across the unfiltered station list (all candidate stations, not yet QC-filtered)
- Output: `data/raw/maiac/maiac_aod_raw.csv`

## Key decisions
- **2026-08-17** — Use AOD at 550nm (`Optical_Depth_055`), not 470nm — matches MERRA-2's reporting wavelength (needed for the gap-fill calibration step), matches AERONET/field-standard convention, and matches both literature anchors (STMEM, Central India paper).
- **2026-08-17** — Keep extraction at raw overpass level for now (no Terra/Aqua averaging yet) — aggregation is deferred to a later processing step so raw per-overpass data (and overpass count/quality) isn't lost.
- **2026-08-17** — Use single-pixel point extraction (not 3x3 buffer averaging) for the first working version. Reasoning: get the core pipeline correct and working end to end first; buffer averaging is a small, isolated add-on later (just changes the geometry + reducer in one function) and having single-pixel values first gives a baseline to measure how much the buffer actually changes results.
- **2026-08-17** — Ran the full extraction against the *unfiltered* station list rather than waiting for CPCB QC to finalize the station set first. Filtering AOD down to the finalized station list will happen as a separate step after CPCB completeness QC completes — this avoids re-running GEE extraction (slow, many `getInfo()` calls) if the finalized list changes.

## Data notes & gotchas
- MAIAC AOD values are stored as scaled integers, not raw decimals — scale factor is 0.001. `aod_055 = 461` in the raw CSV means real AOD ≈ 0.461. **Not yet applied** — raw script captures unscaled values on purpose (scale in a later processing step).
- `AOD_QA` is a packed bitmask, not a simple flag — raw values captured (e.g. 1, 865, 1057, 8193, 9249, 16385) but not yet decoded into cloud/quality categories.
- July–August 2025 (peak monsoon) show a sharp drop in valid overpasses across stations tested — expected, matches known MAIAC cloud-loss behavior in monsoon, not a pipeline bug.
- `filterBounds` returning ~395 "matches" for a single point over 5 days (vs. ~6 genuinely valid ones) was the first sign something was off — root cause was MODIS swath geometry, not a code bug. Documented in Code Editor testing above so it isn't rediscovered later.
- GEE web console can throw `Request had invalid authentication credentials` if the wrong Cloud Project is selected in the Code Editor's project selector, or if the browser session is on a different Google account than the one used for `earthengine authenticate` — check the project selector first if this recurs.

## Pending
- Filter `maiac_aod_raw.csv` down to the finalized station list once CPCB completeness QC (see `cpcb.md`) determines which stations survive.
- Apply the ×0.001 scale factor to `aod_055` / `aod_047`.
- Decode `AOD_QA` bitmask into usable cloud/quality flags and drop bad-quality pixels.
- Temporal alignment: join cleaned AOD to CPCB PM2.5 (overpass-window-averaged), per the Blueprint's Section 5.1.
- Decide on and implement Terra vs Aqua identification (via overpass index / acquisition time) for later temporal-window matching.

## Ideas / under consideration
- 3x3 pixel buffer averaging around each station point (instead of single-pixel) — deferred, not rejected. Revisit once single-pixel results are validated against known station values; compare buffer vs. single-pixel to see if it meaningfully changes values before deciding whether to adopt it.