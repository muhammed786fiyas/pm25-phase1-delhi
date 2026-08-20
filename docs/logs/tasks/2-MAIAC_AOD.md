# Task Log: MAIAC AOD
_Last updated: 2026-08-20_

## Scope
Google Earth Engine extraction of MAIAC (MCD19A2) AOD at 550nm for all Delhi/NCR stations,
raw overpass level, feeding the AOD side of the ST-LUR feature set.

## Completed
**GEE account setup**
- Signed up for Google Earth Engine, created Cloud Project `internship-pm25` (confirmed no dot — earlier note with a dot was incorrect and has been corrected in memory)
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
- Wrote `scripts/datasets/maiac_aod/1-extract_gee_covariates.py` (raw overpass level, single-pixel point extraction, no 3x3 buffer averaging yet)
- Columns: `location_id`, `date`, `image_id`, `aod_055`, `aod_047`, `aod_uncertainty`, `aod_qa`
- Tested on 3 known stations, then ran the full extraction against all **56 candidate stations** (KEEP + DROP together, unfiltered) — 17,767 rows, output at `data/raw/maiac/maiac_aod_raw.csv`
- DVC stage `maiac_extract` added to `dvc.yaml`; output tracked via `dvc commit --force` (extraction already run manually, no need to re-execute)

**Station finalization (dependency, tracked in `cpcb.md`)**
- CPCB completeness QC is now complete — station list finalized at **42 stations**, recorded in `cpcb_stations_delhi_status.csv`
- MAIAC raw extraction still needs to be filtered down to these 42 (see Pending)

## Key decisions
- **2026-08-17** — Use AOD at 550nm (`Optical_Depth_055`), not 470nm — matches MERRA-2's reporting wavelength (needed for the gap-fill calibration step), matches AERONET/field-standard convention, and matches both literature anchors (STMEM, Central India paper).
- **2026-08-17** — Keep extraction at raw overpass level for now (no Terra/Aqua averaging yet) — aggregation deferred to preprocessing so raw per-overpass data (and overpass count/quality) isn't lost.
- **2026-08-17** — Use single-pixel point extraction (not 3x3 buffer averaging) for the first working version — get the core pipeline correct end to end first; buffer averaging is a small, isolated add-on later.
- **2026-08-17** — Ran the full extraction against the *unfiltered* 56-station list rather than waiting for CPCB QC — avoids re-running slow GEE extraction if the finalized list changes; filtering the output afterward is cheap.
- **2026-08-20** — Preprocessing order confirmed: filter to 42 finalized stations → scale `aod_055`/`aod_047` (×0.001) → check/apply `aod_uncertainty` scale factor → decode `AOD_QA` bitmask, drop/flag bad pixels → duplicate check (`location_id`+`image_id`) → aggregate to daily (mean of valid overpasses, mirroring CPCB's own daily aggregation) → sanity-check ranges.
- **2026-08-20** — Aggregate to daily *before* the CPCB join (not after) — CPCB is already daily, so matching AOD to that shape first makes the join a straightforward merge on `location_id` + `date`.
- **2026-08-20** — Do NOT drop CPCB rows lacking AOD during the join — use a **left join** with CPCB as the base table, so PM2.5 days with missing AOD are preserved for MERRA-2 gap-fill to handle, not silently lost. Plan to add an `aod_source` column (`'maiac'` now, `'merra2_gapfilled'` later) at/after merge time.
- **2026-08-20** — Plan to externalize script constants (`GEE_PROJECT`, `MAIAC_COLLECTION`, `BANDS_TO_PULL`, `STUDY_START`, `STUDY_END`) into `params.yaml` under a `maiac_extract` key, matching the CPCB scripts' pattern, and register them under `params:` in the `dvc.yaml` stage. Not yet implemented.

## Data notes & gotchas
- MAIAC AOD values are stored as scaled integers, not raw decimals — scale factor is 0.001. `aod_055 = 461` in the raw CSV means real AOD ≈ 0.461. **Not yet applied.**
- `aod_uncertainty`'s scale factor has not been confirmed yet — do not assume it's the same 0.001 as the AOD bands; check the catalog spec before applying.
- `AOD_QA` is a packed bitmask, not a simple flag — raw values captured (e.g. 1, 865, 1057, 8193, 9249, 16385) but not yet decoded into cloud/quality categories. Need the exact MAIAC QA bit-layout spec before writing the decode logic.
- July–August 2025 (peak monsoon) show a sharp drop in valid overpasses across stations tested — expected, matches known MAIAC cloud-loss behavior in monsoon, not a pipeline bug.
- `filterBounds` returning ~395 "matches" for a single point over 5 days (vs. ~6 genuinely valid ones) was the first sign something was off — root cause was MODIS swath geometry, not a code bug.
- GEE web console can throw `Request had invalid authentication credentials` if the wrong Cloud Project is selected in the Code Editor's project selector, or if the browser session is on a different Google account than the one used for `earthengine authenticate`.
- Each station's extraction is fully independent — `filterBounds` + `reduceRegion` both run fresh per station inside `extract_station_data()`, not once for the whole region. `filterBounds` has no concept of tile names (`h24v06` etc.) — it only checks geometric footprint overlap, which is why it let through false positives from unrelated tiles.

## Pending
- Filter `maiac_aod_raw.csv` down to the finalized 42 stations (CPCB QC dependency now resolved — this can proceed).
- Apply the ×0.001 scale factor to `aod_055` / `aod_047`; confirm and apply `aod_uncertainty`'s scale factor.
- Decode `AOD_QA` bitmask into usable cloud/quality flags and drop/flag bad-quality pixels.
- Duplicate check (`location_id` + `image_id`).
- Aggregate to daily (mean of valid overpasses).
- Temporal alignment: left join with CPCB (`cpcb_daily.merge(aod_daily, on=['location_id','date'], how='left')`), add `aod_source` column.
- Externalize script constants into `params.yaml` + update `dvc.yaml` stage's `params:` key.
- Decide on and implement Terra vs Aqua identification (via overpass index / acquisition time) — still open, not yet needed for daily aggregation but may matter later.

## Ideas / under consideration
- 3x3 pixel buffer averaging around each station point (instead of single-pixel) — deferred, not rejected. Revisit once single-pixel results are validated; compare buffer vs. single-pixel to see if it meaningfully changes values before deciding whether to adopt it.