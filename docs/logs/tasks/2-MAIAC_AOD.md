# Task Log: MAIAC AOD
_Last updated: 2026-08-21_

## Scope
Google Earth Engine extraction and preprocessing of MAIAC (MCD19A2) AOD at 550nm for all
Delhi/NCR stations, from raw overpass-level extraction through to a clean daily station-level
dataset ready for temporal alignment with CPCB.

## Completed
**GEE account setup** — done 2026-08-17. GEE Cloud Project `internship-pm25` (no dot), `earthengine-api` installed and authenticated.

**Collection and band validation (Code Editor)** — done 2026-08-17. Collection `MODIS/061/MCD19A2_GRANULES`, bands `Optical_Depth_055` (primary), `Optical_Depth_047`, `AOD_Uncertainty`, `AOD_QA`. `filterBounds` false-positive issue found and worked around with per-image `reduceRegion` + null-filtering.

**Script 1 — raw extraction** — done 2026-08-17/20. `scripts/datasets/maiac_aod/1-extract_gee_covariates.py`. Raw overpass level, single-pixel point extraction. Ran against all 56 candidate stations (unfiltered): 17,767 rows, `data/raw/maiac/maiac_aod_raw.csv`. DVC stage `maiac_extract`.

**Station finalization dependency resolved** — CPCB completeness QC completed 2026-08-20, station list finalized at 42 (see `cpcb.md`).

**Script 2 — filter and scale** — done 2026-08-21. `scripts/datasets/maiac_aod/2-filter_and_scale_aod.py`. Filtered to 42 finalized stations (17,767 → 13,339 rows), applied official NASA scale factors (`aod_055`/`aod_047` ×0.001, `aod_uncertainty` ×0.0001, both confirmed from the MCD19A2 User Guide Table 5.3), fill-value (-28672) handling before scaling, duplicate check (0 found), range check (0 out of range). Output: `data/interim/maiac_aod/filter_scale/maiac_aod_filtered_scaled.csv`. DVC stage `aod_filter_scale`, with `params.yaml` section of the same name.

**Script 3 — QA bitmask decode** — done 2026-08-21. `scripts/datasets/maiac_aod/3-decode_qa.py`. Decoded all `AOD_QA` fields (cloud mask, adjacency mask, qa_aod, glint mask, aerosol model) into labeled columns using the confirmed official bit layout (NASA User Guide Table 5.4). No filtering — outputs full decoded dataset plus a `qa_summary.csv` for human review. DVC stage `aod_decode_qa`.

**Script 4 — QA filter** — done 2026-08-21. `scripts/datasets/maiac_aod/4-apply_qa_filter.py`. Applies the human-reviewed strictness decision from `params.yaml` (`aod_qa_filter.strictness`), guarded against a `"NOT_SET"` placeholder so `dvc repro` fails safely until a real decision is made. Chosen strictness: **moderate** (keep best quality + 1-neighbor-cloud). Result: 13,339 → 12,241 rows (91.8% kept), all 42 stations retained with none losing all data. DVC stage `aod_qa_filter`.

**Script 5 — daily aggregation** — done 2026-08-21. `scripts/datasets/maiac_aod/5-aggregate_aod_daily.py`. Aggregates valid overpasses to one row per station-day (mean of `aod_055`/`aod_047`/`aod_uncertainty`), adds `n_overpasses` and `season` columns. Also outputs a per-station-per-season count file (with a `total` column) mirroring CPCB's own completeness-check shape, and an `n_overpasses` distribution summary. Output: `data/processed/maiac_aod/maiac_aod_daily.csv`. DVC stage `aod_aggregate_daily`.

**Final dataset verified** — season labels correct, `aod_047` consistently > `aod_055` (expected physical relationship), July-August monsoon gap consistent across all 42 stations, `n_overpasses` only 1 or 2 (matches QA-filtered data). AOD coverage per station: ~165-208 days out of 365 (45-57%), heavily concentrated gap in monsoon (~10-22% vs CPCB's ~86-97%).

**AOD preprocessing pipeline (scripts 1-5) is fully DONE.**

**Git** — tagged `delhi-phase1-v8` (QA decode+filter) and `delhi-phase1-v9` (full AOD preprocessing complete).

## Key decisions
- **2026-08-17** — Use AOD at 550nm (`Optical_Depth_055`), not 470nm — matches MERRA-2's reporting wavelength, AERONET convention, and both literature anchors.
- **2026-08-17** — Raw overpass level (no Terra/Aqua averaging), single-pixel (no 3x3 buffer) for the first working version — both deliberate, deferred design choices.
- **2026-08-17** — Ran full extraction against unfiltered 56-station list rather than waiting for CPCB QC — avoids re-running slow GEE extraction; filtering afterward is cheap.
- **2026-08-20/21** — Preprocessing order: filter to 42 stations → scale → decode QA → filter QA → aggregate daily → (then) temporal alignment.
- **2026-08-21** — QA filtering strictness set to **moderate**, directly grounded in NASA's documented guidance (single-cloud-adjacency "often represents false cloud detection").
- **2026-08-21** — Human-judgment pipeline steps use a `"NOT_SET"` placeholder pattern in `params.yaml` with a guard clause, so `dvc repro` fails safely rather than silently proceeding with an undecided threshold. This generalizes to future reruns on different regions/times.
- **2026-08-21** — `data/interim/maiac_aod/` restructured into per-stage subfolders (`filter_scale/`, `qa_decode/`, `qa_filter/`) to satisfy DVC's non-overlapping-output requirement.
- Plan (still pending) — left join with CPCB as base table for temporal alignment, add `aod_source` column (`'maiac'` now, `'merra2_gapfilled'` later). Do NOT drop CPCB rows lacking AOD — MERRA-2 gap-fill is meant to fill those.

## Data notes & gotchas
- MCD19A2 official scale/fill values (NASA LP DAAC User Guide V61): `Optical_Depth_047`/`Optical_Depth_055` scale 0.001, fill -28672, valid range -100 to 8000 (raw). `AOD_Uncertainty` scale 0.0001 (NOT the same as AOD bands), fill -28672, valid range 0 to 30000 (raw).
- `AOD_QA` bit layout (16-bit uint): bits 0-2 cloud mask, 3-4 land/water/snow/ice, 5-7 adjacency mask, 8-11 QA-for-AOD (0000=best quality, this is NASA's own pre-computed combination of cloud+adjacency), 12 glint mask, 13-14 aerosol model, 15 reserved.
- `qa_aod` (bits 8-11) directly aggregates `cloud_mask` + `adjacency_mask` — confirmed by matching row counts exactly between the fields in this dataset's QA summary.
- July–August (peak monsoon) shows a sharp, consistent drop in valid overpasses across all 42 stations — expected MAIAC cloud-loss behavior, not a pipeline bug. This is the core justification for MERRA-2 gap-fill.
- `filterBounds` returning ~395 "matches" for a single point over 5 days (vs. ~6 genuinely valid ones) was the first sign something was off in early testing — root cause was MODIS swath geometry, not a code bug.
- Each station's extraction (`filterBounds` + `reduceRegion`) is fully independent per station — not a shared regional filter.
- `image_id` structure: `MCD19A2_A{YYYYDDD}_h{HH}v{VV}_{version}_{production timestamp}_{overpass index}` — e.g. acquisition date is Julian day format, production timestamp shows processing delay (~2 days after acquisition typically), overpass index distinguishes multiple same-day passes.

## Pending
- Decide order: gap-fill calibration before or after CPCB↔AOD temporal alignment join — open question.
- Temporal alignment script: left join `maiac_aod_daily.csv` with CPCB daily PM2.5 on `[location_id, date]`.
- MAIAC-MERRA2 gap-fill calibration: fit `MAIAC_AOD ~= a + b*MERRA2_AOD` per airshed, apply to fill AOD gaps (mostly monsoon).
- Confirm MERRA-2's final output file path/name (pipeline reported complete, exact path not yet in this session's context) before referencing it in the join/gap-fill script.
- Externalize `extract_gee_covariates.py`'s hardcoded constants (`GEE_PROJECT`, `MAIAC_COLLECTION`, `BANDS_TO_PULL`, `STUDY_START`, `STUDY_END`) into `params.yaml` — planned, not yet done.
- Decide on and implement Terra vs Aqua overpass identification — still open, not blocking.

## Ideas / under consideration
- 3x3 pixel buffer averaging around each station point (instead of single-pixel) — deferred, not rejected. Compare against current single-pixel values before deciding.