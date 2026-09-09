# Task Log: Modeling (master feature table + LME/LightGBM prep)
_Last updated: 2026-09-09_

## Scope
Build the final modeling-ready datasets for the Delhi Phase 1 pilot: join every processed covariate table (CPCB PM2.5 target, MAIAC/MERRA2 gap-filled AOD, ERA5-Land met, ERA5-BLH, NDVI, WorldCover land-use, SRTM terrain, OSM road/industrial/powerplant covariates) into one master feature table, then produce two model-specific prepared datasets -- one for the fixed LME equation (random intercept by station, AOD x season interaction), one for a LightGBM comparison model.

## Completed

**Master feature table** (2026-09-09)
- `scripts/modeling/dataset_prep/01_build_master_table.py`: starts from `pm25_daily_final.csv` (13,641 station-days, 42 stations) and left-joins everything else onto it. Canonical station name/lat/lon comes from `cpcb_stations_delhi_status.csv` filtered to `status == 'KEEP'`, not from any of the covariate tables (several of which carry their own copies).
- Time-varying joins on `[location_id, date]`: AOD gapfilled (renaming its `gap_filled` -> `aod_gap_filled` to avoid a name clash with NDVI's own `gap_filled` column), ERA5-Land (dropping `dewpoint_c` -- redundant with `relative_humidity`, an explicit earlier decision; renaming `n_hours_used` -> `n_hours_met`), ERA5-BLH (renaming `n_hours_used` -> `n_hours_blh`).
- NDVI is 5-day composites, not daily, so it needed a period-range join instead of a direct date match: `pd.merge_asof(direction="backward", by="location_id")` matching each station-day to the period whose `period_start` is the closest one at-or-before that date, then verifying every matched date actually falls at-or-before `period_end` too (0 rows failed this check across all 13,641 rows). Its own `gap_filled` column renamed to `ndvi_gap_filled`.
- Static per-station joins on `location_id` alone: WorldCover (all 11 land-use % columns kept in the master table -- reduction happens downstream, differently per model), SRTM terrain, OSM road density/industrial fraction/powerplant distance.
- Master table: 13,641 rows x 42 columns. Missing-value check: `aod_055` missing for 101 rows (station-days where both MAIAC and MERRA2 were unavailable, and which also happened to have a valid PM2.5 reading -- a subset of the 113 fully-unfillable station-days in the full AOD calendar grid), `confidence_rmse` missing for 7,093 rows (these are the *observed*, non-gap-filled AOD rows, where confidence doesn't apply), `nearest_powerplant_name` missing for 10,007 rows (informational only, not a feature).

**LME-ready dataset** (2026-09-09)
- `scripts/modeling/dataset_prep/02_prepare_lme_dataset.py`. Complete-case filter: drops any row with a NaN in a model column. In practice only `aod_055` ever triggers this (met/NDVI/land-use/terrain/OSM covariates have 0 missing across all 13,641 rows) -- 13,641 -> 13,540 rows.
- WorldCover reduced to 6 columns for LME: dropped `snow_ice_pct`/`mangroves_pct`/`moss_lichen_pct` (literally 0% at all 42 Delhi stations -- verified, not assumed), `wetland_herbaceous_pct` (nonzero at only 3 of 42 stations, max 1.28% -- negligible), and `water_pct` as the implicit reference category (smallest mean at 0.51%, present at only 55% of stations) to resolve the perfect multicollinearity from the 11 columns summing to a constant 100%. Kept: `tree_cover_pct, shrubland_pct, grassland_pct, cropland_pct, built_up_pct, bare_sparse_veg_pct`.
- Every continuous regressor (AOD, met, NDVI, terrain, OSM, the 6 kept land-use columns) is z-scored (mean 0, std 1); scaling stats written to a companion `lme_scaling_params.csv` so fitted coefficients can be back-transformed to original units later.
- Season dummies built with `summer` as the reference category (`season_monsoon`, `season_post_monsoon`, `season_winter`), then AOD x season interaction columns (`aod_x_monsoon`, `aod_x_post_monsoon`, `aod_x_winter`) built from the *scaled* AOD -- summer's interaction is implicitly the plain `aod_055` coefficient and needs no separate column.
- `confidence_bucket` kept in the output as metadata only -- not part of the regression, but available for a future sensitivity refit that excludes low-confidence filled AOD days.
- Result: 13,540 rows x 29 columns. Rows per season: summer 3,481 / monsoon 4,617 / post_monsoon 2,375 / winter 3,067, all 42 stations represented in every season. Rows per `confidence_bucket`: observed 6,992 / medium 3,180 / high 2,151 / low 1,217.

**LightGBM-ready dataset** (2026-09-09)
- `scripts/modeling/dataset_prep/03_prepare_lightgbm_dataset.py`. No complete-case filtering -- all 13,641 rows kept, native NaNs preserved (101 rows with NaN `aod_055`, 7,093 with NaN `confidence_rmse`).
- WorldCover reduced to 8 columns (only the 3 always-zero columns dropped) -- deliberately *not* matched to LME's 6, since `water_pct` and `wetland_herbaceous_pct` carry real (if modest) station-to-station variation and trees pay no collinearity cost for keeping them, unlike a linear model.
- `confidence_rmse`, `aod_gap_filled`, `ndvi_gap_filled` kept as informative features (not just for filtering) -- the confidence score reflects real MAIAC-vs-MERRA2 calibration fit quality, a genuine signal a tree model can use.
- `location_id` kept in the file as an ID/grouping column (e.g. for station-based CV splits) but documented as excluded from the actual training feature list. `season` kept as a plain string (CSV can't carry dtype info -- documented to cast to category or pass `categorical_feature=['season']` at training time).
- Result: 13,641 rows x 27 columns. Rows per season: monsoon 4,718 / summer 3,481 / winter 3,067 / post_monsoon 2,375.

**Pipeline wiring** (2026-09-09)
- `params.yaml`: new `modeling` section (`master_table`, `lme_prep`, `lightgbm_prep` sub-sections -- reference season, WorldCover drop lists, scale column list).
- `dvc.yaml`: 3 new stages (`build_master_feature_table`, `prepare_lme_dataset`, `prepare_lightgbm_dataset`), replacing the placeholder "Modeling -- add stages here" comment. 54 stages total.
- Ran all 3 stages via `dvc repro -s <stage>`, verified `dvc status` reports everything up to date.
- Tagged `delhi-phase1-v18`.

## Key decisions
- **2026-09-09** -- WorldCover collinearity handled differently per model, not with one shared reduced set: LME drops to 6 columns (reference-category drop, keeping fine-grained land-use types rather than combining into broad groups -- standard practice in PM2.5/LUR literature since built-up vs. tree-cover vs. cropland carry distinct physical meaning). LightGBM keeps 8 columns, since collinearity isn't a cost for tree models and `water_pct` has genuine signal worth keeping.
- **2026-09-09** -- Confidence/gap-fill metadata (`confidence_bucket`) stays out of the base LME regression entirely -- kept as a non-regressor column for a later, separate sensitivity-analysis refit, not fit directly.
- **2026-09-09** -- The 113 unfillable AOD station-days (101 of which overlap a PM2.5 reading) are not special-cased: LME drops them via ordinary complete-case filtering, LightGBM keeps them with native NaN in `aod_055`.
- **2026-09-09** -- AOD x season interaction columns are built from the *scaled* AOD value, not the raw value -- caught as a bug during verification (interactions were originally built before the scaling step ran, leaving them on a different scale than the final `aod_055` column) and fixed by reordering scale-then-interact.

## Data notes & gotchas
- NDVI's period-range join uses `pd.merge_asof`, which requires the "on" column (`date`/`period_start`) sorted *globally*, not just within each `by` group (`location_id`) -- sorting by `["location_id", "date"]` first (grouped order) raises `ValueError: left keys must be sorted`. Fix: sort by `date`/`period_start` alone, let `by=` handle the station matching, then re-sort the output back to `[location_id, date]` order afterward.
- `master_feature_table.csv` is a deliberate superset -- it keeps metadata columns (`hours_used`, `n_hours_met`, `n_hours_blh`, `fill_source`, `confidence_bucket`, `confidence_rmse`, `nearest_powerplant_name`, NDVI period fields) that aren't features in either prepared dataset, so nothing is lost if a different reduction is wanted later without re-running the full join.
- `dvc` needed reinstalling this session (same non-persistence noted in every prior session's log); `pandas`/`yaml` did persist.

## Pending
- Fit the actual LME model (M0 baseline first per `LME_Architecture_M0_A_B_C_SUMMARY.md`) and a LightGBM comparison model -- these prepared datasets are the input, not the model fit itself.
- Run the confidence_bucket sensitivity analysis (refit LME excluding `confidence_bucket == 'low'` filled AOD days, compare to the full-data fit).
- CPCB<->AOD temporal join question from DAY5 -- the master table's `[location_id, date]` join likely resolves this in practice, but worth explicitly confirming against whatever DAY5's original concern was.

## Ideas / under consideration
- None new this session.


## Note (2026-09-09, later same day)
Moved the 3 dataset-prep scripts into `scripts/modeling/dataset_prep/` (previously flat in `scripts/modeling/`), to make room for upcoming `scripts/modeling/lme/` and `scripts/modeling/lightgbm/` model-fitting sub-pipelines -- mirrors how `scripts/datasets/` is itself a folder of per-module subfolders rather than one flat sequence. `dvc.yaml`'s 3 stage `cmd`/`deps` paths updated to match; re-ran all 3 stages (identical output, only the recorded script path changed in `dvc.lock`).
