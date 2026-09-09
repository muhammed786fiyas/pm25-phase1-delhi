# Task Log: MAIAC Gap-fill (MERRA-2 Calibration)
_Last updated: 2026-09-09_

## Scope
Fill MAIAC AOD gaps (48% of station-days missing, concentrated in monsoon) using a per-station linear calibration against MERRA-2 AOD, with a confidence score attached to every filled value, for the Delhi Phase 1 study window (2025-03-01 to 2026-02-28), across all 42 finalized CPCB stations.

## Completed

**Module scaffolding & calibration dataset** (2026-09-09)
- Built `scripts/datasets/maiac_gapfill/` following the repo's extract/QC/compute pattern, adapted since this module consumes two already-processed datasets rather than extracting from GEE.
- `01_build_calibration_dataset.py`: joins `maiac_aod_daily.csv` and `merra2_aod_daily.csv` on `[location_id, date]`, keeping only rows where both products have a value. Output: `data/raw/maiac_gapfill/calibration_dataset.csv`, 7,830 rows (163-205 overlap days per station, median 188.5 -- none below the QC floor).

**Calibration model selection** (2026-09-09)
- Built `06_test_calibration_variants.py` (exploratory, standalone, not wired into dvc.yaml/params.yaml) to compare four calibration approaches via leave-one-out CV, broken out by season. LOOCV RMSE:

| candidate | summer | monsoon | post_monsoon | winter | overall |
|---|---|---|---|---|---|
| baseline (pooled per-station, one a+b*MERRA2) | 0.189 | 0.338 | 0.382 | 0.276 | 0.288 |
| season-intercept (shared slope, seasonal offset) | 0.171 | 0.246 | 0.353 | 0.278 | 0.267 |
| season-specific fit (separate a,b per season) | 0.167 | 0.229 | 0.349 | 0.285 | 0.266 |

  Plus, for the 38 stations sharing one MERRA-2 grid cell: pooled MERRA2-only RMSE 0.2875, pooled + static covariates (NDVI/land-cover/roads/industrial fraction/elevation) RMSE 0.2878 -- no improvement.
- Full table: `data/interim/maiac_gapfill/calibration_variant_comparison.csv`.
- **Season-intercept promoted to production** -- see Key decisions.

**Production pipeline** (2026-09-09)
- `02_fit_station_calibration.py`: per-station OLS fit `MAIAC ~ a + b*MERRA2 + c_monsoon + c_post_monsoon + c_winter` (summer is the reference season, folded into `a`). All 42 stations fit cleanly: slope `b` 0.867-1.2, R^2 0.488-0.696.
- `03_qc_calibration.py`: hard-fail gate -- missing/unfitted stations, `n_obs < 20`, non-positive slope `b`. Result: 0 hard fails, 0 flags.
- `04_score_confidence.py`: leave-one-out CV per station using the season-intercept model, residuals grouped by `[location_id, season]` -> RMSE per group (168 groups: 42 stations x 4 seasons). Confidence bucket thresholds (`rmse_low_threshold`/`rmse_high_threshold`) use the repo's `"NOT_SET"` guard pattern -- script prints the full RMSE distribution and hard-fails until real cutoffs are set in params.yaml.
- `05_apply_gapfill.py`: builds the full 42-station x 365-day calendar grid, fills MAIAC gaps with `a + b*MERRA2 + season_offset` on days MERRA2 exists, merges `confidence_bucket`/`confidence_rmse` from script 04's output onto every row via `[location_id, season]` (populated for filled rows; `"observed"`/null for real MAIAC readings, since confidence describes the fill, not a measurement). Output: `data/processed/maiac_gapfill/maiac_aod_gapfilled.csv`.
- Result: 15,330 station-days total -- 7,969 observed (52.0%), 7,248 filled (47.3%), 113 unfillable (0.7%, mostly the 6 fully-missing MERRA-2 days in Sept 2025 already documented in `3-MERRA2_AOD.md`).
- Confidence bucket counts among filled rows: 2,370 high / 3,578 medium / 1,300 low.
- `params.yaml`: new `maiac_gapfill` section (`calibration`, `qc`, `fill`, `confidence` sub-sections). `dvc.yaml`: 5 new stages (`maiac_gapfill_build_dataset`, `_fit_calibration`, `_qc_calibration`, `_score_confidence`, `_apply_fill`), 51 stages total.
- Tagged `delhi-phase1-v17`.

## Key decisions
- **2026-09-09** -- Per-station calibration, not per-airshed (reaffirms the prior session's decision, given the scope change to 4 cities deprioritizing airshed stratification).
- **2026-09-09** -- Season-intercept promoted over full season-specific fit: nearly identical overall CV accuracy (0.267 vs 0.266 RMSE) but far more stable, since full season-specific fit refits an entire a/b line on as few as ~19 monsoon days per station. See comparison table above.
- **2026-09-09** -- Static-covariate-augmented pooled model (for the 38-station shared-MERRA2-cell cluster) not pursued into production -- no measurable CV improvement over MERRA2-only pooled.
- **2026-09-09** -- Confidence scoring is leave-one-out CV on the single production model, with residuals grouped by season only for reporting -- not a separate model fit per season. Confidence describes how well the real filling model performs when the true value happens to fall in each season.
- **2026-09-09** -- Confidence bucket thresholds are empirical terciles of the RMSE distribution across all 168 station x season groups, re-derived whenever the underlying calibration model changes. First pass (plain baseline model): 0.26/0.34. Final (after season-intercept promotion): 0.211/0.292.
- **2026-09-09** -- Both `confidence_bucket` and `confidence_rmse` kept in the final output, not just the bucket -- bucketing throws away information for free, and the two have different downstream uses (see Data notes).
- **2026-09-09** -- Pipeline reordered so confidence scoring (script 04) runs before the fill step (script 05), so confidence can be merged directly into `maiac_aod_gapfilled.csv` rather than left as a separate table requiring a manual join by whoever uses the data next.

## Data notes & gotchas
- `confidence_rmse` is a station x season *average*, not a per-day uncertainty -- every filled day in the same station+season group gets the identical value. Worth remembering before treating it as row-level precision.
- Two different downstream uses identified for the two confidence representations (discussed with Muhammed, not yet implemented -- modeling stage hasn't started): `confidence_bucket` for an LME sensitivity analysis (refit excluding low-confidence filled days, standard practice in AOD-gap-fill literature), and raw `confidence_rmse` as a LightGBM feature (trees find their own split thresholds, so a hand-picked bucket is actually *less* useful there than the continuous number -- the bucket was designed for the linear-model use case).
- Post-monsoon, not monsoon, has the worst calibration fit (RMSE ~0.35-0.38 across every model variant tried) despite having more overlap days (~46/station) than monsoon (~19/station) -- contradicts the literature-precedent expectation (central-India MAIAC/MERRA2 paper, Sci Reports 2021) going in. Not investigated further this session; possible explanation is post-monsoon's sharper day-to-day AOD swings (crop-burning/haze season) versus monsoon's washout-driven low-and-stable AOD, but this is a guess, not verified.
- The 6 fully-missing MERRA-2 days (Sept 2025, documented in `3-MERRA2_AOD.md`) show up correctly as part of the 113 unfillable station-days wherever MAIAC was also missing that day -- no special-casing was needed, the existing "both missing -> unfillable" branch in `05_apply_gapfill.py` handles it automatically.
- Same `.git/index.lock` and DVC-cache delete-permission gotchas as the OSM session (documented in the project context doc) -- resolved by requesting delete permission on the repo folder before running DVC this time.
- `dvc` was not installed in the device's Linux VM at session start (same as the OSM session) -- installed via pip mid-session. `pandas`/`numpy`/`pyyaml` were already present, so that part of the environment does persist across sessions even though `dvc` doesn't.

## Pending
- Investigate why post-monsoon calibration is worse than monsoon's, if it turns out to matter for the modeling stage (currently just noted, not resolved).
- Decide how `confidence_bucket`/`confidence_rmse`/`gap_filled` actually get used once the modeling stage starts (LME sensitivity analysis + LightGBM feature, per Data notes above) -- not yet implemented.
- CPCB<->AOD temporal join -- still open from DAY5, now unblocked (the gap-fill-calibration-order question that was blocking it is resolved: gap-fill ran independently of the CPCB join, directly off the already-processed MAIAC/MERRA2 daily files).

## Ideas / under consideration
- None new this session beyond what's captured in Data notes and Pending above.
