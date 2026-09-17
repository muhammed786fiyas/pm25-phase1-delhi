# Task Log: LME Model (fit + validation)
_Last updated: 2026-09-17_

## Scope

Fit and validate the single Delhi Phase 1 LME equation on `lme_ready_dataset.csv`:

```
PM2.5 = b0 + (b1 + b_int*Season)*AOD + b_met*Met + b_lu*LandUse + b_s*Season + u_i + eps
```

Random intercept only, grouped by station (`location_id`). No airshed stratification (the M0/A/B/C ladder in `LME_Architecture_M0_A_B_C_SUMMARY.md` is retired and out of scope -- see the project context doc's 2026-09-09 note). No tree-based models here -- LightGBM is a separate future chat.

## Completed

### 1. Model fit (`scripts/modeling/lme/01_fit_lme_model.py`)

- Fixed effects: `aod_055`, `aod_x_monsoon`, `aod_x_post_monsoon`, `aod_x_winter` (AOD + AOD x season interaction, summer is the reference season so has no separate interaction column), `season_monsoon`/`season_post_monsoon`/`season_winter`, the 4 ERA5 meteorology columns (`temperature_c`, `relative_humidity`, `wind_speed`, `boundary_layer_height`), and everything else as "LandUse" (6 WorldCover fractions, `ndvi_mean`, `elevation_m`, `slope_deg`, `road_density_km_per_km2`, `industrial_landuse_fraction`, `dist_to_nearest_powerplant_km`) -- 23 fixed effects + intercept = 24 fixed-effect params. Random intercept only by `location_id` (no `re_formula` passed -- statsmodels defaults to random-intercept-only).
- Fit via `statsmodels.formula.api.mixedlm(...).fit(reml=True)` on the full 13,494-row dataset (42 stations).
- `MixedLMResults.aic`/`.bic` come back `NaN` in the installed statsmodels version (0.15.0) -- `df_modelwc` isn't populated for `MixedLMResults`. Computed by hand instead: `k = k_fe + n_variance_components` where `n_variance_components = 1 (random-intercept variance) + 1 (residual variance) = 2` for this random-intercept-only model, then `AIC = -2*llf + 2*k`, `BIC = -2*llf + k*ln(n)`.
- Outputs: `models/lme/lme_full_model.pkl` (pickled `MixedLMResults`, DVC-tracked), `reports/lme/primary_log_target/lme_fixed_effects.csv` (coefficient table: term/coef/std_err/z/p_value/ci_lower/ci_upper, fixed effects only -- filtered out the "Group Var" row that `result.params`/`.bse`/etc. otherwise include alongside the fixed effects), `reports/lme/primary_log_target/lme_model_summary.txt` (full statsmodels summary text + logLik/AIC/BIC/variance components).

**Fit results** (see `reports/lme/primary_log_target/lme_fixed_effects.csv` for full table):
- `aod_055` (summer/reference-season AOD slope): coef = 0.219, p < 0.001 -- positive and significant.
- `aod_x_monsoon`: coef = -0.094, p < 0.001 -- AOD's effect is meaningfully weaker in monsoon than summer (matches the physical expectation: wet scavenging decouples column AOD from surface PM2.5 in monsoon). `aod_x_post_monsoon`/`aod_x_winter` are not significant (AOD slope not distinguishable from the summer reference in those seasons).
- `season_post_monsoon` (coef = 0.605) and `season_winter` (coef = 0.705) are large and highly significant on the log scale -- Delhi's well-known winter/post-monsoon PM2.5 spike, captured as a level shift independent of AOD.
- Meteorology: `temperature_c`, `relative_humidity`, `wind_speed` all significant and negative (higher temp/humidity/wind associated with lower PM2.5, physically sensible -- wind disperses, humidity/temp effects consistent with the boundary-layer/mixing story). `boundary_layer_height` significant, small negative effect.
- LandUse: none of the 6 WorldCover fractions, elevation, slope, road density, industrial fraction, or powerplant distance are significant at p < 0.05 -- consistent with the DAY8 EDA's VIF finding (`built_up_pct`/`tree_cover_pct`/`cropland_pct` still elevated even after the reference-category drop), which widens standard errors on exactly these terms. Not a new problem, already a documented limitation.
- Random-intercept variance (station): 0.0233. Residual variance: 0.258. logLik = -10134.4, AIC = 20320.9, BIC = 20516.1.

### 2. Validation (`scripts/modeling/lme/02_validate_lme_cv.py`)

**2km buffer exclusion** (recomputed from station lat/lon every run via haversine distance, not hardcoded): of the 42 KEEP stations, 12 have at least one neighbor within 2km. 4 of those (Shadipur, Lodhi Road IMD, Lodhi Road IITM, Jawaharlal Nehru Stadium) each lose 2 neighboring stations from their fold's training set (they form two overlapping ~3-station clusters in central Delhi/Lodhi Road), the other 8 lose 1 each. Matches Muhammed's brief exactly (verified by hand before wiring into the script). Written to `reports/lme/primary_log_target/station_buffer_exclusions.csv` (each validation variant gets its own copy under its own `reports/lme/<variant>/` subfolder -- see "Pipeline wiring" below).

**Spatial LOSO CV (primary)**: 42 folds, one station held out per fold plus its buffer-excluded neighbors dropped from that fold's training set, prediction is fixed-effects-only (no random intercept available for an unseen station).
- Pooled (all 42 folds' held-out rows combined): R2 = 0.423, within-R2 = **0.211** (Kawano-style formula, see below -- was -0.335 under the original shared-demeaning formula), RMSE = 0.638, MAE = 0.453 (log-PM2.5 scale). Naive (uncorrected) back-transform: RMSE = 77.8 ug/m3, MAE = 40.1 ug/m3.
- Per-fold detail in `reports/lme/primary_log_target/cv_spatial_loso_folds.csv`. Most stations land in a plausible 0.4-0.75 R2 / ~0.1-0.35 within-R2 range. Two stations are severe R2 outliers: station 5598 ("Sector-125, Noida, UP - UPPCB", just across the Delhi/UP border, 4.89km to its nearest neighbor) at R2 = -3.43, and station 6934 ("Dr. Karni Singh Shooting Range, Delhi - DPCC", 3.64km to nearest neighbor) at R2 = -1.44. Neither is a 2km-buffer artifact (0 excluded neighbors for either). **Investigated and resolved**: both sit at genuine extremes of the land-use covariate space relative to the other 41 stations -- 6934's `shrubland_pct` (5.88) and `road_density_km_per_km2` (-2.11) fall literally outside the other stations' range, 5598's `elevation_m` (-1.44) is below every other station's minimum, with `industrial_landuse_fraction` at the boundary. LOSO forces the model to extrapolate its fitted linear land-use coefficients into land-use territory it never saw in training for these two -- the classic linear-extrapolation failure mode, not a data-quality bug. Ruled out an oversized station-specific baseline (both stations' fitted random intercepts sit at unremarkable 31st/67th percentiles among all 42). Notably, on the Kawano within-R2 formula these two stations are *not* outliers at all (5598 = 0.190, 6934 = 0.235, both in the normal per-fold range) -- their R2 is still badly negative (a real per-station level-calibration problem worth addressing, e.g. via winsorizing/robust regression, see Pending), but within-R2 specifically no longer flags them because independent demeaning cancels out exactly the level offset that was driving their old-formula within-R2 to -10.42 / -5.93.
- Some CV folds triggered a `ConvergenceWarning` ("Maximum Likelihood optimization failed to converge... Retrying with lbfgs") during refitting on the reduced training set -- did not investigate which folds specifically; worth a closer look if these recur when the LightGBM comparison stage's CV protocol reuses the same buffer-exclusion logic.

**Random CV (comparison, same model/hyperparameters, only the fold-assignment method differs)**: 42 random folds (row-level, ignores station grouping), matched 1:1 with the spatial-LOSO fold count for a clean side-by-side comparison. Prediction uses fixed effects **plus** the fitted random intercept (`result.random_effects[station]`) when that station has other days in the fold's training set -- this is what actually produces the leakage effect, not the fold split alone.
- Pooled: R2 = 0.634, within-R2 = **0.206** (Kawano-style; was 0.152 under the old formula), RMSE = 0.509, MAE = 0.375.
- **Random CV inflates R2 relative to spatial LOSO** (0.634 vs 0.423) on the identical model and hyperparameters -- only the fold-assignment/prediction method differs. Matches Kawano et al. and the other benchmark papers in the project's literature docs: the random intercept `u_i` is essentially a memorized per-station baseline once a station has any training-fold representation, making its held-out days artificially easy to predict. Per-fold detail in `reports/lme/primary_log_target/cv_random_folds.csv`. Note: under the Kawano within-R2 formula this gap almost disappears (0.211 vs 0.206) even though the R2 gap stays large -- consistent with the leakage effect being mostly a per-station *level*-calibration artifact that independent demeaning cancels out, not a difference in day-to-day anomaly tracking.

**Gap-fill robustness check** (separate MLflow experiment `delhi_phase1_lme_gapfill_robustness`, per explicit instruction): re-runs spatial LOSO only (not random CV) on the subset excluding `confidence_bucket == 'low'` rows (13,494 -> 12,282 rows).
- Pooled: R2 = 0.429, within-R2 = **0.189** (Kawano-style; was -0.443 under the old formula), RMSE = 0.632, MAE = 0.442 -- essentially unchanged from the full-data spatial LOSO run (R2 = 0.423, within-R2 = 0.211, RMSE = 0.638). Confirms results aren't driven by gap-filled (imputed) AOD days.
- Per-fold detail in `reports/lme/gapfill_robustness/cv_spatial_loso_folds.csv`.

**Within-R2 definition -- switched 2026-09-09 to Kawano et al.'s literal formula** (`Corr^2[y_hat_it - y_hat_i, y_it - y_bar_i]`), replacing an earlier shared-demeaning version. The original implementation de-meaned both the true and predicted values by the SAME station x season group's *true* mean, then took classic R2 of the residuals. Kawano's formula de-means the true series by its own group's true mean and the predicted series by its own group's *predicted* mean -- independently -- then takes the squared Pearson correlation of the two demeaned series. The predicted-side group mean is computed once from the full pooled out-of-fold prediction set (same instability-avoidance reasoning as the true-side group means, see "Bug caught and fixed" below), never per-fold. Because subtracting the same constant from both series leaves their difference (and so SS_res) unchanged, the shared-demeaning version can never cancel a model's constant per-station bias -- it only shrinks the denominator, which is what made it swing so negative on stations with a level-calibration problem. Independent demeaning cancels that constant offset before scoring, so it isolates whether the model tracks the day-to-day/seasonal *anomaly pattern*, regardless of whether it also gets the station's absolute level right. On our own spatial-LOSO output the two formulas swing from -0.335 (shared demeaning) to +0.211 (Kawano) on identical predictions -- see the DAY9 daily log follow-up entry for the full comparison.

**Bug caught and fixed before finalizing**: the first implementation computed each fold's station x season group means from that fold's own test rows. This is fine for spatial LOSO (a fold's test set is one whole station, ~80 rows per season -- large enough groups) but breaks badly for the 42-fold random CV, where a single fold's ~321 test rows spread across up to 168 possible station x season groups leave most groups with 1-2 rows. A singleton group's "mean" is just its own value, so it contributes exactly 0 to SS_tot while its prediction error still inflates SS_res -- wildly unstable, mostly-negative per-fold within-R2 (observed values down to -0.91 before the fix, vs. a sensible 0.05-0.27 range after). Fixed by computing group means once from the full dataset, used consistently for both per-fold and pooled reporting. The pooled/aggregated numbers were unaffected either way (pooling every fold's held-out rows reconstructs the full dataset's group memberships regardless of which level the means were computed at) -- only the per-fold table was actually wrong pre-fix. Two stale MLflow runs logged before this fix (with the broken per-fold numbers) were deleted from the tracking store; only the corrected runs remain.

### 3. MLflow logging

Tracking URI: `MLFLOW_TRACKING_URI=file:./models/mlflow_tracking` (added to `.env`). Hit a real gotcha: MLflow 3.16 (the version that installed) refuses the plain filesystem tracking backend by default ("in maintenance mode", pushes toward a database backend) unless `MLFLOW_ALLOW_FILE_STORE=true` is set -- set via `os.environ.setdefault(...)` in both scripts so the repo's file-based tracking convention keeps working without Muhammed needing to set it manually.

- Experiment `delhi_phase1_lme`: run `full_data_reml_fit` (fixed-effect coefficients + SEs as MLflow metrics, plus `reports/lme/primary_log_target/lme_fixed_effects.csv` logged as an artifact; random-intercept variance, residual variance, logLik, AIC, BIC as metrics; the fitted model logged both via `mlflow.statsmodels.log_model()` and as a plain pickle artifact, then registered to the MLflow Model Registry as `delhi_phase1_lme` (currently version 3 -- versions 1 and 2 were superseded by path-only/flag-only re-runs during the reports/lme/ reorg and the diagnostic-variant fit reports work below, all three byte-identical models) via `mlflow.register_model()` -- confirmed the registry works on this repo's plain file-store tracking URI in the installed MLflow 3.16, no database backend needed); runs `spatial_loso_cv` and `random_cv` (per-fold R2/within-R2/RMSE/MAE logged as step-indexed metrics so MLflow renders them as a per-fold chart, plus pooled/aggregated metrics and the fold-level CSV as an artifact).
- Experiment `delhi_phase1_lme_gapfill_robustness`: run `spatial_loso_cv_excl_low_confidence`, same metric/artifact shape as the primary spatial-LOSO run.

### 4. Pipeline wiring

- `params.yaml`: `modeling.lme_validation` (`buffer_km: 2.0`, `n_random_folds: 42`, `random_seed: 42`).
- `dvc.yaml`: LME stages grew from 3 (`fit_lme_model`, `validate_lme_cv`, `validate_lme_cv_gapfill_robustness`) to 8 over the session -- added `winsorize_aod_dataset`, `validate_lme_cv_aod_winsorized`, `validate_lme_cv_excl_outlier_stations`, then `fit_lme_model_gapfill_robustness`, `fit_lme_model_aod_winsorized`, `fit_lme_model_excl_outlier_stations` (see Completed section 7). Ran each via `dvc repro -s <stage>`.
- No shared utils module introduced -- `FIXED_EFFECTS`/`build_formula()`/etc. are duplicated between `01_fit_lme_model.py` and `02_validate_lme_cv.py`, matching the repo's existing no-shared-module convention (every other multi-script module duplicates its own constants rather than importing from a sibling). Flagged here as a "must keep in sync" risk if the equation changes.

### 5. AOD winsorizing + outlier-station exclusion comparison

Built the two follow-up diagnostics from the 5598/6934 investigation.

**`03_winsorize_aod.py`**: caps `aod_055` at +/-3 z-score (already standardized in `lme_ready_dataset.csv`), then recomputes `aod_x_monsoon`/`aod_x_post_monsoon`/`aod_x_winter` from the capped value (verified these are exactly `aod_055 * season_<name>` -- leaving them uncapped would let the extremes leak back in through the interaction terms). Writes a separate DVC-tracked `data/processed/modeling_datasets/lme_ready_dataset_aod_winsorized.csv`, not an in-script transform, so `01`/`02` need zero code changes -- just point `--input` at the new file. 174 of 13,494 rows (1.29%) capped. New `winsorize_aod_dataset` stage, `modeling.lme_winsorize.aod_zscore_cap: 3.0`.

**`--exclude_stations`/`--run_tag` flags on `02_validate_lme_cv.py`**: `--exclude_stations` drops a comma-separated `location_id` list from the validation dataset entirely; `--run_tag` disambiguates MLflow run names so diagnostic variants coexist in the `delhi_phase1_lme` experiment without overwriting the primary run's run names (output filenames no longer need a suffix for this -- see the reports/lme/ reorg in Completed section 6). New stages `validate_lme_cv_aod_winsorized` and `validate_lme_cv_excl_outlier_stations` (`modeling.lme_validation_excl_outlier_stations.exclude_stations: "5598,6934"`).

**Results** (spatial LOSO, pooled):

| Variant | R2 | within-R2 (Kawano) | RMSE | MAE |
|---|---|---|---|---|
| Primary (all 42 stations, raw AOD) | 0.423 | 0.211 | 0.638 | 0.453 |
| AOD winsorized (+/-3 z) | 0.431 | 0.228 | 0.634 | 0.448 |
| Excluding stations 5598, 6934 | 0.589 | 0.211 | 0.537 | 0.397 |

Winsorizing gives a modest, uniform lift (trims noise across the whole dataset). Excluding the 2 outlier stations gives a much larger R2 jump (0.423 -> 0.589) while within-R2 is essentially unchanged (0.211 -> 0.211) -- confirms these two stations are specifically a *level*-calibration problem (land-use extrapolation, per the earlier investigation), not a day-to-day anomaly-tracking problem, and that AOD winsorizing does not address it. Random-CV moves by similarly small amounts in both variants (leakage mechanism itself is untouched by either change).

### 6. Reorganized reports/lme/ into per-run subfolders

`reports/lme/` had grown to 20+ files in one flat directory once the aod-winsorized and excl-outlier-stations variants were added (suffixed filenames like `cv_spatial_loso_aod_winsorized_folds.csv`). Reorganized into one subfolder per run -- `reports/lme/primary_log_target/` (folding in the old `fit/` outputs), `gapfill_robustness/`, `aod_winsorized/`, `excl_outlier_stations/` (the `cv_` prefix on the latter 3 was itself dropped in a later pass, see Completed section 7) -- each with plainly-named files (`cv_spatial_loso_folds.csv`, `cv_random_aggregated.json`, `station_buffer_exclusions.csv`, etc.), since the folder is now the disambiguator instead of a filename suffix.

Simplified `02_validate_lme_cv.py`: dropped the filename-suffix logic entirely (`--output_dir` already gives every variant its own directory), kept `--run_tag` only for MLflow run-name disambiguation (still needed -- all non-gapfill variants share the `delhi_phase1_lme` experiment, so run names still need to differ even though filenames no longer do). `dvc.yaml`'s `outs:`/`cmd:` updated for all 5 lme stages to the new paths.

Deleted the old flat files and re-ran all 5 stages fresh. Confirmed every number is bit-for-bit identical to before the reorg (checked R2/within-R2/RMSE across primary, gapfill, aod-winsorized, and excl-outlier-stations) -- purely a file-organization change, no result changed. Cleaned up the resulting MLflow duplicates: 8 stale run duplicates (one per run name, from re-running with a changed script hash) and 1 orphaned model registry entry (`delhi_phase1_lme` version 1 -- its underlying run was the stale duplicate that got deleted). The registered model was version 2, byte-identical to version 1, at that point -- superseded again by version 3 during the fit-reports work in Completed section 7 (same byte-identical primary fit, flag-only script change).

### 7. Fit reports for the 3 diagnostic variants

Until this point `01_fit_lme_model.py` had only ever been run on the primary/raw dataset -- the gapfill-robustness, AOD-winsorized, and excl-outlier-stations variants had CV validation metrics (section 2/5) but no coefficient table / model summary / AIC-BIC of their own. Added `--exclude_low_confidence`/`--exclude_stations`/`--run_tag` flags to `01_fit_lme_model.py`, mirroring `02_validate_lme_cv.py`'s filtering + MLflow-experiment-branching exactly. Diagnostic-variant fits log to MLflow but are NOT registered to the Model Registry -- only the primary fit is, so the registry's version history stays unambiguous about which version is the production-representative one.

Folded `reports/lme/fit/` into `reports/lme/primary_log_target/` and dropped the `cv_` prefix from the other 3 variant folders (`gapfill_robustness/`, `aod_winsorized/`, `excl_outlier_stations/`), since each folder now holds both its fit outputs (`lme_fixed_effects.csv`, `lme_model_summary.txt`) and its CV outputs together -- one folder per scenario. `models/lme/` stays flat with suffixed filenames (`lme_full_model_gapfill_robustness.pkl`, `lme_full_model_aod_winsorized.pkl`, `lme_full_model_excl_outlier_stations.pkl`), too few files to need subfolders. Added 3 new dvc.yaml stages and re-ran all 8 lme stages fresh; every CV metric confirmed bit-for-bit identical to before the rename.

**Fit results:**

| Variant | n rows | n stations | AIC | BIC | logLik |
|---|---|---|---|---|---|
| Primary (unchanged) | 13,494 | 42 | 20320.9 | 20516.1 | -10134.4 |
| Gap-fill robustness (excl. low confidence) | 12,282 | 42 | 17716.5 | 17909.3 | -8832.3 |
| AOD winsorized (+/-3z) | 13,494 | 42 | 20043.1 | 20238.3 | -9995.5 |
| Excl. outlier stations (5598, 6934) | 12,845 | 40 | 19317.8 | 19511.7 | -9632.9 |

The excl-outlier-stations fit shows a real coefficient shift, not just a metric change: `aod_x_post_monsoon`/`aod_x_winter` drop from clearly significant (coef ~0.06/~0.10, p<0.001 in the primary/winsorized fits) to essentially zero and non-significant (coef 0.004/0.005, p=0.82/0.75), and the land-use coefficient block becomes much larger and less precise (e.g. `tree_cover_pct`: 0.09 -> -1.48, SE: ~0.23 -> ~1.04; `shrubland_pct` becomes newly significant at -0.511, p=0.048). Consistent with the land-use-extrapolation diagnosis (section 2): those 2 stations were doing a lot of the work pinning down the land-use coefficients, so removing them destabilizes several fixed-effect estimates even as it improves LOSO R2. Worth flagging in any writeup that cites the excl-outlier-stations numbers alongside the primary ones.

### 8. Random-slope-AOD equation variant (comparison, not adopted)

Tried the equation variant from the project blueprint / "Ideas" section: `(b0+u0i) + (b1+u1i)*AOD` -- a per-station random slope for `aod_055` alongside the existing random intercept, via `re_formula="~ aod_055"` (statsmodels default: correlated 2x2 random-effects covariance). Added `--random_slope_aod {true,false}` to both `01_fit_lme_model.py` and `02_validate_lme_cv.py`, mirroring the existing flag pattern. `compute_aic_bic()`/`build_coef_table()` in the fit script already generalized cleanly to a 2x2 `cov_re` (verified empirically before writing production code, no changes needed); added `variance_component_rows()` to report every `cov_re` entry generically instead of hardcoding "the one random-intercept variance". `predict_with_random_intercept()` in the validation script renamed to `predict_with_random_effects()` and extended: with a random slope, a station's own slope-BLUP multiplies that row's own `aod_055` value (each row has a different AOD reading, unlike the intercept). Spatial-LOSO prediction stays fixed-effects-only regardless -- a held-out station gets no random-effect BLUP of any kind under either equation, so only random-CV's prediction path can actually use the new slope term. Not registered to the Model Registry (a different equation spec, not a diagnostic data variant -- see `01_fit_lme_model.py`'s `is_primary` gate).

**Fit (full data, 42 stations)**: the random-slope model wins on AIC/BIC despite 2 extra parameters -- AIC 20279.9 vs primary's 20320.9, BIC 20490.2 vs 20516.1, logLik -10111.9 vs -10134.4. Throws `ConvergenceWarning: The MLE may be on the boundary of the parameter space` (still `result.converged == True` -- expected with only 42 groups estimating a full 2x2 covariance, a known risk for random-slope models with limited groups, same caution as `LME_Architecture_M0_A_B_C_SUMMARY.md` flags for the analogous airshed case). Random AOD-slope variance = 0.00184, intercept-slope covariance = -0.00224 (correlation ~ -0.34, not itself degenerate).

**Spatial-LOSO CV (the number that actually matters for generalization)**: R2 = 0.412 vs primary's 0.423 (slightly worse), within-R2 (Kawano) = 0.211, unchanged. The two known outlier stations are unaffected or slightly worse: 5598 R2 -3.428 -> -3.299, 6934 R2 -1.444 -> -1.864. **Confirms the reasoning from the "Ideas" section empirically**: a held-out station gets no random-effect BLUP of any kind under spatial LOSO, so richer random effects can't fix the land-use-extrapolation problem driving those two outliers -- the better full-data AIC/BIC reflects genuine within-sample fit (stations do differ somewhat in AOD sensitivity), but that flexibility doesn't transfer to an unseen station. Random-CV moves by a similarly small, non-systematic amount (R2 0.634 -> 0.635, within-R2 0.206 -> 0.204) -- as expected, since random-CV's "memorized per-station" prediction path is the one place a random slope structurally *could* help, and it barely moves.

**Verdict**: kept as a documented comparison, not adopted as the primary model. The primary random-intercept-only equation and its Model Registry entry are unchanged.

### 9. Duan's smearing bias correction (back-transform to ug/m3)

Implemented in `02_validate_lme_cv.py`: `compute_smearing_factor()` computes Duan's (1983) nonparametric smearing estimator `S = mean(exp(train_residuals))` from each fold's own TRAINING residuals (never the held-out rows, to avoid leaking test information into the correction), matched to the same prediction type used on that fold's test rows -- fixed-effects-only residuals for spatial LOSO (`predict_fixed_effects`), fixed+random-effects residuals for random CV (`predict_with_random_effects`), since a mismatched residual type would bias the correction. `duan_backtransform_metrics()` applies it as `y_raw_corrected = exp(y_log_pred) * S`. Both `rmse_ugm3_naive`/`mae_ugm3_naive` (old, uncorrected) and `rmse_ugm3_duan`/`mae_ugm3_duan` (new) are reported side by side -- in every per-fold row, the pooled aggregated JSON, and MLflow (`fold_smearing_factor`, `rmse_ugm3_duan_pooled`, `mae_ugm3_duan_pooled`, `smearing_factor_mean/min/max`) -- so the size of the correction stays visible rather than silently replacing the old number. Re-ran all 5 CV stages (script change touches every stage's dep hash).

Smearing factor is consistently ~1.12-1.13 across every variant (spatial LOSO and random CV alike), meaning the naive `exp()` back-transform was underpredicting raw-scale PM2.5 by roughly 12-13% on average, as expected from Jensen's inequality.

| | Spatial LOSO naive RMSE | Spatial LOSO Duan RMSE | Random CV naive RMSE | Random CV Duan RMSE | Smearing factor (mean) |
|---|---|---|---|---|---|
| Primary | 77.79 | **84.29** (worse) | 51.83 | **51.20** (better) | 1.132 |
| Gap-fill robustness | 79.97 | 86.82 (worse) | n/a (spatial-LOSO only) | n/a | 1.122 |
| AOD winsorized | 76.54 | 83.04 (worse) | -- | -- | 1.129 |
| Excl. outlier stations | 53.54 | 53.05 (better) | 51.93 | 51.29 (better) | 1.129 |
| Random-slope-AOD | 76.50 | 82.70 (worse) | 52.21 | 51.73 (better) | 1.132 |

**Not a uniform improvement -- this is the noteworthy finding, not a formality.** Duan's correction removes systematic bias in the back-transformed *mean* (that's what it's designed to do), but that doesn't mechanically improve RMSE/MAE on a specific held-out set. On random CV, where held-out rows come from the same stations the smearing factor was estimated on, the correction reliably helps (RMSE drops in every variant). On spatial LOSO, where held-out stations are genuinely different (that's the point of the test), correcting the mean bias upward moves predictions further from the true values more often than not -- RMSE gets *worse* in 4 of 5 variants (all except excl-outlier-stations, where removing the two land-use-extrapolation stations apparently also removes whatever was making the naive back-transform look artificially good there). MAE shows the same pattern. The Duan-corrected number is the statistically correct one to report (it's the textbook bias correction, and RMSE getting worse doesn't make it wrong -- it makes the naive number optimistic in a way that happens to look better); report both naive and Duan-corrected in any writeup for transparency, and don't expect the corrected one to look better on spatial LOSO -- it doesn't, and that's itself informative about the outlier-station extrapolation problem (section 2).

**Verdict**: implemented, re-run for all 5 variants, both naive and Duan-corrected numbers now reported side by side going forward. Resolves the pending item.

### 10. LandUse fixed-effect grouping -- decided (documentation only, no model/code change)

Decided 2026-09-16: the fitted model is unchanged -- the 12 non-AOD/non-season/non-met columns (`ndvi_mean`, 6 WorldCover fractions, `elevation_m`, `slope_deg`, `road_density_km_per_km2`, `industrial_landuse_fraction`, `dist_to_nearest_powerplant_km`) are already estimated as 12 separate fixed-effect coefficients in the code (`LANDUSE_COLS` in both scripts) -- "LandUse" has only ever been a documentation label, not a modeling choice. For any writeup, present the equation in both forms rather than picking one:

**Simplified/grouped form** (compact, matches the current `claude/PROJECT_CONTEXT_pm25-phase1-delhi.md` framing):
```
PM2.5 = b0 + (b1 + b_int*Season)*AOD + b_met*Met + b_lu*LandUse + b_s*Season + u_i + eps
```

**Detailed form** (spells out the 4 sub-groups inside "LandUse", each with its own coefficient block):
```
PM2.5 = b0 + (b1 + b_int*Season)*AOD + b_met*Met
      + b_wc*WorldCover + b_ndvi*NDVI + b_terrain*Terrain + b_osm*OSM
      + b_s*Season + u_i + eps
```
where `WorldCover` = 6 land-cover fractions (tree/shrubland/grassland/cropland/built-up/bare-sparse-veg), `NDVI` = `ndvi_mean` alone, `Terrain` = `elevation_m` + `slope_deg` (SRTM), `OSM` = road density + industrial land-use fraction + distance to nearest power plant.

Use the grouped form when equation compactness matters (e.g. a methods-section overview); use the detailed form whenever a reader needs to know exactly what's inside "LandUse", or that WorldCover/NDVI/terrain/OSM are estimated as genuinely separate covariate families, not one combined coefficient.

### 11. Headline spatial-LOSO number for any writeup -- decided: report both

Decided 2026-09-16: report both the full 42-station primary run (R2=0.423, the honest worst-case, includes the 2 known land-use-extrapolation outlier stations) and the excl-outlier-stations run (R2=0.589, see Completed section 5) side by side, each clearly labeled -- neither replaces the other. The full run is the number that should headline any abstract/summary (it's what the model actually does across all 42 stations); the excl-outlier-stations run is the companion sensitivity number showing how much those 2 stations specifically drag the pooled metric down, with the mechanism already documented (section 2: genuine land-use-covariate extrapolation, not a data-quality problem).

### 12. Raw ug/m3 metrics + a raw-target LME variant (added 2026-09-17, for the LightGBM head-to-head)

Added because the LightGBM comparison is reported in physical units. Two separate pieces.

**(a) Raw-scale metrics for the existing log-fit model.** `02_validate_lme_cv.py` now reports `r2_ugm3` / `within_r2_ugm3` / `rmse_ugm3` / `mae_ugm3` alongside the log-scale block, plus `r2_ugm3_naive` and `r2_ugm3_duan` next to the existing naive/Duan RMSE/MAE pair. Needed because **R2 is not invariant under a nonlinear transform** -- the log-scale R2 = 0.423 and a raw-scale R2 are different quantities with different denominators, so quoting 0.423 against LightGBM's raw 0.803 is not a valid comparison. Only `within_r2_ugm3_duan` is computed (no naive counterpart): Kawano's within-R2 is a squared Pearson correlation and therefore invariant to multiplying predictions by a positive constant, which is exactly what Duan's correction does, so the two flavours differ only via the small between-fold variation in the smearing factor.

**The log-fit model's raw-scale spatial-LOSO R2 is negative:**

| Variant (spatial LOSO) | R2 log | R2 ug/m3 naive | R2 ug/m3 Duan |
|---|---|---|---|
| Primary | 0.423 | -0.044 | **-0.226** |
| Gap-fill robustness | 0.429 | -0.122 | -0.323 |
| AOD winsorized | 0.431 | -0.011 | -0.190 |
| Excl. outlier stations | 0.589 | 0.506 | **0.515** |
| Random-slope-AOD | 0.412 | -0.010 | -0.180 |

Not a bug: spatial-LOSO RMSE of 84.29 ug/m3 already exceeds the data's own sd of 76.12, so `R2 = 1 - RMSE^2/Var` is forced negative. In physical units the log-fit model's out-of-site predictions are worse than predicting the citywide mean. Raw-scale R2 is dominated by large absolute errors, and station 5598's 422.6 ug/m3 RMSE dominates everything -- which is why `excl_outlier_stations` flips to +0.515. Both the 0.423 and the -0.226 are true statements about the same model; the log scale flatters it because logging compresses exactly the errors that hurt most. Random CV is unaffected by this (raw R2 = 0.548) because it has no extrapolating held-out stations.

**(b) A raw-target LME variant** (`reports/lme/primary_raw_target/`, `models/lme/lme_full_model_raw_target.pkl`). Motivation is fairness, not flattery: scoring a log-fit model on raw compares LightGBM (fit raw, scored raw) against an LME handicapped by a back-transform artifact rather than by any real modelling deficiency. New `--target_transform {log,raw}` flag on both LME scripts; on `raw` the validation script skips the back-transform and Duan smearing entirely (mandatory -- `np.exp()` on a raw PM2.5 value overflows). New `prepare_lme_dataset_raw_target` stage via the prep script's already-supported `--target_transform raw`.

**Result -- refitting on raw moves spatial-LOSO R2 from -0.226 to +0.437:**

| Spatial LOSO, raw ug/m3 | log-fit (Duan back-transformed) | raw-fit |
|---|---|---|
| R2 | -0.226 | **0.437** |
| within-R2 (Kawano) | 0.167 | **0.220** |
| RMSE ug/m3 | 84.29 | **57.10** |
| MAE ug/m3 | 42.14 | **38.34** |

Random CV, raw ug/m3: R2 0.548 -> 0.567, RMSE 51.20 -> 50.08.

**But refitting on raw does NOT fix the two outlier stations** -- it only removes the `exp()` amplification on top of them:

| Station | log-fit R2 | raw-fit R2 | log-fit RMSE | raw-fit RMSE |
|---|---|---|---|---|
| 5598 | -3.428 | -2.340 | 422.6 ug/m3 | 138.8 ug/m3 |
| 6934 | -1.444 | -0.966 | 87.3 ug/m3 | 103.3 ug/m3 |

Both stay badly negative. Expected, and it sharpens the DAY9 diagnosis: the failure is **linear extrapolation into unseen land-use covariate space**, which is a property of the model class, not of the target scale. No choice of transform can fix it.

**Which model to use for what** (decided 2026-09-17):
- **Prediction / the LightGBM head-to-head**: the **raw-fit** LME. Both models then fit and scored on ug/m3, so the comparison is like-for-like.
- **Inference / the coefficient table**: the **log-fit** LME, and only that one. Newly measured from the actual fitted models, residual variance across fitted-value quintiles grows **1.77x** on the log fit but **12.79x** on the raw fit -- confirming DAY8 section 9's exploratory ~12.8x estimate almost exactly. That heteroscedasticity invalidates the raw fit's standard errors, p-values and confidence intervals. It does **not** invalidate its point predictions or CV metrics, which is why the split works: the raw fit is a legitimate predictor with an untrustworthy coefficient table, and the log fit supplies the coefficients.
- `logLik`/`AIC`/`BIC` are **not comparable across target transforms** (different response variable, different likelihood scale). The raw fit's AIC of 143927.3 must never be tabled against the log fit's 20320.9. Both scripts now print and write this warning.

The raw-target fit is **not** registered to the Model Registry -- diagnostic/alternate-specification variant, same treatment as the other four.

### 14. reports/lme/ renamed: primary -> primary_log_target, raw_target -> primary_raw_target (2026-09-17)

Once section 12 established that the log fit and the raw fit are co-primary for different purposes -- the log fit is the inference model (valid p-values), the raw fit is the predictive comparator against LightGBM -- the folder name `primary/` became actively misleading, since it silently meant "the log one". Renamed both to say which is which:

| Before | After |
|---|---|
| `reports/lme/primary/` | `reports/lme/primary_log_target/` |
| `reports/lme/raw_target/` | `reports/lme/primary_raw_target/` |

The other four diagnostic folders (`aod_winsorized/`, `excl_outlier_stations/`, `gapfill_robustness/`, `random_slope_aod/`) are unchanged -- they are all variants of the log fit, and nesting them under a `log_target/` parent was considered and rejected as moving six folders to express a distinction only two of them need. The flat layout also stays symmetric with `models/lme/`, which uses flat suffixed filenames.

Done as a pure path change: directories moved with `git mv`, `dvc.yaml` cmd/outs updated for the 4 affected stages, then `dvc commit -f` to re-point `dvc.lock` without re-running anything. **Verified every file byte-for-byte identical before and after** (md5 of all 14 files). No metric changed, no stage re-executed.

### 13. Environment note -- dataset regenerated with 1e-15 float drift

Re-running `prepare_lme_dataset` under the currently-installed numpy 2.5.3 / pandas 3.0.5 produced `lme_ready_dataset.csv` with a **different DVC hash** (`0704a733...` -> `1e472e12...`, size 6,119,824 -> 6,138,496 bytes) despite identical row counts and zero non-numeric differences. Cause is floating-point last-bit noise in the standardization arithmetic: max absolute difference across all columns is **1.8e-15**, i.e. agreement to ~15 significant figures, with the size change coming from marginally longer float reprs.

Harmless numerically -- every log-scale CV metric and every AIC/BIC reproduced its previously documented value to 3+ decimals across all 5 variants. But it does invalidate every downstream DVC stage hash, so "bit-for-bit identical" is no longer an achievable verification standard on this machine; verify to a tolerance instead.

## Data notes & gotchas

- `MixedLMResults.aic`/`.bic` are `NaN` in statsmodels 0.15.0 for this model class -- compute by hand (see Completed section 1).
- MLflow 3.16 needs `MLFLOW_ALLOW_FILE_STORE=true` to use a plain `file:./...` tracking URI -- new gotcha this session, not present in any prior module (this is the first MLflow usage in the repo).
- Within-R2 must be computed against group means taken from the full dataset, not per-fold -- see "Bug caught and fixed" above. Any future validation script that reports a station x season (or similar small-group) metric under k-fold CV should watch for the same singleton-group instability.
- `dvc`, `statsmodels`, `mlflow`, `scikit-learn`, `scipy` all needed installing this session (same non-persistence noted in every prior session's log).
- Two stations (5598, 6934) are severe spatial-LOSO R2 outliers, unrelated to the buffer logic -- resolved: both sit at genuine extremes of the land-use covariate space (outside or at the boundary of the other 41 stations' range), forcing the model to extrapolate its linear land-use coefficients when either is held out. Not a within-R2 problem under the Kawano formula (see Completed section 2) -- still worth addressing for R2/RMSE via winsorizing or robust regression, see Pending.
- 4 of the 42 spatial-LOSO folds (holding out stations 235 Anand Vihar, 5626 DTU, 6359 IHBAS Dilshad Garden, 6938 Vivek Vihar) throw a `ConvergenceWarning` during refit on the reduced training set -- statsmodels' default optimizer fails first, then it auto-retries with lbfgs and that's what's actually returned. **Investigated and resolved (benign)**: `result.converged == True` on all 4 final results; cross-checked each against explicit `method=['cg']` and `method=['bfgs']` refits -- `cg` lands at the exact same log-likelihood and coefficients (matches to 5 decimal places), `bfgs` within a negligible difference (llf off by ~0.16, `aod_055` coef off by ~0.00002). Three independent optimization paths agreeing is good evidence these are genuine optima, not flukes, and the saved fold R2 values for these 4 stations (0.529/0.333/0.472/0.751) are unremarkable -- no relation to the 5598/6934 outlier issue. One real gotcha found along the way, not currently a problem: forcing `method=['lbfgs']` alone from a naive start (rather than as statsmodels' internal fallback step, which starts from wherever the failed first attempt left off) is genuinely unstable on this model -- it collapsed the random-intercept variance to exactly 0 and blew the log-likelihood to infinity in all 4 cases. Worth remembering if this model is ever refit with an explicit single-method override.
- The random-slope-AOD variant (Completed section 8) throws its own, different `ConvergenceWarning` ("The MLE may be on the boundary of the parameter space") on every fit -- expected given only 42 groups estimating a full 2x2 random-effects covariance matrix, not investigated further since the variant wasn't adopted (spatial-LOSO performance didn't improve).

## Pending

- ~~Investigate why stations 5598 and 6934 are severe spatial-LOSO R2 outliers~~ -- done, see Completed section 2. ~~Winsorize AOD + exclude-outlier-stations comparison~~ -- done, see Completed section 5. ~~Each diagnostic variant needs its own fit report (coefficient table + AIC/BIC), not just CV metrics~~ -- done, see Completed section 7. ~~Open question: whether the headline number for any writeup should be the full-42-station run or the excl-outlier-stations run~~ -- **decided 2026-09-16, see Completed section 11: report both**, each clearly labeled.
- ~~Some spatial-LOSO CV folds triggered `ConvergenceWarning` during refit~~ -- investigated and resolved (benign), see "Data notes & gotchas" above.
- ~~Duan's smearing bias correction for back-transforming log-scale predictions to ug/m3~~ -- done, see Completed section 9. Both naive and Duan-corrected ug/m3 metrics are now reported side by side for every CV variant.
- LightGBM comparison model (`scripts/modeling/lightgbm/`, on `lightgbm_ready_dataset.csv`) using the same station-grouped, 2km-buffer spatial CV protocol -- separate future chat, not started.
- ~~Whether the "LandUse" fixed-effect grouping (WorldCover + NDVI + terrain + OSM, since the equation doesn't split them further) should be split more finely~~ -- **decided 2026-09-16, see Completed section 10**: no model change; present both a grouped and a detailed equation form in any writeup.
- Whether the random-slope-AOD comparison (Completed section 8, a legitimate negative result) is worth including in any writeup -- not decided with Muhammed.

## Ideas / under consideration

- ~~If the two outlier stations turn out to be a genuine local-effect issue, a random slope for AOD by station might help~~ -- tried, see Completed section 8. It does NOT help: a held-out station under spatial LOSO gets no random-effect BLUP of any kind (neither intercept nor slope), so this equation change can't fix an unseen-station extrapolation problem -- confirmed empirically, not just reasoned about in advance.
