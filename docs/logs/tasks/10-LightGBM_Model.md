# Task Log: LightGBM Model (tune + fit + validation)
_Last updated: 2026-09-17_

## Scope

Build the LightGBM comparison model on `lightgbm_ready_dataset.csv` and evaluate it head-to-head against the completed LME (`docs/logs/tasks/9-LME_Model.md`) using the identical validation protocol: same station file, same 2km buffer-exclusion spatial LOSO, same 42-fold random CV, same Kawano within-R2 formula.

Target is `pm25_daily` **raw** -- no log transform (tree splits are invariant to monotonic transforms of the target, so a log target would change nothing about the learned structure and only complicate reporting). This also means **no Duan's smearing back-transform applies here** -- that was purely a consequence of the LME's log target. LightGBM's RMSE/MAE are natively in ug/m3.

First module built with Claude Code rather than cloud/Cowork Claude.

## Completed

### 1. Hyperparameter tuning (`scripts/modeling/lightgbm/00_tune_lightgbm_hyperparameters.py`)

Block-nested Optuna search, split into its own script/stage rather than folded into the fit script, so `02_validate_lightgbm_cv.py` depends on the tuning output directly (not on the fit) and re-running the cheap fit does not drag the expensive search with it.

**Block partition**: 42 stations into 6 blocks of exactly 7. Blocks are built from the connected components of the 2km buffer graph, shuffled with a fixed seed and assigned largest-first to whichever block is currently smallest. **Buffer clusters are deliberately kept intact inside a block** -- this closes a residual leak the plain "~7 random stations per block" design would have left open: if a held-out station's sub-2km near-duplicate neighbour sat in a *different* block, that neighbour's rows would still have been in the tuning universe that chose the held-out station's hyperparameters. The Delhi roster has 5 clusters (two of 3 stations, three of 2) plus 30 singletons, so an exact 6x7 partition is available. Written to `reports/lightgbm/tuning/station_tuning_blocks.csv`.

**Per-block search**: for each block Bk, the entire inner-CV universe is the other 35 stations. 100 Optuna trials, each scored by 5-fold `GroupKFold` **grouped by `location_id`** -- never a plain random `KFold`. This matters concretely here: 13 of the 23 features are constant within a station, so a random row split would let a trial score well purely by memorizing per-station baselines, and the search would then select exactly the hyperparameters that overfit station identity. Objective is inner-CV RMSE (picked once, used for every search).

**Full-data search**: a 7th search over all 42 stations with no block held out, whose winner is used only by the production fit and by the random-CV arm. Chosen over the cheaper "reuse whichever Hk scored best" shortcut -- there is no held-out test set left to protect once the deliverable model is being built, and the extra search costs ~1/6 on top of the six block searches.

700 trials total, ~48 minutes wall-clock (searches sped up from ~15 min to ~5 min per block as TPE converged). All trials written to `reports/lightgbm/tuning/optuna_trials.csv`; the 7 winning sets plus block assignments to `models/lightgbm/tuned_hyperparameters.json`.

**Search space**: `num_leaves` (15-255, log), `min_child_samples` (5-200, log), `colsample_bytree` (0.4-1.0), `subsample` (0.4-1.0) + `subsample_freq` (1-7), `reg_alpha`/`reg_lambda` (1e-8 to 10, log). `learning_rate` fixed at 0.03 and `n_estimators` never tuned directly -- the boosting-round count comes from early stopping on every fit.

Winning inner-CV RMSE per block ranged 30.86 (block_2) to 35.16 (block_6); full-data search landed at 33.12. The winning hyperparameters differ substantially between blocks (`num_leaves` from 68 to 219, `min_child_samples` from 8 to 36) -- expected and fine under this design, and a useful reminder that a single global search's winner is not a stable quantity here.

### 2. Full-data fit (`scripts/modeling/lightgbm/01_fit_lightgbm_model.py`)

Two-step recipe: early-stop against a validation slice split off **by station** (15% of stations, 6 of 42) to discover the boosting-round count, then discard that probe and refit on **100% of the rows** with that count. The station-wise split matters -- with 13 station-constant features, a random row split would fill the validation slice with near-copies of training rows and early stopping would not trigger until the model had memorized station baselines.

Early stopping selected 166 rounds (cap 3000). In-sample R2 = 0.921, RMSE = 21.37 ug/m3, MAE = 12.61 ug/m3 -- recorded for completeness only, explicitly not a generalization estimate.

Outputs: `models/lightgbm/lightgbm_full_model.txt` (LightGBM's native text format, not a pickle -- the idiomatic choice for a Booster and diffable, unlike the LME's `.pkl`), `reports/lightgbm/primary/lightgbm_feature_importance.csv`, `reports/lightgbm/primary/lightgbm_model_summary.txt`. Registered to the MLflow Model Registry as `delhi_phase1_lightgbm` version 1, in a new `delhi_phase1_lightgbm` experiment (not shared with `delhi_phase1_lme` -- different model families).

**Feature importance (gain-based, this model's analog of the LME's coefficient table):**

| Rank | Feature | Gain % | Splits |
|---|---|---|---|
| 1 | `season` | 32.8 | 169 |
| 2 | `temperature_c` | 16.1 | 2293 |
| 3 | `aod_055` | 10.5 | 1965 |
| 4 | `boundary_layer_height` | 10.1 | 2355 |
| 5 | `relative_humidity` | 9.8 | 2627 |
| 6 | `wind_speed` | 7.7 | 2525 |
| 7 | `confidence_rmse` | 1.9 | 593 |
| 8 | `ndvi_mean` | 1.7 | 1295 |

Season + meteorology + AOD account for **87.0%** of total gain; all 13 static land-use features together account for **8.3%**. This independently echoes the LME's finding that none of the land-use fixed effects reached significance -- two very different model families agreeing that the static site covariates carry little predictive weight relative to season and meteorology. Note `season` earns the top gain from only 169 splits (it is a 4-category feature used near the root), whereas the meteorology columns earn theirs from 2000+ small refinements -- gain and split count tell genuinely different stories, which is why both are reported.

### 3. Validation (`scripts/modeling/lightgbm/02_validate_lightgbm_cv.py`)

Buffer-exclusion logic, station x season group-mean logic and the Kawano within-R2 formula are **duplicated by hand** from `02_validate_lme_cv.py` per the repo's no-shared-utils convention, including the full-dataset-group-means fix (see 9-LME_Model.md "Bug caught and fixed") -- must be kept in sync if the CV protocol ever changes. Buffer exclusion reproduces the LME's result exactly: 12 of 42 stations have at least one neighbour within 2km.

**Spatial LOSO (primary)**: 42 folds, one station held out per fold plus its buffer-excluded neighbours dropped from that fold's training set. Station *i* is predicted using its own block's hyperparameters Hk -- chosen by a search in which *i* never appeared, as training data or as inner-validation data.

**Random CV (comparison)**: 42 row-level folds. All folds use the **full-data** hyperparameter set, not the block-nested ones -- see Key decisions below.

Each fold uses the same two-step recipe as the production fit (early-stop probe on a slice carved from that fold's own training set, then refit on the fold's full training set with that round count). Early-stopping slices are split **by station** for spatial LOSO but **by row** for random CV: random CV needs every station represented in training for the leakage effect being measured to exist at all, so holding out whole stations there would break the comparison it is designed to make. Median selected round count across LOSO folds was 348 (range 62-924).

## Results

### Head-to-head vs the LME -- everything in raw ug/m3

**Decided 2026-09-17: the comparison is reported entirely in physical units.** The LME side gained a raw ug/m3 metric block and a raw-target model variant (see `9-LME_Model.md` Completed section 12); this script's earlier log-scale metric set was removed once that landed. The comparator is the **raw-fit** LME, so both models are fit AND scored on ug/m3 -- scoring a log-fit model on raw would handicap the LME with a back-transform artifact rather than any real modelling deficiency.

**Spatial LOSO** -- the number that measures generalization to an unseen station:

| Metric (raw ug/m3) | LME (raw-fit) | LightGBM |
|---|---|---|
| R2 | 0.437 | **0.803** |
| within-R2 (Kawano) | 0.220 | **0.721** |
| RMSE | 57.10 | **33.73** |
| MAE | 38.34 | **20.76** |

**Random CV** (the leakage comparison arm):

| Metric (raw ug/m3) | LME (raw-fit) | LightGBM |
|---|---|---|
| R2 | 0.567 | **0.882** |
| within-R2 | 0.215 | **0.775** |
| RMSE | 50.08 | **26.07** |
| MAE | 33.07 | **14.90** |

LightGBM beats the LME on every metric in both arms: out-of-site RMSE is **41% lower** (57.10 -> 33.73 ug/m3) and within-R2 more than triples. Per-station, **LightGBM wins on 40 of the 42 spatial-LOSO folds** (median fold R2 0.850 vs 0.529).

For reference, the log-fit LME scored on raw comes in at R2 = **-0.226** (spatial LOSO, Duan-corrected) -- worse than predicting the citywide mean. That number is a property of the back-transform, not of the LME specification, which is exactly why the raw-fit variant is the fair comparator. Full table in `9-LME_Model.md` section 12.

### The spatial-leakage gap is smaller for LightGBM

| (raw ug/m3) | Spatial LOSO R2 | Random CV R2 | Inflation |
|---|---|---|---|
| LME (raw-fit) | 0.437 | 0.567 | **+0.130** |
| LightGBM | 0.803 | 0.882 | **+0.079** |

Both models show the expected inflation, so the finding that random CV overstates performance replicates on a tree model. The LME's gap is ~1.6x larger. Interpretation: the LME's random intercept `u_i` is a pure per-station memorized offset, unavailable for an unseen station, so it contributes nothing under LOSO and everything under random CV. LightGBM has no such term -- it must learn cross-station structure from the covariates either way, so having seen a station before helps it comparatively less.

Note the earlier draft of this log quoted the gap as +0.211 vs +0.081 from log-scale numbers; on the like-for-like raw comparison it is +0.130 vs +0.079. The qualitative finding is unchanged, the magnitude is less dramatic.

### The two LME outlier stations are resolved by the model change -- and NOT by the target scale

The LME's severe spatial-LOSO failures (documented in 9-LME_Model.md as genuine land-use-covariate extrapolation, not a data bug), now with the raw-fit LME as the middle column:

| Station | LME log-fit R2 | LME raw-fit R2 | LightGBM R2 |
|---|---|---|---|
| 5598 (Sector-125, Noida) | **-3.428** (RMSE 422.6) | **-2.340** (RMSE 138.8) | **0.684** (RMSE 42.6) |
| 6934 (Dr. Karni Singh Shooting Range) | **-1.444** (RMSE 87.3) | **-0.966** (RMSE 103.3) | **0.875** (RMSE 26.0) |

Refitting the LME on the raw target improves these substantially -- station 5598's RMSE falls from 422.6 to 138.8 ug/m3, since the `exp()` back-transform no longer amplifies the error on top of it -- but leaves both **still badly negative**. Only the model-class change actually fixes them.

That sharpens the DAY9 diagnosis rather than merely repeating it: the failure is linear extrapolation into land-use covariate space the model never saw, which is a property of the **model class, not of the target transform**, so no choice of scale can address it. Both stations sit outside the other 41 stations' covariate range; a linear model forced to extrapolate its fitted coefficients into unseen territory produces wild predictions, whereas a tree ensemble cannot extrapolate at all and instead clamps to the nearest leaf region it saw in training. Here that inability is a virtue.

Worth noting for any writeup: this substantially weakens the case for the LME's "excl-outlier-stations" sensitivity run (9-LME_Model.md Completed section 5, R2 = 0.589 on 40 stations). That run existed to quantify how much those two stations dragged the LME's pooled metric down. LightGBM handles them without special treatment, so the honest full-42-station LightGBM number needs no companion sensitivity number.

### New weakness: two specific stations (5630, 6359) -- buffer exclusion is NOT the explanation

Only 2 of 42 folds have negative R2: 5630 Shadipur (-0.111, 2 neighbours excluded) and 6359 IHBAS Dilshad Garden (-0.037, 1 neighbour excluded). Both are buffer-affected, and both are stations where the raw-fit LME is positive (+0.065 and +0.127) -- in fact they are the **only 2 of 42 stations where the LME beats LightGBM at all**.

**Correction (2026-09-17)**: an earlier version of this section claimed LightGBM's R2 "degrades systematically with the number of buffer-excluded neighbours", citing means of 0.799 / 0.755 / 0.542 for 0/1/2 excluded. **That gradient is entirely an artifact of these two stations.** Excluding just them, the trend reverses -- buffer-affected stations average slightly *better* than unaffected ones:

| Buffer-excluded neighbours | Stations | Mean R2 (all) | Mean R2 (excl. 5630, 6359) |
|---|---|---|---|
| 0 | 30 | 0.799 | 0.799 |
| 1 | 8 | 0.755 | **0.868** |
| 2 | 4 | 0.542 | **0.760** |

The decisive counter-evidence: three other stations carry the same 2-neighbour buffer burden as 5630 and are unaffected -- 11607 (0.524), 5634 (0.850), 6957 (0.905, among the best folds in the run). Losing two neighbours plainly does not cause failure. The bare association is weak too: 2 of 12 buffer-affected stations go negative vs 0 of 30 unaffected, Fisher's exact **p = 0.077**.

**What survives**: both failures are buffer-affected and no unaffected station fails -- a real but weak association. **What does not survive**: any claim of a systematic buffer effect. The honest position is that something about these two sites specifically makes them hard for a tree model; buffer exclusion may contribute but is clearly not sufficient.

Original proposed mechanism: the same no-extrapolation property that rescues stations 5598/6934 works against LightGBM here -- when a station's nearest analogues are deliberately removed from training, a tree ensemble has nothing close to interpolate from and falls back to a poorly-matched leaf region, whereas the LME's global linear structure still yields a sensible if imprecise answer. **This mechanism is now doubtful as stated**, since the three other 2-exclusion stations are unharmed (see the correction above). If losing near neighbours were the driver, 5634 and 6957 should suffer too, and they are among the best folds in the run.

**What the models' failure sets do show**: against the raw-fit LME, LightGBM wins on **40 of the 42** spatial-LOSO folds, and the only two exceptions are exactly its own two negative folds. So the failures are genuinely disjoint -- the LME fails on land-use extrapolation stations (5598, 6934, both with **zero** buffer exclusions), LightGBM on 5630/6359, and neither fails where the other does. That complementarity is solid and is the reportable finding. What is *not* established is why 5630 and 6359 in particular are hard for a tree model.

Next step accordingly widened: the `buffer_km 0` re-run is still worth doing, but should be paired with a covariate-space look at 5630/6359 of the kind that resolved 5598/6934 on the LME side -- the buffer alone cannot be the explanation.

### Pipeline wiring

- `params.yaml`: `modeling.lightgbm_tuning`, `modeling.lightgbm_fit`, `modeling.lightgbm_validation`.
- `dvc.yaml`: 3 new stages (`tune_lightgbm_hyperparameters`, `fit_lightgbm_model`, `validate_lightgbm_cv`), 68 stages total. Ran each via `python -m dvc repro -s <stage>`.
- `requirements.txt`: added `optuna>=3.6.0` and `scikit-learn>=1.3.0` (the latter was already a de facto dependency of the LME validation script but had never been declared).
- MLflow experiment `delhi_phase1_lightgbm`, 10 runs: `tune_block_1`..`tune_block_6`, `tune_full_data`, `full_data_fit`, `spatial_loso_cv`, `random_cv`.

## Key decisions

- **Tuning split into its own script/stage** rather than folded into `01_fit_lightgbm_model.py` -- cleaner DVC dependency graph (see Completed section 1).
- **Production hyperparameters from a fresh full-data search**, not from reusing the best-scoring block set.
- **Random CV uses the full-data hyperparameter set for all 42 folds.** The kickoff's rule ("station *i* gets its block's Hk") has nothing to key on for row-level folds that contain rows from all 42 stations. Guarding the deliberately-leaky comparison arm against hyperparameter leakage would also be incoherent. **Consequence to state in any writeup**: unlike the LME comparison -- where the two arms shared a model and hyperparameters and *only* the fold-assignment method differed -- the LightGBM arms do not share hyperparameters. The inflation gap reported above is therefore a slightly less clean like-for-like than the LME's. If that cleanliness is ever needed, the fix is a third run: spatial LOSO using the full-data hyperparameters, giving an exactly-matched pair.
- **All 13,595 rows kept**, including the 101 with NaN `aod_055`. See Data notes below.
- **The comparison is reported entirely in raw ug/m3** (revised 2026-09-17). An earlier version of this script reported a parallel log-scale metric set as a bridge to the LME's published log-scale R2, because R2 is not invariant under a nonlinear transform and the raw 0.803 is not the same quantity as the LME's 0.423. That bridge was removed once the LME side gained its own raw ug/m3 block and a raw-fit variant (`9-LME_Model.md` section 12) -- physical units are the more natural reporting scale, and the head-to-head now happens entirely in them. Each model keeps its native metrics; the comparison layer is raw for both.
- **CV folds mirror the production recipe exactly** (early-stop probe, then refit on the fold's full training set). Keeping the early-stopped probe would have validated a model trained on ~85% of the stations the production model gets.
- **Buffer clusters kept intact within tuning blocks** (see Completed section 1).
- **Boosting settings drift guard**: `learning_rate`/`max_boosting_rounds`/`early_stopping_rounds` appear in all three `params.yaml` blocks and must match, since a tuned set is only valid for the learning rate it was searched under. The tuning script records what it ran under in its output json, and both `01_` and `02_` hard-fail on a mismatch. Same pattern guards the feature list.

## Data notes & gotchas

- **`lightgbm_ready_dataset.csv` has 13,595 rows vs `lme_ready_dataset.csv`'s 13,494.** The gap is exactly the 101 rows where `aod_055` is NaN (`aod_gap_filled == 1` but the fill produced no value, so `confidence_rmse` is empty too). statsmodels silently dropped them; LightGBM handles NaN natively and keeps them. **Decided: keep all 13,595 and footnote the 0.75% difference** rather than discarding rows the prep stage deliberately kept. Any head-to-head table should carry that footnote -- the two models are scored on very slightly different row sets.
- **`confidence_rmse` NaN has three-way semantics, not two.** It is missing for the 6,985 observed-AOD rows AND for the 101 failed-gapfill rows, and present for 6,509 successfully gap-filled rows. So `confidence_rmse.isna()` is NOT equivalent to `aod_gap_filled == 0`. Harmless for trees (and the feature earns a respectable 1.9% gain), but do not treat it as a clean gap-fill indicator.
- **The kickoff brief's description of "two near-constant flag columns" was inaccurate** and is corrected here. The two *flag* columns carry real signal and are not near-constant: `aod_gap_filled` is 48.6%/51.4% and `ndvi_gap_filled` is 24%/76%. The genuinely near-constant column is `wetland_herbaceous_pct` (only 4 distinct values across 42 stations, 39 of them 0.0, and last place in feature importance at 0.07% gain); `water_pct` is partly so (19 stations at 0). Conclusion is unchanged -- keep them all, trees are not harmed -- but the reasoning in the brief did not match the data.
- **13 of the 23 features are constant within a station** (8 WorldCover fractions, elevation, slope, road density, industrial fraction, powerplant distance). Jointly they uniquely fingerprint each of the 42 stations. This is what makes random CV leaky for LightGBM at all -- it is the structural analog of the LME's random intercept -- and it is why every grouped split in this module (tuning inner CV, and the spatial-LOSO early-stopping slices) must group by `location_id` rather than splitting rows.
- **`season` must be cast to a pandas Categorical with an explicit fixed category order**, not a per-fold `astype("category")`. A per-fold cast derives the integer codes from whichever seasons that fold happens to contain, which would silently remap codes between train and test. Handled once up front in all 3 scripts via a `SEASON_CATEGORIES` constant.
- **LightGBM 4.7.0 deprecated the sklearn API's `eval_set` argument** in favour of `eval_X`/`eval_y`. Using `eval_set` still works but emits a `LGBMDeprecationWarning` per fit (i.e. thousands of lines during tuning). All three scripts use `eval_X`/`eval_y`.
- **numpy int64 is not JSON-serializable.** Station ids pulled from a pandas column come back as `np.int64` and blow up `json.dump` when written into the block-assignment output (and print as `np.int64(17)` rather than `17`). Cast to plain `int` at the boundary. Hit twice while building this module.
- **MLflow's file store hits the Windows 260-character path limit** when the tracking directory is itself deeply nested -- a metric named `gain_dist_to_nearest_powerplant_km` becomes a file of that name, and the write fails with a bare `FileNotFoundError` that looks nothing like a path-length problem. Not an issue for the repo's own `file:./models/mlflow_tracking` (~180 chars), but it does bite when pointing a test run at a deep temp directory.
- **`MLFLOW_ALLOW_FILE_STORE=true` is still required** on MLflow 3.16.1, same as the LME module -- set via `os.environ.setdefault(...)` in all 3 scripts.
- Environment non-persistence continues: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `lightgbm`, `optuna`, `mlflow`, `python-dotenv` and `dvc` all needed installing this session. `pip` also failed with a permission error on its own wheel cache -- `--no-cache-dir` worked around it. `dvc` had no CLI entry point on PATH, so `python -m dvc` was used throughout (the already-documented intermittent issue).

## Pending

- **Explain why stations 5630 and 6359 fail under LightGBM.** The original buffer-exclusion hypothesis is doubtful: three other stations share 5630's 2-neighbour buffer burden and score 0.524/0.850/0.905, and the apparent buffer gradient reverses once these two are removed (Fisher exact on negative-R2 vs buffer-affected: p = 0.077). Two checks worth running together: (a) re-run spatial LOSO with `buffer_km 0` to see whether they recover at all, (b) a covariate-space comparison of 5630/6359 against the other 40, mirroring the investigation that resolved 5598/6934 on the LME side.
- **Whether to add a matched-hyperparameter spatial-LOSO run** so the leakage-inflation comparison is exactly like-for-like with the LME's (see Key decisions).
- **Whether the LME's excl-outlier-stations sensitivity run should still be reported** now that LightGBM handles those two stations without special treatment (see Results).
- **Whether the LME's headline number should switch to the raw-fit figure.** The log-fit LME's 0.423 (log scale) has been its headline throughout the project; the raw-fit LME's 0.437 (raw ug/m3) is now the like-for-like comparator against LightGBM, and the log-fit model scored on raw is -0.226. Three defensible numbers for the same model family, not yet decided with Muhammed which headlines a writeup.
- **MLflow cleanup outstanding**: this round produced a duplicate run per re-run stage plus `delhi_phase1_lme` registry version 5. Unlike previous rounds v5 is **not** byte-identical to v4 (the dataset regenerated with ~1e-15 float drift, see 9-LME_Model.md section 13), so verify on metrics rather than bytes before deleting either.
- Nothing has been git-committed or tagged for this module -- pending explicit instruction, per repo convention.

## Ideas / under consideration

- **The two models appear to fail on complementary station types** -- the LME on land-use-extrapolation stations (5598, 6934), LightGBM on buffer-excluded stations (5630, 6359). If the buffer hypothesis above holds up, a simple ensemble or a per-station selection rule might beat both. Speculative; would need the hypothesis verified first.
- SHAP values as a richer alternative to gain-based importance, if the writeup needs to say *how* a feature acts rather than just how much it matters. Gain-based importance cannot show direction, so there is currently no LightGBM counterpart to statements like the LME's "wind speed is significant and negative".
- The tuning search fixed `learning_rate` at 0.03 and never explored it. A lower rate with more rounds sometimes buys a little accuracy; untested here, and unlikely to change the LME-vs-LightGBM conclusion given the size of the gap.
