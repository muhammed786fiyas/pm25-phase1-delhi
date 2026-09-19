# Task Log: Pipeline Reproducibility + Multi-City Portability
_Last updated: 2026-09-19_

## Scope

Two goals, in order:

1. **Make the Delhi pipeline fully reproducible** -- `dvc repro` from a clean state must regenerate every documented result, experiments included. Nothing is deleted or moved out; the broken stages get fixed in place so the `delhi-phase1-complete` tag stays reproducible.
2. **Make it portable** to Mumbai, Chennai and Kolkata with minimal manual intervention.

**Status: Steps 1 and 2 COMPLETE. The Delhi pipeline now reproduces end to end -- `dvc repro` regenerates the station roster from rules and rebuilds every documented result. Steps 3-6 (city config, de-Delhi-ing the thresholds, Mumbai) not started.**

## Completed

### 1. Reproducibility audit (read-only)

74 stages: 52 data-pipeline, 22 modelling/prep. Findings grouped by severity.

**(A) Breaks today, even for Delhi**

- **A1 -- `cpcb_list_stations` cannot run at all.** Two independent failures, verified by executing it:
  - the declared `cmd` passes no `--region`/`--output`, but both are `required=True` in the script;
  - `from cpcb_fetcher import ...` raises `ModuleNotFoundError` -- the file is `1-cpcb_fetcher.py`, and a leading digit plus hyphen makes it un-importable under that name. No `sys.path` shim exists.

  This is the DAG's only root, so `dvc repro` from scratch fails at step one.

- **A2 -- the station-file cycle.** `cpcb_list_stations` declares `outs: data/stations` (`cache: false`), and **35 stages depend on `data/stations/cpcb_stations_delhi_status.csv`**, a file inside that output directory. Re-running the stage would overwrite hand-curated QC work.

  **The cycle is one line** -- `scripts/datasets/cpcb/3-download_pm25.py:172`:
  ```python
  if "status" in df.columns:
      df = df[df["status"] == "KEEP"]
  ```
  Download filters on `status`, but `status` includes `DROP_low_completeness` and `DROP_qc_failed`, which can only be known *after* downloading. DVC cannot see the cycle because the hand-transcription step is not a stage.

  The file's 16 columns have three provenances: base OpenAQ metadata (from the script); `pct_2025_possible` (from `cpcb_completeness_check`, downstream), `nn_km`/`nn_station`/`shares_maiac_pixel_risk` (from `cpcb_find_clusters`, downstream); and `coord_verified`/`status`/`qc_note` (human).

  **All 14 non-KEEP drops are derivable in principle**, and split cleanly by when they become knowable:

  | Knowable | Count | Reasons |
  |---|---|---|
  | Before download | 9 | `DROP_dead_since_2018` (5), `DROP_no_sensors_confirmed` (2), `DROP_no_2025_data` (1), `DROP_duplicate_site_of_5404` (1) |
  | After download | 5 | `DROP_qc_failed` (4), `DROP_low_completeness` (1) |

- **A3 -- `cpcb_qc_check` declares `data/interim/cpcb/pm25_qc_hourly` twice** in `outs`.

**(B) Inputs no stage produces**

`data/raw/osm/{roads,industrial_landuse,power_plants}.geojson` (manual Overpass exports) and `notebooks/01_eda_master_feature_table.ipynb` (a notebook gating `prepare_lme_dataset`). Also absent from a fresh clone: `.env` (gitignored -- `OPENAQ_API_KEY`, `MLFLOW_TRACKING_URI`) and Earth Engine credentials.

**(C) Delhi-calibrated values that break elsewhere**

- **C2 is the most dangerous** because it fails *silently*: `UTM_CRS = "EPSG:32643"` is hardcoded in all three OSM scripts. Correct for Delhi and Mumbai (43N); **wrong for Chennai (44N, 32644) and Kolkata (45N, 32645)**. No error -- just wrong road densities, industrial areas and power-plant distances feeding the model.
- `elevation_min_m: 150` / `elevation_max_m: 300` -- Delhi sits ~200m; the other three are coastal (0-15m), so every station falls outside. Warning-only (`09_qc_srtm.py` has no `raise SystemExit`), but guarantees noise that trains you to ignore QC.
- `rmse_low_threshold: 0.211` / `rmse_high_threshold: 0.292` -- Delhi's own gap-fill RMSE terciles. Must be re-derived per city or `confidence_bucket` is meaningless.
- Hardcoded Delhi station ids: `drop_stations: [10820, 10900, 10825, 6936]`, `exclude_stations: "5598,6934"`, `exclude_stations: "5622,5630,6359,7005"`.
- **Mangroves**: `worldcover_drop_cols` discards `mangroves_pct` as always-zero and `implausible_classes: [70, 95, 100]` flags class 95 as geographically unlikely. True for Delhi; **Mumbai has extensive mangroves and Kolkata borders the Sundarbans**. Would discard a genuinely predictive feature. (QC path is a WARNING, not a hard fail.)
- `CITY_BBOX` has `delhi`, `chennai`, `gurugram`, `india` -- **no `mumbai`, no `kolkata`**.
- Other bounds to re-derive: `temp_min_c: 0`/`temp_max_c: 50`, `blh_max_m: 6000`, `max_plausible_density_km_per_km2: 50`, `edge_effect_distance_km_threshold: 110`, `high_value_limit: 1000`. Three sit behind genuine hard-fail gates (`02_qc_roads`, `05_qc_industrial`, `08_qc_powerplants` all `raise SystemExit`).
- `target_transform: NOT_SET` is a deliberate human gate requiring EDA review per city.

**(D) Structural**: 74 occurrences of "delhi" in `dvc.yaml` alone, plus `cpcb_stations_delhi_status.csv`, `pm25_delhi_*` directories, MLflow names `delhi_phase1_*`. No `city` parameter exists anywhere.

**What is already sound**: the extract -> QC -> compute pattern is consistent throughout; 70 of 74 stages have clean dep/out wiring with no orphans; `params.yaml` already separates knobs from paths; `CITY_BBOX` shows the fetcher was designed multi-city from the start; the station file is referenced via params/CLI, not hardcoded inside scripts.

### 2. Feasibility probes (2026-09-17)

Run before planning, because each answer reshapes the work.

**Probe 1 -- Overpass/Nominatim: REACHABLE.** A live query returned 193 ways in 13.8s; Nominatim and openstreetmap.org both HTTP 200. **The "blocked from both shells" note in the project context doc was true for the cloud sandbox and does NOT apply to this machine.** The three manual OSM exports can become real stages for all four cities. Caveat: 13.8s for a 0.01-degree box means full metro queries will be slow and rate-limited -- those stages need retry/caching discipline. (The `overpass.kumi.systems` mirror timed out; mirror-specific, not a block.)

**Probe 2 -- Earth Engine: HEADLESS, no manual gate.** Cached credentials present (last modified 2026-08-12); `ee.Initialize(project='internship-pm25')` succeeded with no browser; live queries returned 2,925 MODIS granules and an ERA5-Land value of 302.66 K at Delhi. The ~20 GEE stages need no human in the loop. The credential file is machine-local and absent from a fresh clone.

**Probe 3 -- OpenAQ station availability.** Metro-region bounding boxes, matching Delhi's NCR-wide philosophy (decided 2026-09-17 -- Delhi's existing box already captures Noida station 5598 and Loni/Ghaziabad 7005, both outside NCT, so city-proper boxes elsewhere would make counts non-comparable).

| City | bbox | total | has pm25 | alive in 2025 |
|---|---|---|---|---|
| Delhi NCR (control) | existing | 56 | 56 | **49** |
| Mumbai MMR | 72.75-73.20, 18.85-19.50 | 43 | 42 | **41** |
| Kolkata KMA | 88.20-88.55, 22.35-22.85 | 15 | 15 | **13** |
| Chennai CMA | 79.95-80.35, 12.75-13.35 | 11 | 11 | **9** |

Control validates the probe exactly: 56 found vs 56 rows in the repo's status file (an earlier draft of this log said 55 -- corrected after counting), 42 KEEP after QC -- roughly 80% attrition from "alive" to "KEEP". **Widening Chennai from its existing tight box to the metro box changed nothing (11 either way)** -- the CPCB network there is genuinely that sparse, not a bbox artifact.

**Implication -- this is the finding that reshapes the plan.** Projecting Delhi's attrition: Mumbai ~35 KEEP, Kolkata ~11, Chennai ~8. Delhi and Mumbai are comparable; **Chennai and Kolkata are not**, and several Delhi methodology choices do not survive at that n:

- **Block-nested Optuna tuning becomes impossible** -- it needs 6 blocks of 7 stations; with 8-11 stations blocks would hold 1-2 and the inner `GroupKFold(5)` would be degenerate.
- Spatial LOSO becomes 8-11 folds, each holding out 9-12% of the network rather than 2%.
- The 2km buffer exclusion could remove a large fraction of an 8-station network if any sites cluster.
- Random CV "matched 1:1 with the LOSO fold count" has no natural counterpart.

**Small-n fragility, quantified**: in Delhi with 42 stations, a single station (5598) swung pooled raw-scale R2 from +0.515 to -0.226. At 9 stations that fragility is roughly five times worse -- one bad site can dominate the headline. Sparse-city results must be reported with per-fold spread or a confidence interval, never a bare point estimate.

### 3. Step 1 -- root stage made runnable (commit e4c4c88)

`cpcb_list_stations` had three stacked breaks: missing required CLI args; an un-importable `1-cpcb_fetcher.py` (a library, never a stage -- renamed to `cpcb_fetcher.py` rather than adding a `sys.path` shim); and four emoji in print statements that crash on the Windows cp1252 console. The stage also claimed all of `data/stations/`, which DVC deletes before re-running -- fixing the cmd alone would have wiped the curated roster on the first successful repro. Outs narrowed to the one file the stage writes. Duplicate `outs` in `cpcb_qc_check` removed. Verified by hashing all five files in `data/stations/` before and after: only the raw list changed, the curated roster is byte-identical.

After Step 1, `cpcb_stations_delhi_status.csv` is produced by **no** stage -- an explicit external input, like the OSM geojsons. So `dvc repro` works given that file exists, but does not generate it. Step 2 closes that.

The raw list is inherently non-deterministic: `datetime_last` advanced for 47 of 56 stations in one re-run (live sensors). Contained today because nothing depends on it; Step 2's screening stage must therefore write *decisions* derived from `datetime_last`, never the timestamp itself, or every repro cascades into all 35 downstream stages.

### 4. Step 2 -- roster rules prototyped: acceptance test PASSES (2026-09-17)

**Only one station-file column matters to code.** Of the 16 columns, scripts read `location_id`, `name`, `latitude`, `longitude` and `status` -- and all 27 `status` consumers test only `== "KEEP"`. `coord_verified` (blank for 55 of 56 rows), `shares_maiac_pixel_risk`, `nn_km`, `nn_station`, `pct_2025_possible`, `region` and `qc_note` are read by nothing. Every column that needed human judgement is documentation, not input -- so automation only has to get `status` right, and reason labels are free to change.

**Five rules reproduce the hand-curated roster exactly** (prototype read the regenerated raw OpenAQ list plus the downloaded hourly data, wrote nothing into the repo):

| Rule | Stations | Margin |
|---|---|---|
| No datetimes -> `DROP_no_data` | 15, 16 | -- |
| Last record before window start -> `DROP_ended_before_window` | 13, 103, 236, 431, 2503 | ended 2018, window starts 2025 |
| First record after window end -> `DROP_started_after_window` | 3409496 | started 10 days after close |
| Coordinates within 10 m of an eligible station -> `DROP_duplicate_site_of_<id>`, keeping the earliest `datetime_first` | 6356 (of 5404) | next-closest eligible pair ~590 m |
| Daily completeness < 60% -> `DROP_low_completeness` | 301, 6936, 10820, 10825, 10900 | drops <= 44.1%, KEEP >= 72.9% |

Result: 47 eligible == the 47 stations actually downloaded; **42 rule-based KEEP == the 42 hand-curated KEEP, identical sets.**

Design details that matter:
- **Duplicate detection must run after the alive filter.** Dead predecessor ids sit ~260 m from their live replacements (236/6358, 431/6359); checked first, those pairs would threaten live KEEP stations.
- **Tie-break is earliest `datetime_first`** (longest-running instrument), with `location_id` as the final tie-break. "Most sensors" would have wrongly kept 6356. `datetime_first` is stable over time, unlike `datetime_last`.
- **The four `DROP_qc_failed` stations are explained by completeness alone.** Their zero-value and stuck-sensor rates were corroborating, not decisive -- 10820 has only 1.0% zeros and 1.6% stuck hours (cleaner than several KEEP stations) and was always a completeness drop, while KEEP station 5613 sits at 8.4% zeros against 10825's 10.4%, so a zero-rate threshold would be a knife-edge fit. Completeness separates all five with ~16 points of margin either side of the threshold already in `params.yaml`.
- **This makes the hardcoded `cpcb_filter_aggregate.drop_stations: [10820, 10900, 10825, 6936]` redundant** -- the same four stations fall out of a rule that transfers to any city.
- Completeness must be computed on **all** eligible stations, before any drop. Today `8-filter_and_aggregate.py` drops the four hardcoded stations first, so `station_completeness.csv` has only 43 rows and never sees them.

**Reason labels change** (safe, since no code reads them): `DROP_dead_since_2018` -> `DROP_ended_before_window` (the old label hardcodes a year that will not hold elsewhere), `DROP_no_2025_data` -> `DROP_started_after_window`, `DROP_no_sensors_confirmed` -> `DROP_no_data`, and `DROP_qc_failed` -> `DROP_low_completeness`.

**New defect found**: `10-completeness_check.py` hardcodes `COMPLETENESS_THRESHOLD = 60` and the window dates, although `dvc.yaml` lists `cpcb_completeness.threshold` as a param dependency. Changing the param re-runs the stage but changes nothing.

### 5. Step 2 -- the roster is now generated (2026-09-18)

Two new stages, 76 total. `cpcb_stations_delhi_status.csv` is no longer hand-maintained: it is a stage output.

```
cpcb_list_stations     raw OpenAQ list (56)            frozen
cpcb_screen_stations   NEW -> screened.csv, 47 ELIGIBLE   metadata + geometry only
cpcb_download_pm25     downloads ELIGIBLE, not KEEP    frozen
   ... qc, aggregate, completeness (all 47 stations) ...
cpcb_finalize_roster   NEW -> status.csv, 42 KEEP      + hand-written overrides
```

- **`2b-screen_stations.py`** applies the four pre-download rules. Its output deliberately omits `datetime_last`, so the file changes only when a decision changes.
- **`10b-finalize_roster.py`** turns ELIGIBLE into KEEP or `DROP_low_completeness`, then applies `data/stations/cpcb_station_overrides_delhi.csv` -- a hand-written **input**, never a stage output, so a re-run cannot wipe a manual decision. Empty for Delhi: the rules recover all 42.
- `3-download_pm25.py` filters on `--status-value` (default ELIGIBLE) instead of the hardcoded `"KEEP"` -- the one line that made the pipeline circular.
- `8-filter_and_aggregate.py` no longer drops the four hardcoded stations; `params.yaml`'s `cpcb_filter_aggregate.drop_stations` is deleted. Completeness now runs on all 47 stations rather than 43.
- `10-completeness_check.py` reads its threshold and window from `params.yaml` instead of hardcoding them (fixes the defect noted in section 4); `passes_60pct` renamed `passes_threshold`, since the column name should not hardcode the value.
- The curated roster is archived at `data/stations/archive/cpcb_stations_delhi_status_manual.csv` for its `qc_note` reasoning. Nothing reads it.

**Frozen stages.** 12 stages that fetch from OpenAQ or Earth Engine carry `frozen: true`. `dvc repro` rebuilds everything downstream from their saved output rather than re-fetching, because both services can revise historical data -- the snapshot is what makes the results reproducible. Refreshing data or running a new city means `dvc unfreeze <stage>` deliberately. Trade-off, worth remembering: **a frozen stage ignores changes to its inputs**, so it must be unfrozen whenever the roster or coordinates genuinely change.

**Verification -- full `dvc repro`, 32 stages, exit 0, pipeline clean afterwards.**

| Result | Now | Documented |
|---|---|---|
| LightGBM spatial LOSO (R2 / within-R2 / RMSE) | 0.803 / 0.721 / 33.727 | 0.803 / 0.721 / 33.73 |
| LightGBM random CV | 0.882 / 0.775 / 26.071 | 0.882 / 0.775 / 26.07 |
| LME raw-fit spatial LOSO | 0.437 / 0.220 / 57.095 | 0.437 / 0.220 / 57.10 |
| LME log-fit spatial LOSO | 0.423 / 0.211 / 0.638 | 0.423 / 0.211 / 0.638 |
| LME log-fit, raw ug/m3 scale | -0.226 / 0.167 / 84.29 | -0.226 / 0.167 / 84.29 |

Every headline number reproduces, and the acceptance test holds: the generated roster's 42 KEEP stations are the hand-curated 42, with identical coordinates. `pm25_daily_final.csv` -- the ground-truth table every model trains on -- is **byte-identical**. Tuned hyperparameters identical.

Of 71 report files: 27 byte-identical, 29 same numbers with different bytes, 15 flagged as differing. All 15 checked individually:
- 6 x `station_buffer_exclusions.csv` -- **row order only**. The generated roster is sorted by `location_id`, the hand-curated one was not. Sorted, the station set, neighbour counts and excluded ids match exactly. (Worth knowing: a diff on this file looks alarming -- `location_id` "differs by 6930" -- and means nothing.)
- 6 x `lme_model_summary.txt` -- float noise in the 15th significant digit (`1.7677323918005814` -> `...816`).
- The rest -- station 6931's name lost a trailing space upstream at OpenAQ.

The master table changed only in those two ways plus float noise <= 2.3e-12 (new numpy/geopandas versions). That was still enough to make DVC re-run the whole modelling suite, which is what made this a genuine end-to-end test.

**Gotcha found while verifying**: in pandas 3, `astype(str)` does not make two missing values compare equal, so a naive text comparison reported 10,007 "changed" power-plant names that were all missing on both sides. Fill missing values before comparing text columns.

**A second broken stage, missed in the section 1 audit**: `cpcb_download_pm25` was also unrunnable -- its `cmd` passed none of the three required arguments, and its output holds two directories from two manual runs with different date ranges. Fixed with a DVC multi-command `cmd` (one stage, two invocations: 2025-01-01..2025-12-31 and 2026-01-01..2026-03-31, the ranges recorded in the script's own docstring), so the output layout is unchanged. Lesson: I had only tested the DAG root, not the stage after it.

### 6. Closing the fresh-clone gaps (2026-09-19)

Step 2's full repro proved the pipeline logic, but only on this laptop. Checking what a fresh clone would lack found two gaps, both now closed:

- **The three OSM inputs were tracked nowhere.** `data/raw/osm/{roads,industrial_landuse,power_plants}.geojson` (152 MB, 468 KB, 84 KB) were gitignored, had no `.dvc` file, and no stage produces them -- so neither `git push` nor `dvc push` had ever sent them anywhere. A fresh clone would `dvc pull` everything else and then fail at the three OSM extraction stages. Fixed with `dvc add` + `dvc push` (3 files pushed); hashes unchanged, pipeline still clean. They are now snapshots, the same reasoning as the frozen stages. Making them real Overpass download stages remains the better answer for the other cities.
- **`README.md` was empty -- zero lines.** A fresh clone had no setup instructions, and five model scripts stop at start-up without `MLFLOW_TRACKING_URI` in a `.env` file that is not in the repo. README now covers setup, reproduction, frozen stages, refreshing data, the generated station list and the overrides file. Every command in it was tested or is standard DVC -- one written from memory (`mlflow ui --backend-store-uri file:...`) turned out to fail, because MLflow 3 also refuses the file store in the UI without `MLFLOW_ALLOW_FILE_STORE=true`; corrected before commit.

After this, every non-script dependency of every stage is either produced by a stage, tracked by DVC, or tracked by git. (The two git-tracked ones are deliberate hand-written inputs: the overrides file and the EDA notebook.)

**Gap still open -- and it is the important one: the DVC remote is on this laptop.** `.dvc/config` is committed with `url = E:\PROJECTS\MAIN PROJECTS\INTERNSHIP\CODEBASE`, a local folder on the same drive as the repo. `dvc push` succeeds, but the data never leaves this machine, and anyone else cloning gets a remote path that does not exist for them. So Delhi reproduces **on this machine**; reproducing it anywhere else needs a shared remote (Google Drive, S3, network folder) first. README states this plainly rather than giving setup steps that would only work here. The fresh-clone test (clone to a temp folder, install, `.env`, `dvc pull`, `dvc status`) is deferred -- with the current remote it would pass on this machine and still say nothing about anyone else's.

Also noted: Python bytecode in `scripts/` shows both 3.10 (older runs) and 3.13 (this session's full repro), so README states "developed on 3.10, verified end to end on 3.13".

## Key decisions (pre-registered 2026-09-17, before any other city is run)

**1. Publication threshold for the LightGBM model -- relative, not absolute.**

> **Publish LightGBM if it beats the LME on spatial-LOSO R2 (raw ug/m3) in that city. Otherwise publish the LME. Absolute floor: spatial-LOSO R2 > 0.**

Chosen over an absolute number because a relative criterion is pre-registerable with zero knowledge of the city, **self-calibrates to city difficulty** (a sparse hard city sets a low bar for both models, a rich one a high bar), and answers the actual question -- is the extra complexity earning its place? If LightGBM cannot beat a linear model, the simpler interpretable one wins by default.

The floor is not vacuous: Delhi's log-fit LME scored on raw ug/m3 fails it at **-0.226** (worse than predicting the citywide mean).

Recorded explicitly to prevent a moving goalpost. Deciding the threshold after seeing results would make whatever was obtained "acceptable" by construction.

**Utility anchor for writeups (a reporting aid, NOT a gate)**: India's national AQI PM2.5 categories are 30 ug/m3 wide through "Poor" (0-30, 31-60, 61-90, 91-120). Delhi's LightGBM RMSE of 33.7 is about one category width and its MAE of 20.8 is under one -- far more meaningful to a domain reader than an R2.

**Trap to avoid**: do NOT benchmark against literature R2 values. Satellite-PM2.5 papers overwhelmingly report random or sample-based CV, not spatial LOSO. Delhi's own gap shows why -- 0.882 random CV vs 0.803 spatial LOSO for the identical model. Comparing our spatial LOSO against a paper's random CV would make a strong result look weak.

**2. Both LME fits for every city** (decided 2026-09-17). The **log** fit supplies the coefficient table -- it is the only one whose standard errors and p-values are valid (residual variance ratio across fitted-value quintiles: **1.77x log vs 12.79x raw**, measured from the actual fitted models, see `9-LME_Model.md` section 12). The **raw** fit is the fair comparator against LightGBM, both fit and scored in ug/m3. Two LME fits cost seconds each.

**3. Acceptance test for the station-cycle fix**: **the rebuilt roster must regenerate Delhi's exact 42 KEEP stations.** If it produces a different roster, every downstream number changes and Delhi stops reproducing. Sharp and checkable, unlike "looks reasonable".

**4. Delhi keeps every stage.** No stage is deleted or moved to a separate pipeline file. `dvc repro` must regenerate the whole documented result set including the diagnostic variants, so the `delhi-phase1-complete` tag stays reproducible. (An earlier suggestion to split experiments into `experiments/dvc.yaml` was considered and rejected -- keeping everything reproducible is the stronger guarantee.)

**5. Other cities run a reduced set**: raw LME + log LME + LightGBM only. The nine Delhi-only experiment stages are not replicated -- `winsorize_aod_dataset`, `fit`/`validate_lme_cv_aod_winsorized`, `fit`/`validate_lme_cv_excl_outlier_stations`, `fit`/`validate_lme_cv_random_slope_aod`, `validate_lightgbm_cv_excl_weak_aod_coupling`, `validate_lightgbm_cv_excl_weak_aod_from_training`. Delhi was the exploratory pilot whose job was to discover which decisions matter; the other cities inherit those decisions rather than re-deriving them.

Two stages worth keeping for every city despite looking Delhi-specific: **`*_gapfill_robustness`** (answers "are results driven by imputed AOD?" -- gap-fill rates differ per city; Delhi is ~48%) and **`diagnose_lightgbm_aod_coupling`** (predicts which stations will be hard from observed data alone, before any modelling -- with 9 stations in Chennai, knowing upfront that several couple weakly tells you whether the city is viable at all).

**6. Tuning design at small n.** Block-nesting exists because 42 stations make 42 tuning searches expensive. With 8-11 stations it collapses naturally into proper nested LOSO -- for each held-out station, tune on the remaining n-1 via inner `GroupKFold`. That is *more* rigorous than the block approximation and affordable, since there is less data per fold. Small n costs statistical power but buys a cleaner tuning design.

## Plan

**Step 1 -- unbreak what is broken. DONE (commit e4c4c88).** Fix A1 (the import and the missing cmd args) and A3 (duplicate outs). Acceptance: `dvc repro -s cpcb_list_stations` runs.

**Step 2 -- break the station cycle. DONE (2026-09-18, see Completed section 5).** Restructure around *when* information becomes available:

```
list_stations      all candidates in bbox                  (automatic)
      |
screen_stations    metadata + geometry rules -> eligible   (automatic, NEW)
      |
download_pm25      downloads ELIGIBLE, not KEEP            (breaks the cycle)
      |
qc + completeness  unchanged
      |
finalize_roster    eligibility + completeness + QC         (automatic, NEW)
      |            + station_overrides.csv
      v
    status  ->  consumed by the other 35 stages
```

Two things make it work: `station_overrides.csv` becomes a hand-curated **input** (git-tracked, never a DVC output) holding only what cannot be derived -- so human judgement enters as a source rather than as an edit to a generated file; and `status` becomes a genuine stage output that `dvc repro` regenerates instead of clobbering. Acceptance test is decision 3 above.

**Step 3 -- city config.** Add `city:` to `params.yaml` (name, bbox, UTM EPSG, study window); parameterize the 74 hardcoded `delhi` strings in `dvc.yaml`. **Fix C2 (UTM) here** -- it is the silent-failure bug.

**Step 4 -- separate physical constants from city-calibrated values.** Universal (AOD valid range, unit sanity bounds) vs per-city re-derivation (elevation range, gap-fill RMSE terciles, temperature bounds, road density, station id lists, the mangroves assumption).

**Step 5 -- prove it on Mumbai.** 41 alive stations, methodologically closest to Delhi, so it tests portability without confounding it with a small-n statistics problem.

**Step 6 -- decide the Chennai/Kolkata methodology separately**, informed by Mumbai. This is a study-design question, not a code-portability one.

## Data notes & gotchas

- The project context doc's claim that Overpass/Nominatim are blocked is **stale for this machine** -- verified reachable 2026-09-17. It remains true for the cloud/Cowork sandbox.
- `parameters` in the OpenAQ station record is a **comma-separated string**, not a list. Iterating it with `any(... for p in parameters)` walks characters and silently returns zero matches -- hit while writing probe 3.
- Station counts must be filtered by `datetime_last >= study_start` to be meaningful. Kolkata's raw 15 includes stations dead since 2017; only 13 are alive in the 2025 window.
- `earthengine-api`, `pytz` and `requests` needed installing this session (same non-persistence as every prior session).

## Pending

- Everything in the Plan section -- no implementation has started.
- **Whether Chennai and Kolkata get their own models at all, or share a pooled multi-city model to sidestep small n.** A genuine study-design decision, not yet made with Muhammed. "Pooled" covers four meaningfully different designs, expanded 2026-09-17:

  | # | Design | What it does |
  |---|---|---|
  | 1 | **Per-city models** (current plan) | Four separate models, each validated by its own spatial LOSO. Simple; small-n fragility is the whole problem. |
  | 2 | **Fully pooled** | One model on all four cities combined (~96 stations, ~55k rows). Restores block-nested tuning and stabilises estimates, but *assumes* the AOD-PM2.5 relationship transfers -- Delhi is landlocked and dust-dominated, Mumbai and Chennai coastal and humid. If the assumption is wrong, pooling actively hurts. |
  | 3 | **Pooled + city as a feature** | Same data, city identity or city-level covariates added as a predictor, so LightGBM can learn city-specific adjustments rather than assuming one shared relationship. |
  | 4 | **Hierarchical / partial pooling** | LME with nested random effects, station within city -- `(1 \| city/station)` instead of station alone. |

  **Option 4 is the textbook answer to this specific problem.** Sparse cities borrow strength from data-rich ones: Chennai's ~8 station intercepts get shrunk toward the global mean, stabilised by Delhi's 42 and Mumbai's ~35, with the amount of shrinkage estimated from the data so a genuinely different city is still allowed to differ. It is a natural extension of the existing equation rather than a new model family -- today's `u_i` for station gains a city level above it, and the blueprint equation survives with one extra random-effect term. It also directly fixes the fragility concern: under partial pooling a single bad station cannot swing a 9-station pooled R2 the way station 5598 swung Delhi's (+0.515 -> -0.226).

- **Leave-one-city-out validation -- worth adding regardless of which pooling option is chosen.** Train on three cities, predict the fourth. This is arguably the most scientifically interesting question a four-city study can answer: *can a satellite-PM2.5 calibration trained in other cities predict an unseen city?* That is the practical question for extending this to cities with no ground network at all, which is the real-world use case for satellite calibration. It also turns Chennai's sparseness into an asset rather than a liability -- a city with few stations is exactly where a transferred model would be wanted, and 9 stations is still enough ground truth to check it against. The existing spatial-LOSO machinery generalises almost unchanged: swap the grouping variable from station to city.

- **Whether pooling helps is an empirical question that cannot be answered yet** -- it depends on how similar the cities' AOD-PM2.5 relationships are, and data exists for exactly one city. Delhi hints at the scale involved (between-station variance 0.0233 vs residual 0.258 on the log scale, so station-to-station differences are small next to day-to-day variation), but between-*city* differences are likely much larger and are unmeasured until Mumbai exists. **Decided approach: do not commit now.** Once Mumbai is done, test it directly -- fit pooled Delhi+Mumbai and check whether each city's spatial LOSO improves or degrades against its own per-city model. That comparison is itself a reportable result ("does pooling across cities help?") and costs one extra fit. Per-city and pooled are not either/or.

- **Structural consequence, already open in the project context doc**: sibling repos vs. one multi-city repo. **Pooling requires a single pipeline that ingests all four cities.** Committing to four sibling repos first and then wanting to pool would mean merging them afterwards -- so keep the multi-city repo option alive until pooling has been tested on Delhi+Mumbai.
- Whether Chennai's existing tight `CITY_BBOX` entry should be widened to the metro box for consistency with the Delhi philosophy (it makes no difference to the station count -- 11 either way -- but matters for consistency).
- The `notebooks/01_eda_master_feature_table.ipynb` dependency on `prepare_lme_dataset`: a notebook gating a pipeline stage. Not yet decided whether to keep, replace with a script, or drop.

## Ideas / under consideration

- If Overpass proves reliable at metro scale, the three OSM stages could cache their raw responses under `data/raw/osm/<city>/` so re-runs do not re-query -- keeping the pipeline reproducible without hammering a public API.
- A `city` dimension in the MLflow experiment name (`<city>_phase1_lme`) rather than a separate experiment per city, so cross-city comparison is a single query.
