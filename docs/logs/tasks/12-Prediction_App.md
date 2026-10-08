# Task Log: Prediction App + Research Data Service
_Last updated: 2026-10-08_

## Scope

Serve the Delhi Phase 1 LightGBM model locally: a map UI for point estimates plus
download endpoints for research use. `app/` in this repo, local-only deployment.

**Audience reframed 2026-10-08** from "someone checking their air quality" to
"researchers who need historical Delhi PM2.5 for study". This resolves a real tension:
the model physically cannot answer "what is the PM2.5 right now" (ERA5 reanalysis has
~5 day latency, MAIAC AOD 1-3 days, both aggregated to a daily satellite-overpass
window), so a consumer nowcasting app would be fighting its own inputs. A retrospective
research service makes that limitation irrelevant -- researchers want 2025 data, not
today's.

Multi-city replication is **deferred** (decided 2026-10-08): Delhi only for now, possibly
scaling later. The portability work in `11-Pipeline_Reproducibility.md` stays recorded
as pending, not deleted.

## Completed

### API + UI against a stub (2026-10-08) -- before the 10-step plan

Built interface-first so the design could be reacted to before spending GEE time on the
expensive per-cell feature grid.

`app/api.py` (FastAPI) and `app/static/index.html` (Leaflet + OSM tiles, served by
FastAPI rather than as a standalone artifact, since a local API is not reachable from a
sandboxed page).

**Everything is real except the surface between stations.** Real: the trained booster
(166 trees, 23 features), predictions at all 42 stations for any of the 346 dates in the
window, station coordinates, the published spatial-LOSO metrics, and the coverage logic.
Synthetic: `/api/grid` interpolates the 42 station predictions by inverse distance
weighting, flagged `synthetic: true` in every response and in a UI banner.

Endpoints: `/api/model-info`, `/api/stations`, `/api/dates`, `/api/predict`, `/api/grid`.

Verified end to end -- Connaught Place on 2025-12-15 returns 209.1 ug/m3 against 223.8
observed at the nearest station; December grid spans 160-318 ug/m3 (Very Poor/Severe),
June spans 40-58 (Satisfactory).

Run with `python app/api.py`; `--grid_km`, `--max_station_distance_km`, `--port` are the
useful flags.

### Step 1 -- Attribution and citation (2026-10-08)

Added a "Data sources and citation" section to `README.md`: every upstream dataset
credited against the feature it contributes, GEE collection ids pointed at `params.yaml`
for exact product versions, and a usage line ("free to use for research and educational
purposes with attribution").

**Decided: citation is sufficient, a formal licence file is not needed.** Two separate
things were being conflated -- *attribution* is what we owe upstream (Copernicus, ESA,
NASA, OSM; all satisfied by a credit line) and a *licence* is permission we grant
downstream (optional; one sentence covers it for an academic project). Flagged for
confirmation rather than asserted: the CPCB/OpenAQ redistribution terms, and whether the
OSM-derived columns are an ODbL "Produced Work" (attribution only, the likely reading for
computed per-cell statistics) or a "Derivative Database" (share-alike).

### Step 2 -- Out-of-fold predictions saved per row (2026-10-08)

`02_validate_lightgbm_cv.py` scored each spatial-LOSO fold and threw the predictions
away, keeping only aggregates. It now writes
`reports/lightgbm/primary/cv_spatial_loso_oof_predictions.csv` -- one row per
(station, date) with fold id, observed, predicted and residual.

This is the input the next two steps need, and it is worth more than the metrics file:
13,595 honest out-of-sample predictions are a reusable calibration set, and the same
file is what lets the station download ship an out-of-sample estimate instead of an
in-sample fit.

### Steps 3 and 4 -- Conformal intervals replace the distance confidence badge (2026-10-08)

`scripts/modeling/lightgbm/04_calibrate_conformal_intervals.py`, wired as a DVC stage,
output `reports/lightgbm/primary/conformal_calibration.json` plus a per-station
held-out coverage CSV.

Normalized conformal, scaled by the prediction: `interval = prediction * (1 +/- q)`,
with `q` a percentile of the relative absolute residual on out-of-fold rows.
At 90%, **q = 0.5514** -- plus or minus 55.1% of the estimate. Held-out coverage,
measured by calibrating on 41 stations and testing on the 42nd, is **89.9%** against
a 90% target.

The app serves the 90% level (`SERVED_COVERAGE_LEVEL`); 80% and 95% are also
calibrated and sit in the JSON if the level is ever changed.

What this replaced: a four-tier confidence badge keyed to distance from the nearest
station. That was tested against actual per-fold error and **found unsupported** --
r = 0.149, p = 0.33, n = 42. Distance still gates *scope* (`max_station_distance_km`,
and the 10.6 km largest-validated-gap flag) because no correlation across the
0.67-10.6 km the network spans does not license unlimited extrapolation. But it is no
longer presented as a confidence measure anywhere in the API or UI.

Full reasoning, including the three approaches rejected and the one falsified before
being built, is in "Uncertainty: what was tested and what was rejected" below.

### Step 5 -- Station data download (2026-10-08)

Download product 1 of 3. Endpoints in `app/api.py`:

| Endpoint | Returns |
|---|---|
| `/api/download/stations/columns` | selectable features, always-included columns, full data dictionary |
| `/api/download/stations/preview` | row/column/station counts and a size estimate, before committing to a download |
| `/api/download/stations` | the data, `format=zip` (default), `parquet` or `csv` |

**What a row carries.** Identifiers, the **measured** `pm25_observed_ugm3`, the model's
`pm25_predicted_ugm3` with `pm25_lower_90_ugm3` / `pm25_upper_90_ugm3`, the two AOD
provenance columns, and whichever of the 21 features the user selected. 13,595 rows and
32 columns at the full selection -- 0.59 MB as parquet, 5.8 MB as CSV.

**The prediction is out-of-sample, not an in-sample fit.** It is joined from the
spatial-LOSO OOF file, so each station's values come from a model fitted without that
station and without anything within 2 km of it. Verified on the delivered file: RMSE
33.73 ug/m3, exactly the published spatial-LOSO figure, and interval coverage 90.0%
against the 89.9% the calibration predicted. An in-sample fit would have scored far
better and been worth far less.

**The ZIP is the default on purpose.** It bundles `DATA_DICTIONARY.txt` (units,
meaning and source for every column present) and `ATTRIBUTION.txt` (source credits, the
citation line, validated performance, and the marginal-vs-conditional coverage caveat)
alongside both file formats. A CSV that travels without its provenance is the real
failure mode -- the numbers end up in a paper with no route back to their source. Raw
`csv` and `parquet` remain available for anyone scripting against the endpoint.

**Feature selection** is a checkbox grid in the UI, `features=a,b,c` on the API (or
`all` / `none`). Unknown names are ignored and *reported* in the preview response
rather than silently dropped.

**Four columns are not selectable**, and the UI says why. The identifiers and the
observed/predicted/interval columns are the product. `aod_gap_filled` and
`confidence_rmse` are excluded from selection because AOD-PM2.5 coupling nearly halves
on gap-filled rows (0.516 observed vs 0.296 filled, network-wide) -- a researcher who
could drop the provenance would be left with a column whose meaning silently changes
between rows.

## Key decisions

**Scope (confirmed 2026-10-08)**: retrospective point + map, 1 km target resolution,
`app/` inside this repo so DVC can own the prediction grid, local-only deployment.

**AQI colour palette, with one deviation.** Uses the CPCB National AQI band colours for
instant domain recognition, against the alternative of a sequential single-hue ramp.
Severe darkened from the official `#AF2D24` to `#7E1A14`: Very Poor and Severe are the
two commonest bands in Delhi winter (53%/47% of cells on 2025-12-15) yet were only
dE 12.6 apart, which this widens to 24.1. The palette fails a strict adjacent-pair
check (worst pair is now Good/Satisfactory at dE 13.2, below the 15 floor), overridden
deliberately because domain recognition is the point and because every colour is paired
with a text label in the readout, tooltips and a legend table -- so identity never rests
on colour alone, which is also the relief the low-contrast yellow requires.

**Coverage gated on distance to the nearest station, not a bounding box.** The rectangle
was incoherent as a validity test: it refused East Noida at 6.7 km from station 5598
while accepting Gurugram at 10.4 km, and happily answered at the NW corner 23.8 km from
any station. Sampled across the old bbox, 20% of its area was already beyond 10.6 km.
Default cutoff 15 km (~1.4x the largest gap spatial-LOSO validation ever spanned); the
grid is seeded over the bbox plus a margin then filtered to the cutoff, so the coloured
area on the map **is** the area the API will answer for.

**Region for the real grid: within 10.6 km of a station (2,554 cells at 1 km).** Tighter
than the 15 km serving gate, because once the map claims per-cell model output it should
not extend past validated range. Also *cheaper* than the bbox (2,554 vs 2,600 cells)
while being more defensible, and 38% cheaper than a 15 km region whose extra cells would
have to be labelled unvalidated anyway.

## Uncertainty: what was tested and what was rejected

The confidence indicator shipped in the stub is **distance to the nearest station**, and
testing showed it is not supported. Correlations against actual per-fold error, n=42:

| Candidate | vs fold R2 | vs RMSE | vs within-R2 |
|---|---|---|---|
| **AOD-PM2.5 coupling** | **0.624** (p<0.001) | **-0.636** (p<0.001) | **0.729** (p<0.001) |
| Features outside training range | 0.035 (p=0.83) | 0.062 (p=0.70) | -0.118 (p=0.46) |
| Mahalanobis distance to training set | -0.055 (p=0.73) | 0.155 (p=0.33) | -0.130 (p=0.41) |
| Geographic distance to nearest station | 0.029 | 0.149 | -- |

Only coupling predicts error. Notably **feature-space outlierness does not** -- which
surprised, since that was the LME's failure mechanism. The reason is visible per station:
5598 and 6934 are the feature-space outliers (3 and 2 features outside the other 41's
range; Mahalanobis 621 and 30) and LightGBM scores 0.684 and 0.875 on them. Trees clamp,
so extrapolation stops being the failure mode, and the LME's diagnostic does not transfer
to this model. LightGBM's three worst folds have **zero** features out of range.

**REJECTED: stratifying intervals by AOD coupling.** It is the strongest predictor but
cannot be computed for an unmonitored cell (coupling needs PM2.5, which is the unknown),
so it would have to be spatially interpolated from the 42 stations. Tested leave-one-out
by inverse distance weighting: **r = -0.183 (p = 0.25)**, and stratum assignment was
correct for 24/42 stations = 57%, against 67% for always guessing "rest". Of 14 truly
low-coupling stations it caught 2. Coupling is **not spatially smooth** -- it depends on
local emission character (traffic vs industrial vs residential), which changes sharply
over a couple of kilometres in a dense city. The assumption the plan rested on was
falsified before any of it was built.

**ACCEPTED: normalized conformal prediction, scaled by the prediction.** Error is
multiplicative, not additive -- the 90% width divided by the prediction sits near-constant
at 0.55-0.69 across prediction quintiles while the absolute width spans +/-21 to +/-115
(a 5.4x range, against the 1.8x the coupling split would have given). So one calibrated
ratio produces a per-cell width from information always available at prediction time,
with no spatial-smoothness assumption anywhere:

```
interval = prediction x (1 +/- q)        q = 0.457 (80%) / 0.609 (90%) / 0.751 (95%)
```

Not circular: q is calibrated on **held-out** spatial-LOSO residuals, so the prediction
acts as an index into a table of measured past performance rather than the model
assessing itself. Conditioning on `y_hat = f(x)` is conditioning on a summary of the
features, which is ordinary conditional coverage. Conditioning on the *observed* value
would be the cheat.

Why conformal over a Gaussian `+/- 1.645 x RMSE`: the residuals are not remotely normal
(skew 0.57, **kurtosis 12.6**, median error 14.7 against a maximum of 425). A Gaussian
interval is 90% too wide at the 50% level and 40% too narrow at the 99% level -- it
overstates ordinary uncertainty and understates the extremes, which for air quality is
the worse failure. Measured: Gaussian +/-68 delivers 92.8% coverage; conformal +/-57
delivers exactly 90.0%.

**Symmetric, not asymmetric.** Checked for the regression-to-the-mean bias that would
mis-centre a symmetric interval -- signed error by prediction quintile stays within
+/-5 ug/m3 across a 9-454 range with no monotonic drift, so the model is essentially
unbiased. Coverage misses split 4.4% below / 5.6% above against an ideal 5/5; an
asymmetric two-constant version reaches exactly 5.0/5.0, which is not worth a second
calibration constant and a less intuitive interval.

**Season rejected as a second stratifier.** Strongly predictive and fully knowable, but
almost entirely confounded with prediction level (mean predicted value by season: monsoon
39.7, summer 66.2, winter 147.4, post-monsoon 149.7 ug/m3 -- a 3.8x spread). The seasons
barely overlap in prediction range at all; monsoon and summer never reach the top
quintile, winter barely appears below the third. After removing prediction level, season
still explains a statistically real but practically tiny residual (ANOVA F=8.36,
p=1.6e-05, effect +/-2-3 ug/m3 against interval widths of 21-115, i.e. a 2-5%
correction). Not worth a collinear second dimension that splits the residuals into sparse
cells.

## Download products (planned)

Three products with deliberately distinct identities:

| Product | Contents | Audience |
|---|---|---|
| **1. Stations** | **measured** PM2.5 + features, feature selection | training/validation studies -- the only product with ground truth |
| **2. 1 km grid** | features + estimates at *native* resolution, feature selection | fully reproducible: `model(features)` reproduces the prediction exactly |
| **3. Coarser grids** | estimates only, aggregated. **No features** | convenience, small files, areal means |

Sizes, which are what drive the design:

```
product      cells    rows/year      CSV    parquet
stations        42       13,595    5.3 MB    0.4 MB
1 km         2,554      883,684    346 MB     52 MB
2 km           650      224,900     88 MB     13 MB
5 km           110       38,060     15 MB      2 MB
```

A full-year 1 km CSV at 346 MB is too large for a browser download, so **parquet is the
default** (13x smaller for identical data) and date-range plus feature selection are what
keep the service usable -- one month is 30 MB, five columns for a full year is 64 MB.
Row count and size estimate must be shown *before* the download starts.

**Decided: no features at coarse resolution** (Muhammed, 2026-10-08). This removes a real
footgun rather than documenting it. LightGBM is nonlinear, so
`mean(f(x)) != f(mean(x))` -- the same Jensen's-inequality issue that drove the LME's
Duan smearing work. Shipping coarse features *and* coarse predictions would mean a
researcher re-running the model on the supplied features could not reproduce the supplied
prediction. Not shipping them makes that impossible to get wrong, and it is why product 2
at native resolution is the "fully reproducible" one.

**Aggregation is of predictions, never of features**: `mean(f(x_i))` answers "average
PM2.5 over this area", which is what anyone wants; `f(mean(x_i))` answers "PM2.5 at a
hypothetical location with average conditions", which nobody is asking.

**Aggregated intervals genuinely tighten, which was a surprise.** Same-day residual
correlation between stations is near zero at every distance (0.084 within 2 km, 0.032
overall) -- neighbouring cells do **not** share errors through a common AOD pixel or ERA5
cell as expected. So averaging N cells shrinks the interval by
`sqrt((1 + (N-1)*rho) / N)`:

```
grid     N    if independent    with rho=0.032
1 km     1          1.000x            1.000x
2 km     4          0.500x            0.523x
5 km    25          0.200x            0.266x
10 km  100          0.100x            0.204x
```

Coarse products are roughly 4x more precise at 5 km -- a trade of spatial detail for
statistical precision, not a downgrade. **Caveat: rho comes from 16 stations with only 8
pairs under 2 km**, thin exactly where 1 km -> 2 km aggregation depends on it; re-measure
on all 42 folds before publishing aggregated intervals, since underestimating rho makes
coarse intervals look better than they are.

**Critical labelling requirement.** The shrunk interval is valid for the **areal mean**
and not for a point inside the cell -- a point query inherits within-cell spatial
variability that aggregation discarded. Columns must be named
`pm25_areal_mean`, `pm25_areal_mean_lo_90`, `pm25_areal_mean_hi_90`, `n_source_cells`,
and point queries must always serve from the 1 km base, never from an aggregate.
Otherwise someone downloads 5 km because it is 2 MB instead of 52 MB and reads cell
values as point estimates with an interval they are not entitled to.

**Distance to nearest station ships as provenance metadata, not uncertainty.** It does
not predict error (r = 0.149, p = 0.33), so if it sits beside the interval a researcher
will reasonably assume it is an uncertainty proxy. It is genuinely useful for filtering
to cells near monitors, validation study design and exposure sensitivity checks. For
aggregated products ship both `_mean` and `_min`, since a 5 km cell spans 25 different
distances.

### Step 6 (in progress) -- repo restructure: training and prediction split (2026-10-08)

Done before any grid work was built on top, because it was the cheapest moment --
the prediction tree did not exist yet, so every stage added later would have been
another path to move.

`data/` now splits at the top into `training/` and `prediction/`, each with its own
`raw/`, `interim/`, `processed/`. The reason is a real collision rather than tidiness:
training and prediction run the SAME extractor scripts, so both emit a
`srtm_terrain.csv`, a `worldcover_landuse.csv`, a `road_density.csv`. Same basename,
different meaning.

`data/stations/` stays shared at the top level -- both pipelines read it, and the grid
builder should not be reading from a folder called "training".

`reports/` was deliberately NOT split. It is organised by model family, a different
axis, and prediction produces nothing that collides with `cv_spatial_loso_folds.csv`.
Prediction artifacts get a `reports/prediction/` sibling instead. `models/` untouched --
prediction consumes a model, it does not produce one.

Scripts: new `scripts/prediction/` as a sibling to `scripts/modeling/`. The per-source
extractors deliberately stay in `scripts/datasets/` and are invoked with `--stations
<grid>`; six `static_gee` scripts gained an optional `--stations` override (default None
keeps station behaviour byte-identical, verified). They are NOT copied into
`prediction/`, because the model was fitted on features those exact scripts produced and
a second copy would drift.

**Verification, since this touched a pipeline whose reproducibility was hard-won**:
328 path references rewritten across dvc.yaml/params.yaml/scripts; `dvc commit -f`
rebuilt the lock without re-running anything; **249 of 249 data file hashes identical**
before and after (the only 4 changes were scripts edited on purpose); `dvc status` ->
"Data and pipelines are up to date"; all 12 frozen stages intact; the app loads and
serves unchanged. Git recorded all 29 moves as renames.

The 5 km grid artifacts were deleted rather than moved -- 1 km is the target (see below),
and what the 5 km extraction produced was never a shippable product anyway: 1 km-buffer
features sampled every 5 km is a sparse point sample, not a 5 km areal mean.

## Pending

Steps 1-5 are done -- see Completed above. Step 6 is the next one and the first
expensive one.

**Step 6** -- grid feature extraction at 5 km to prove the vertical slice before
committing GEE time. Static features and ERA5/MERRA-2 reuse the existing scripts with a
grid CSV substituted for the station file; **MAIAC AOD and NDVI need new region-export
extractors** because point-sampling 2,554 cells x 365 days is ~932k extractions, 62x the
station pipeline.

**Step 7** -- wire the real grid in: `method` becomes `grid_cell`, `/api/grid` drops
`synthetic: true`, stub banner removed.

**Step 8** -- grid aggregation plus per-level interval calibration.

**Step 9** -- grid download with feature and date selection and a size preview.

**Step 10** -- decide whether 1 km is affordable, with measured throughput rather than
estimates.

**Multi-resolution map view** -- 2,554 rectangles will make Leaflet's default SVG
renderer sluggish (it degrades past roughly a thousand vector features). Use
`preferCanvas: true` and tie resolution to zoom level, so all 2,554 are never rendered at
once.

**Unverified**: CPCB/OpenAQ redistribution terms, and the ODbL Produced Work vs
Derivative Database question for the OSM-derived columns.

## Data notes & gotchas

- `app/api.py` keeps its own copy of the feature contract (`ID_COLS`, `TARGET_COL`,
  `SEASON_CATEGORIES`) per the repo's no-shared-utils convention -- must stay in sync
  with `scripts/modeling/lightgbm/*.py` if the feature set changes.
- `season` must be cast with the explicit fixed category order, as in the training
  scripts. A plain `astype("category")` on a subset would derive different integer codes
  and silently disagree with the booster.
- Predictions near zero need a floor before dividing for the relative residual (used
  5 ug/m3), and the interval's lower bound must clip at 0 -- negative PM2.5 is meaningless.
- The bbox genuinely cuts through NCR: Noida at 77.39 lon is outside it while station
  5598 (Noida Sector-125) sits inside at 77.33. This was the original argument for
  replacing the rectangle with a distance gate.
- **`aod_gap_filled` has three meanings, not two**, and this was found while writing the
  download's data dictionary. The flag means "not a MAIAC retrieval", not "a value was
  imputed": `flag=0` is a MAIAC retrieval (6,985 rows, 51.4%), `flag=1` with a non-null
  `aod_055` is MERRA-2 calibrated (6,509 rows, 47.9%), and `flag=1` with a **null**
  `aod_055` means neither was available (101 rows, 0.7%, all in monsoon). Those 101 rows
  come out of `05_apply_gapfill.py`'s `unfillable` branch, which sets `gap_filled=1`
  while leaving the value `None`. They stay in the modelling dataset because LightGBM
  splits on missing natively. `aod_055` is the only column in the dataset with nulls.
  Anything filtering on the flag alone is wrong -- filter on both columns.
- **`confidence_rmse` is in AOD units, not ug/m3.** It is the RMSE of the per-station
  per-season MAIAC~MERRA-2 regression that produced a fill (range 0.134-0.464), so it
  describes error in the *AOD fill*, not in the PM2.5 estimate. Present on exactly the
  6,509 MERRA-2-calibrated rows; null elsewhere, including the 101 unfillable ones.
- `season` is both a model feature and a download identifier, so it must be excluded from
  the selectable feature list or it lands in the frame twice.
- A long-running dev server cannot be held open from a Claude Code background task (10
  minute ceiling). Run `python app/api.py` directly for a real session.
