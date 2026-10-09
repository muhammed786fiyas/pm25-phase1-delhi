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

### Step 6 -- Grid feature extraction at 1 km (2026-10-08)

Target: all 23 model features on a 2388-cell 1 km grid. Every decision below is
recorded with the alternatives rejected, because most of them were made while
Muhammed was asleep.

**Why 1 km and not the planned 5 km.** The plan called for a 5 km vertical slice
first. Dropped it, because what a 5 km extraction produces is not a 5 km product:
it is 1 km-BUFFER features sampled every 5 km, a sparse point sample rather than
an areal mean, and the download design says coarse products are AGGREGATED from
1 km cells. 1 km is also MAIAC's native resolution and the buffer radius every
static covariate was computed at, so it is the only resolution whose features
mean what the model was fitted on. The slice had already paid for itself by
then: it proved the schema reuse, measured throughput, and surfaced both the OSM
coverage gap and the land-use extrapolation problem. *Alternative rejected:*
extract 5 km, validate, then redo at 1 km -- two GEE runs for information the
first run had already produced.

**Grid definition.** 2600 cells in the study bbox, 2388 kept after dropping
those more than 15 km from a station -- the radius at which the API refuses to
answer, so extracting features for them is wasted GEE time. Written in the
station roster's exact schema (location_id, name, latitude, longitude, status),
which is what lets every existing extractor read it unchanged. Synthetic ids
from 900000 so a grid row can never be mistaken for a CPCB station.

**OSM: a prediction-only extract, from Geofabrik, not Overpass.**
Muhammed's call, and it was the right one: training has NO gap (all 42 stations
sit well inside the hand-drawn exports), so only the grid needs new data.
Extending the training files would have dirtied every downstream training stage
and left one file holding two OSM vintages, to deliver something training never
uses. *Alternative rejected:* gap-only merge into the training geojsons.

Overpass then proved unusable and cost four failed attempts before the switch:

| query | result |
|---|---|
| 1 km box, dense Delhi | OK, 166 ways, 0.12 MB, 3.2 s |
| 5 km box, dense Delhi | OK, 2721 ways, 1.63 MB, 7.8 s |
| 6 km box, dense Delhi | **504 in 12 s** |
| 1/16 of study area, `out geom` | **504 in 7 s** |
| 1/16 of study area, `out ids` | **OK, all 23180 ways** |

A 504 in 6-12 s cannot be the 120 s query budget expiring, and `out ids`
succeeding where `out geom` fails on the SAME tile proves the gateway refuses
large RESPONSES rather than timing out on computation. Rate limiting was ruled
out (status endpoint reported 4 free slots throughout) and no mirror helps
(kumi.systems and private.coffee 500 on every attempt, osm.jp and maps.mail.ru
fail TLS). Tiling to 4 km made each request legal but failures were intermittent
and across 182 tiles the projected runtime CLIMBED 91 -> 158 -> 176 minutes.
A pre-built extract is one download, has no rate limits, and carries a single
published timestamp -- so the vintage is an exact fact rather than "whenever the
182 tiles happened to run". **The decisive evidence (`out ids` succeeding)
arrived early and should have ended the Overpass attempt two iterations sooner.**

**TWO Geofabrik zones, not one.** Northern Zone alone silently excluded Uttar
Pradesh: 112 power plants against the training export's 170 (every one of the 57
east of lon 77.36 missing, including Dadri, which task log 6 names as a key
out-of-Delhi plant) and only 76% of training's roads in the lon 77.30-77.36
strip -- which is Noida and Ghaziabad, inside the study bbox. Both would have
produced plausible wrong numbers with no error anywhere: eastern cells would
simply have reported a further power plant and less road. Caught by comparing
against the training export rather than eyeballing the new files. Northern +
Central covers both bboxes.

**Batching the static extractors, with a hard verification rule.** Per-point
extraction is ~2.5 s/cell, fine for 42 stations and ~100 minutes for 2388. Added
an optional `--batch_size` to the EXISTING scripts rather than writing batched
copies -- a second copy of a feature definition would drift, which is the same
argument that keeps the extractors in scripts/datasets/ at all. Default 0 keeps
the per-point path so station stages are untouched.

Every batched path is verified by running it over the 42 stations and diffing
against the committed output BEFORE it is used on the grid. That rule earned its
place immediately: the first batched SRTM attempt used a point value with a
`first` reducer and was off by up to 7 m, because the station script actually
uses a 1 km buffer with a MEAN reducer. Corrected, it reproduces elevation and
slope to 1e-9 on all 42; WorldCover likewise matches all 12 class counts exactly.
Result: SRTM 2388 cells in 22 s instead of ~100 min.

**QC thresholds are station-calibrated and some are too tight for the grid.**
`predict_qc_srtm` hard-failed on one cell at 301.0 m against a [150, 300]
ceiling. Not an error: every high cell is in the grid's southernmost row, the
Aravalli ridge in south Delhi, and the bounds were set from 42 urban stations
spanning 199.7-271.7 m. Added an optional bound override and set 350 m for the
grid stage, which keeps the check meaningful (1000 m still fails).
*Alternatives rejected:* suppressing the check, or widening the shared params
and loosening it for training too.

**MAIAC: a new batched extractor, adaptively chunked.** The station script
queries ONE POINT at a time over the whole window -- 67.6 s/point measured, so
2388 cells would be 44.8 HOURS. The batched version maps over the collection and
reduceRegions over many cells at once, the pattern 04_extract_ndvi_raw.py already
uses. Output schema is identical to the station extractor's, so stages 2-5
consume it unchanged, which matters because the model was fitted on AOD those
stages produced.

Chunk size had to become adaptive. Rows per chunk swing by an order of magnitude
across the year -- a 100-cell block returned 4659 rows in March and 67 in July,
because the monsoon wipes out MAIAC retrievals -- so one fixed size either
overflows in winter or wastes requests in the monsoon. November overflowed GEE's
5000-element limit at 5114 rows. The chunk now halves on that specific error and
retries each half. The first version retried the identical request four times
with backoff before giving up, which was pure waste: that error is
DETERMINISTIC, so the same request can never succeed. *Alternative rejected:*
globally shrinking the chunk size, which would pay the winter cost in every
month of the year.

**Per-chunk resume caching** for both the OSM tiles and the MAIAC chunks. At 288
chunks a non-resumable run is a coin flip; when the 5000-element failure killed
the first MAIAC attempt, the 8 completed chunks were reused rather than redone.

### Step 6 COMPLETE -- all 23 features on the 1 km grid (2026-10-09)

871,620 rows, 2388 cells x 365 days. Committed one feature group at a time so
any single feature can be reverted independently.

Cost collapsed far below the estimate, because the extractors already
deduplicate to unique PIXELS and the reanalysis grids are coarse:
ERA5-Land 2388 cells -> 33 pixels (2.2x station cost), ERA5 BLH -> 6 (1.5x),
MERRA-2 -> 4 (~1x). Only MAIAC is genuinely per-point: 115 min batched against
44.8 h per-point.

Assembly reuses the training master-table builder, which gained an optional
`--base_table` because the grid has no PM2.5 target to define its rows. A
separate assembly script was rejected: it would have duplicated the NDVI
merge_asof period join, a feature-defining operation. The training path was
re-run and diffed to confirm it is untouched -- 13,641 rows, max |diff| 0.0.
`season` is derived for the grid from SEASON_MONTHS copied verbatim from the
CPCB script, and the resulting month-to-season mapping was verified identical
to the training data's.

**Extrapolation -- the open product question.** 1334 of 2388 cells (56%) fall
outside the training range on at least one feature. An earlier figure of 37%
in this log was computed over the land-use features alone and is superseded.
Severity, as a fraction of the training range's width:

| | cells | |
|---|---|---|
| inside range | 1054 | 44% |
| <5% beyond (trivial) | 262 | 11% |
| 5-25% | 487 | 20% |
| 25-50% | 249 | 10% |
| 50-100% | 243 | 10% |
| >100% (no information) | 93 | 4% |

So 55% are inside or trivially outside; 585 (25%) are materially extrapolating.
`dist_to_nearest_powerplant_km` drives the worst overshoot for 537 of them,
`slope_deg` for 206. `road_density` and `built_up_pct` have training MINIMA well
above zero -- every CPCB station is urban -- so any genuinely rural cell is
extrapolation by construction.

LightGBM cannot extrapolate (it returns the boundary prediction) and the
conformal interval was calibrated only on out-of-fold residuals at the 42 urban
stations, so neither the estimate nor its interval is validated there. Flags are
in `reports/prediction/`; how they are surfaced in the API, map and download is
left for Muhammed.

### Step 7 COMPLETE -- the real grid is served (2026-10-09)

The app serves the model's own per-cell prediction. The synthetic IDW surface
and the nearest-station proxy are both gone.

**Inference.** 871,620 predictions precomputed by
`scripts/prediction/inference/01_predict_grid.py` and stored as parquet (~5 MB).
Precomputed rather than per-request: the booster is deterministic, the output
is a DVC-tracked artifact that can be diffed, and the API serves it straight
from a date-indexed dict. The script hard-fails if the feature matrix does not
exactly match the booster's feature names and order -- silently reordered
features would give plausible-looking wrong numbers, the failure least likely
to be noticed.

Predictions are physically sensible: 12.9-440.2 ug/m3, no negatives, overall
mean 86.7 against the stations' observed 86.9, and the seasonal ordering is
right for Delhi (monsoon 37.9, summer 66.8, winter 137.3, post-monsoon 139.5).

**The validation that matters.** Grid cells have no ground truth, so the only
end-to-end test is whether the cell CONTAINING a station predicts what the
station predicts, using the same booster. That exercises all 23 features, every
join, the gap-fill, the NDVI period mapping and the season assignment at once.

| | |
|---|---|
| correlation | 0.9901 |
| mean difference | -0.228 ug/m3 |
| median difference | -0.110 ug/m3 |
| median abs difference | 2.678 ug/m3 (4.05%) |

Across 13,595 station-days. The near-zero bias is the informative part: a
mis-joined feature or shifted date would show as a systematic offset, not
scatter. The scatter present is expected -- cell centres sit up to 640 m from
their station and every covariate is a 1 km buffer statistic, so the two
describe genuinely different ground. Worst agreement is Jahangirpuri
(14.8 ug/m3 median), a dense heterogeneous area where 640 m changes the land
cover.

**Point queries return the containing cell** (Muhammed, 2026-10-09): "that's
why we build a model -- if we are giving the value of the nearest cell why do
we need a model". Connaught Place reads 204.5 ug/m3 from its own cell while the
monitor 2.6 km away observed 223.8 and the model predicts 206.9 there. The
station is still reported as context for judging the estimate, never as the
estimate.

**Extrapolation is surfaced, not hidden.** 585 of 2388 cells carry the flag
through both endpoints, are drawn with red hatching, and the readout names the
offending feature and how far beyond the training range it sits. Red hatching
(extrapolation) is kept SEPARATE from grey hatching (more than 10.6 km from a
station) because they are different claims: a cell can be close to a monitor
yet unlike every monitor, or far from one yet ordinary.

**All 365 dates are servable**, up from 346. The other 19 are days when no
monitor reported a usable reading, so they never entered training -- but the
satellite and meteorology inputs exist and the model predicts them normally.
Refusing them would withhold estimates for a gap in the GROUND TRUTH, which is
the gap a satellite model exists to fill.

**Leaflet runs with preferCanvas**: 2388 rectangles would make the default SVG
renderer sluggish, since it creates a DOM node per feature.

### Step 8 COMPLETE -- aggregation with measured interval shrinkage (2026-10-09)

Coarse products at 2, 5 and 10 km, as areal means of the 1 km grid.

**rho re-measured on all 42 stations.** rho is the same-day correlation between
two locations' prediction errors. It decides how much an areal mean's interval
may shrink, because averaging N cells helps only as far as their errors are
independent: `shrinkage = sqrt((1 + (N-1)*rho) / N)`.

Measured on spatial-LOSO out-of-fold RELATIVE residuals (relative because the
served interval is multiplicative) across all 861 station pairs:

    rho = 0.0500, bootstrap 95% CI [0.0219, 0.0852]

against the previously published 0.032, so the old aggregation table was
optimistic.

**Three findings, and one corrected a recommendation I had already given.**

1. rho is higher than published: 0.050, not 0.032.
2. I expected re-running on all 42 folds to fix the thin close-pair sample. It
   did not -- there are STILL exactly 8 pairs under 2 km. The constraint was
   never the number of folds; it is Delhi's station geometry, where only 12 of
   42 stations have a neighbour within 2 km.
3. **rho does not decay with distance** (slope +0.0003/km, p = 0.62). Stations
   1 km and 40 km apart share the same correlation, so this is a WHOLE-DAY
   effect rather than a local one: on 2025-05-22, 35 of 38 stations erred in
   the same direction by a mean of -29%, against 60% / -3% on a typical day.
   Something regional the model cannot see.

Finding 3 then overturned my own recommendation. I had argued for the
close-pair rho of 0.130 because it was "the right distance" for aggregating
adjacent cells. But if rho does not vary with distance, close pairs are not
better targeted -- they measure the same quantity with n=8 instead of n=861.
Bootstrapped over STATIONS (pairs sharing a station are dependent, so a
pair-level CI is several times too narrow), 0.130 falls outside the honest
interval. Using it would have meant choosing a value the data's own CI
excludes -- the same error as inventing a widening factor for extrapolating
cells.

**Decision (Muhammed): the point estimate, with the CI carried through.**
Every coarse row ships `interval_shrinkage` plus the shrinkage at both CI
bounds, so the uncertainty in the shrinkage is visible rather than a point
estimate being presented as exact.

| grid | cells | shrinkage | interval | CI range |
|---|---|---|---|---|
| 1 km | 1 | 1.000 | +/-55.0% | -- |
| 2 km | 4 | 0.536 | +/-29.5% | 0.527-0.548 |
| 5 km | 25 | 0.297 | +/-16.3% | 0.247-0.349 |
| 10 km | 100 | 0.244 | +/-13.4% | 0.180-0.309 |

Partial cells at the coverage edge get shrinkage from the number of cells
ACTUALLY averaged, not the full block count -- a 2-cell edge block given the
25-cell shrinkage would overstate its precision.

**A floor exists, and it is a design finding.** Because rho is a whole-day
effect it cannot be averaged away, so shrinkage tends to sqrt(rho) = 0.224
rather than zero as N grows. 10 km already sits at 0.244. **Aggregating beyond
about 5 km buys very little precision** -- worth offering for file size, not
for accuracy.

### Step 9 COMPLETE -- gridded downloads, products 2 and 3 (2026-10-09)

`/api/download/grid` with `/preview` and `/columns`, at 1, 2, 5 and 10 km, in
zip (default), parquet or csv.

**Product 2 (1 km, WITH features) is reproducible, and that was verified rather
than claimed.** Downloaded a day's file, re-ran the booster on the shipped
feature columns, and all 2388 rows reproduce `pm25_predicted_ugm3` to within
0.005 ug/m3 -- which is only the 2dp rounding in the stored value. That
property is the entire reason the native-resolution product ships features.

**Product 3 (2/5/10 km) ships NO features, and the omission is the feature.**
LightGBM is nonlinear, so `mean(f(x)) != f(mean(x))`. Supplying averaged
features beside averaged predictions would invite a researcher to re-run the
model and get a different answer, with nothing to say which was right. Not
shipping them makes the mistake impossible rather than merely documented.

**THE AREAL-MEAN CAVEAT is the most important thing in these products.** A
coarse interval is narrower -- +/-16% at 5 km against +/-55% at 1 km -- and the
natural misreading is that the coarse product is simply better data.

A real 5 km cell on 2025-12-15 makes the problem concrete. It reports 282.9
ug/m3 with a 90% interval of 237-329. Inside it, the 25 one-kilometre
predictions range from 201 to 327 -- a 126 ug/m3 spread, with the lowest cell
falling outside the 5 km interval entirely.

So a user who reads "283, give or take 45" as a statement about their street is
wrong, and the data looks like it supports them. The +/-16% describes how well
the AVERAGE across 25 km2 is known. Individual streets vary for real reasons:
one is beside a highway, another backs onto parkland.

For a point question the honest answer is the 1 km value with its +/-55%. The
coarse product is better only if the question genuinely is about an area
average -- regional exposure, or comparing districts.

The warning therefore appears in three places, because one is not enough:
`ATTRIBUTION.txt` in every coarse bundle (so it travels with the file even if
passed on), the `/columns` response (for anyone scripting), and the UI note.
This is the same class of problem as the AOD provenance flag: a number that is
correct but easy to misread, where the fix is making the misreading hard rather
than trusting people to know.

Extrapolation carries through at every resolution -- 1 km ships
`outside_training_range`, `worst_overshoot_frac` and `worst_feature` per cell;
coarse products ship `fraction_outside_training_range`.

Sizes, one week: 1 km 719 KB parquet / 6.6 MB csv; 5 km 25 KB / 66 KB. A full
year at 1 km is 326 MB as CSV, which is why parquet is the default and the UI
defaults to one month -- 871,620 rows is a surprising thing to hand someone who
just clicked Download.

### UI rework -- multi-resolution map and an About page (2026-10-09)

Muhammed's request: let the map show every resolution, and strip the main
screen back because it had become a wall of prose with awkward gaps.

**Multi-resolution map.** `/api/grid` takes `resolution_km` and serves 1, 2, 5
or 10 km from the aggregated frames, rectangle size derived per level:
2388 / 605 / 105 / 29 cells at +/-55.0% / 29.5% / 16.3% / 13.4%.

**The main screen is now map, readout, compact AQI legend and four model
numbers.** Everything explanatory moved to `/about.html`: validation including
the station-agreement check, what the interval does and does not promise, the
marginal-vs-conditional limitation, why distance is not a confidence grade,
where the model extrapolates and why the interval is not widened there, the
areal-mean trap with the worked example, the three AOD provenance cases, and
sources. Feature pickers collapsed behind a disclosure instead of showing 21
checkboxes by default. The page went from 608 lines of mostly prose to a
22 KB screen plus a 13 KB reference page.

**One deliberate departure from "move the warnings out".** Explanations moved;
warnings that change what the CURRENT number means stayed inline, shown only
when they apply. There are two: switching to a coarse resolution (the interval
tightens, and the reason is that the quantity changed from a point to an area
average, not that the data improved), and querying a cell outside the training
range. Both are one line and link to the full account. Muhammed's point that
these already ship with downloads is right, but someone reading the map may
never download anything, and the misreading happens at the moment the number
changes.

`/about.html` is routed explicitly, since StaticFiles is mounted at `/static`
and the bare path would otherwise 404.

### UI fixes and the reset control (2026-10-09)

Two bugs Muhammed found, and one addition.

**The extrapolation warning never cleared.** Clicking a flagged cell showed it;
clicking a valid one left it on screen, so a cell inside the training range
appeared to be outside it. A stale warning is worse than none, because it makes
the flag untrustworthy everywhere. The cause was a missing else, but the fix
needed care: the context strip has two independent sources, and the resolution
warning must survive a point query while the point warning must not survive a
valid click. They are now separate state, re-rendered together. Verified by
driving the real page and reading the strip after each of seven actions.

**Table headers were left-aligned over right-aligned numbers.** The rule was
`td.num{text-align:right}`, which targets `td` only, so `th.num` stayed left and
the two read as unrelated columns. Now `td.num,th.num`, applied in index.html
too so the next table added there does not inherit it.

**Reset-view control**, under the zoom buttons. `fitBounds` with
`animate:false` -- the flag is load-bearing, not stylistic: an animated reset
can fail to land, and did exactly that under test, with the click reaching the
handler but the view not moving. `HOME_BOUNDS` is defined once and used for
both the initial view and the reset so they cannot drift.

The icon colour is deliberately not themed with `var(--ink)`: Leaflet's
controls keep a white background in dark mode, so an `--ink` colour rendered
the glyph white-on-white and it vanished. It matches Leaflet's own `#333`.

`window.map` is now exposed. A top-level `const` is not a window property, so
the map was unreachable from the console -- a nuisance on a local research tool
and the reason the UI could not be tested from outside.

**Testing Leaflet here is harder than it looks, and four approaches gave
confident wrong answers before one worked.** Recorded because they will recur:

- a synthetic double-click to zoom ALSO fires the map click handler, which runs
  a prediction and drops a pin, so the comparison measured page state;
- comparing screenshots conflated tile loading with view changes;
- the map pane's CSS transform does not change on pure zoom, only on pan, so it
  is useless as a zoom indicator;
- reading zoom from tile URLs is unreliable because Leaflet keeps stale tiles
  during transitions;
- and Leaflet's animated zoom does not complete under headless Chrome's virtual
  clock at all, which made the zoom-in BUTTON look broken and sent the
  diagnosis after the wrong fault.

Only setting the view with `animate:false` and reading `map.getZoom()` directly
measured what it claimed to.

### docs/PROJECT_SUMMARY.md -- the presentable account (2026-10-09)

Muhammed's request: the task logs have everything but are too long and carry
too much incidental detail to revise from. He needed one document he could read
before presenting the project or sitting an interview.

**Organised by what someone will ASK, not by what happened when.** The task
logs are already chronological; a second chronology would have been a shorter
copy of the same thing. So: a one-paragraph version with the single number to
lead with, then problem, data, features, modelling, validation, uncertainty,
the product, a decisions table, problems-and-how-found, limitations, likely
questions with answers, and a numbers cheat-sheet.

**Every figure was read out of `reports/` rather than from the logs or from
memory**, because he will be quoting them under pressure. Five things that
check found, which is why the rule was worth following:

- the Geofabrik loss was garbled in my first draft (I had written "57 power
  plants, 24% of roads"; the artifact says 112 against 170 with all 57 east of
  lon 77.36 missing, and 76% of training's roads RETAINED in the strip);
- a set of coverage-by-unusualness-band figures I had carried in my head could
  not be reproduced from `extrapolation_vs_coverage.csv` at all, so they were
  replaced with the median split that IS in the JSON (89.6% vs 90.2%, p = 0.78);
- the station driving the extrapolation correlation (5598, 102 range-widths out
  on `wetland_herbaceous_pct`) is also the worst-covered station at the 90%
  level (0.720) -- the two facts were recorded separately and the connection
  had not been made anywhere;
- "LightGBM beats the LME on 40 of 42" was RE-verified on the corrected data
  by merging the two fold tables, and holds on both R2 and RMSE. Not a new
  finding -- the `delhi-phase1-v21` tag message already carried "winning on 40
  of 42 stations" at the pre-fix numbers, which I had forgotten while writing
  the summary and claimed as a first. Worth recording because the count
  surviving the industrial-overlap fix unchanged is itself the useful fact;
- the Duan correction's awkward result was restated honestly -- it removes bias
  in the back-transformed mean (factor 1.132) but made spatial-LOSO RMSE
  slightly WORSE in 4 of 5 variants, which is a thing to volunteer rather than
  be caught by.

**One number computed fresh:** the top six features carry 87.7% of total gain
(season plus four meteorological variables plus AOD), against 7.8% for all 13
static land-use features combined. That pairing answers "why not add more
land-use features" in one line and is now the stated answer.

Interview questions were chosen as the ones the project's own weak points
invite, so each has a real answer: is 0.80 good (the comparable figure matters
more than the absolute), why not deep learning, how the grid can be validated
without ground truth, why distance was removed as a confidence measure, whether
the coarse products are better data, and what the weakest part is. The last one
is answered rather than deflected: conditional coverage, where the interval can
drop to 72% at a single station, and the stations where it fails are the
weakly-AOD-coupled ones -- a property that cannot be computed for an
unmonitored cell, which is exactly where it would be needed.

## Pending

**The 10-step plan is complete.** Steps 1-9 are in Completed above. Step 10 --
"decide whether 1 km is affordable" -- is moot: 1 km was built, and the answer
turned out to be yes, because the extractors deduplicate to unique pixels and
only MAIAC is genuinely per-point (115 min batched against a projected 44.8 h).

What remains is unforced work, not blockers.

**Decisions left with Muhammed:**

- **How prominent the extrapolation flag should be.** Currently red hatching
  plus a note. An alternative is refusing to serve the 93 cells that are more
  than a full training-range width beyond, which is defensible but withholds
  estimates a researcher might legitimately want.

**Known limitations, documented rather than fixed:**

- **The conformal interval guarantees MARGINAL coverage, not conditional.**
  Pooled held-out coverage is 89.9%, but per station it ranges 72.0% to 99.1%
  with 18 of 42 below target. The worst-covered station is 5598, the land-use
  outlier. Fixing this would need q = 1.049 (+/-105%), nearly doubling every
  interval.
- **rho's close-pair sample cannot be improved** without more monitors. 8 pairs
  under 2 km is what Delhi's network provides, and more folds do not help.
- **585 of 2388 cells are materially outside the training range**, 93 of them
  beyond any information the model has. Flagged everywhere, not corrected --
  whether unusualness predicts worse coverage was tested and found unsupported.
- **CPCB/OpenAQ redistribution terms** and the ODbL Produced Work vs Derivative
  Database question for the OSM-derived columns: still unverified, flagged
  rather than asserted since the first download product shipped.

**Not started, and deliberately so:** deployment beyond localhost. The scope
agreed was a local demo for the internship.

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
