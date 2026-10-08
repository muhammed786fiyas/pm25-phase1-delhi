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

### 1. API + UI against a stub (2026-10-08)

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

### 2. Attribution and citation (2026-10-08)

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

## Pending

**Step 2** -- save per-row out-of-fold predictions from `02_validate_lightgbm_cv.py`
(`reports/lightgbm/primary/oof_predictions.csv`, 13,595 rows). They are computed on every
validation run and discarded; they are the raw material for all uncertainty work and
worth keeping independent of the app.

**Step 3** -- calibrate q on all 42 folds, write
`reports/lightgbm/primary/conformal_calibration.json`, verify coverage on held-out
stations. The q values above come from 12 of 42 stations: the method is settled, the
constant is not.

**Step 4** -- replace the distance-based confidence badge with the interval in API and
UI. Distance stays as a *scope* gate (where we are willing to answer at all, since the
absence of correlation across 0.67-10.6 km does not license unlimited extrapolation) but
stops being presented as *confidence*.

**Step 5** -- station download (cheap, the data exists today): parquet default, feature
selection, bundled data dictionary and attribution.

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
- A long-running dev server cannot be held open from a Claude Code background task (10
  minute ceiling). Run `python app/api.py` directly for a real session.
