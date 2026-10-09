import argparse
import io
import json
import math
import os
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# Local-only serving app for the Delhi Phase 1 LightGBM model. Historical /
# retrospective point queries over the trained study window -- NOT a live air
# quality service. The model's inputs (ERA5 reanalysis ~5 day latency, MAIAC
# AOD 1-3 day latency, both aggregated to a daily satellite-overpass window)
# physically cannot answer "what is the PM2.5 right now".
#
# STUB STATUS (2026-10-08): everything here is real except the surface between
# stations. Real: the trained booster, predictions at the 42 station locations
# for any date in the window, station coordinates, the published CV metrics,
# and the coverage/confidence logic. Synthetic: /api/grid interpolates station
# predictions by inverse distance weighting, flagged synthetic=true in every
# response, and is a placeholder until the per-cell feature grid is built.

# Same feature contract as scripts/modeling/lightgbm/*.py -- kept as a separate
# copy here (repo convention: no shared utils module across scripts). Must be
# kept in sync if the feature set changes.
ID_COLS = ["location_id", "name", "date"]
TARGET_COL = "pm25_daily"
GROUP_COL = "location_id"
DATE_COL = "date"
SEASON_COL = "season"
SEASON_CATEGORIES = ["summer", "monsoon", "post_monsoon", "winter"]

# Same mapping as scripts/datasets/cpcb/6-trim_and_season.py, which assigns
# season for the training data. Must stay in sync: season is a model feature,
# so a different month boundary would be a different feature.
SEASON_FOR_MONTH = {
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
    12: "winter", 1: "winter", 2: "winter",
}

# Delhi NCR bounding box -- the box the station roster was drawn from
# (scripts/datasets/cpcb/cpcb_fetcher.py CITY_BBOX["delhi"]). Used to frame the
# map and seed the grid, but NOT as the coverage gate -- see below.
BBOX_MIN_LON = 76.8381
BBOX_MIN_LAT = 28.4126
BBOX_MAX_LON = 77.3477
BBOX_MAX_LAT = 28.8814

# Coverage is gated on DISTANCE TO THE NEAREST TRAINING STATION, not on the
# bounding box. The rectangle is incoherent as a validity test: sampled across
# the bbox, 20% of its area is already more than 10.6 km from any station and
# the NW corner is 23.8 km out, while East Noida sits 6.7 km from station 5598
# yet falls outside the box. The old rule refused a point 6.7 km from data and
# accepted one 10.4 km away (Gurugram). Distance tracks what actually
# determines validity; the rectangle tracks nothing.
#
# Default 15 km is ~1.4x the largest gap spatial-LOSO validation ever spanned
# (10.6 km). It keeps 92% of the old bbox area, drops the corners that were
# least defensible, and admits nearby NCR that is better supported than parts
# of the box.
DEFAULT_MAX_STATION_DISTANCE_KM = 15.0

EARTH_RADIUS_KM = 6371.0

# Uncertainty is reported as a conformal prediction interval, NOT as a
# distance-based confidence band. An earlier version of this app graded
# confidence by distance to the nearest station; that was measured against
# actual per-fold error and found unsupported (r = 0.149 with RMSE, p = 0.33,
# n = 42). Distance still gates SCOPE -- see DEFAULT_MAX_STATION_DISTANCE_KM --
# because the absence of a correlation across the 0.67-10.6 km the network
# spans does not license unlimited extrapolation. But it is not a confidence
# signal and is no longer presented as one.
#
# The interval comes from scripts/modeling/lightgbm/04_calibrate_conformal_intervals.py:
#   interval = prediction * (1 +/- q)
# with q a percentile of the relative absolute residual measured on
# spatial-LOSO out-of-fold predictions. 90% is the level served (chosen
# 2026-10-08); the calibration file also carries 80% and 95%.
SERVED_COVERAGE_LEVEL = "90"

# 10.6 km is the largest gap spatial-LOSO validation ever spanned (the station
# network's own maximum nearest-neighbour distance). Reported as context
# alongside a query, not as an uncertainty estimate.
VALIDATED_GAP_KM = 10.6

# India National AQI breakpoints for 24h PM2.5 (ug/m3). Used for the
# interpretable band shown next to the number -- far more meaningful to a
# reader than a bare concentration.
AQI_BANDS = [
    (0, 30, "Good"),
    (30, 60, "Satisfactory"),
    (60, 90, "Moderate"),
    (90, 120, "Poor"),
    (120, 250, "Very Poor"),
    (250, float("inf"), "Severe"),
]

# Columns a station download always carries, whatever the user selects.
# Identifiers are needed to make sense of a row at all; the observed value is
# the whole point of this product (it is the only one of the three downloads
# with measured rather than modelled PM2.5); and the model columns are the
# out-of-fold prediction, i.e. from a model that never saw this station -- the
# honest estimate, not the optimistic in-sample fit.
DOWNLOAD_ID_COLS = ["location_id", "name", "latitude", "longitude", "date", "season"]
DOWNLOAD_OBSERVED_COL = "pm25_observed_ugm3"
DOWNLOAD_MODEL_COLS = ["pm25_predicted_ugm3", "pm25_lower_90_ugm3", "pm25_upper_90_ugm3"]

# AOD provenance travels with the AOD column rather than being selectable.
# aod_055 has three provenances, not two:
#   aod_gap_filled=0                      -> MAIAC retrieval     6985 rows (51.4%)
#   aod_gap_filled=1, aod_055 not null    -> MERRA-2 calibrated  6509 rows (47.9%)
#   aod_gap_filled=1, aod_055 null        -> unfillable           101 rows (0.7%)
# The flag means "not a MAIAC retrieval", NOT "a value was imputed" -- on the
# unfillable rows neither MAIAC nor a usable MERRA-2 calibration was available,
# so aod_055 is left null (LightGBM splits on missing natively, so those rows
# still train and predict). aod_055 is the only column with nulls.
#
# This matters because network-wide AOD-PM2.5 coupling halves on filled rows
# (0.516 observed vs 0.296 gap-filled). A researcher who cannot tell the three
# apart is working with a column whose meaning silently changes between rows.
DOWNLOAD_PROVENANCE_COLS = ["aod_gap_filled", "confidence_rmse"]

# units / definition / source for every column that can appear in a download.
COLUMN_DICTIONARY = [
    ("location_id", "-", "CPCB station identifier (as used by OpenAQ)", "CPCB via OpenAQ"),
    ("name", "-", "CPCB station name", "CPCB via OpenAQ"),
    ("latitude", "degrees north", "station latitude, WGS84", "CPCB via OpenAQ"),
    ("longitude", "degrees east", "station longitude, WGS84", "CPCB via OpenAQ"),
    ("date", "YYYY-MM-DD", "calendar date, India Standard Time", "-"),
    ("season", "-", "summer / monsoon / post_monsoon / winter", "derived"),
    ("pm25_observed_ugm3", "ug/m3", "MEASURED daily mean PM2.5 at the station", "CPCB via OpenAQ"),
    ("pm25_predicted_ugm3", "ug/m3",
     "MODEL ESTIMATE from spatial leave-one-station-out cross-validation -- the "
     "prediction from a model fitted without this station, so it is an honest "
     "out-of-sample estimate and not an in-sample fit", "this model"),
    ("pm25_lower_90_ugm3", "ug/m3",
     "lower bound of the 90% conformal prediction interval", "this model"),
    ("pm25_upper_90_ugm3", "ug/m3",
     "upper bound of the 90% conformal prediction interval", "this model"),
    ("aod_055", "unitless",
     "aerosol optical depth at 550 nm, daily, 1 km", "MODIS MAIAC (MCD19A2), NASA"),
    ("aod_gap_filled", "0 or 1",
     "0 = aod_055 is a MAIAC retrieval. 1 = it is NOT a MAIAC retrieval, which "
     "covers two cases: filled from MERRA-2 (aod_055 has a value) or unfillable "
     "(aod_055 is null, 101 rows). So the three provenances are: flag=0 -> "
     "observed; flag=1 and aod_055 not null -> MERRA-2 calibrated; flag=1 and "
     "aod_055 null -> no AOD available. Filter on both columns, not the flag alone.",
     "derived"),
    ("confidence_rmse", "AOD units",
     "RMSE of the per-station per-season MAIAC~MERRA-2 regression that produced "
     "the fill, so smaller means a better-constrained fill. Blank on MAIAC-observed "
     "rows (no fill was needed) and on the 101 unfillable rows (no fill happened). "
     "Present on exactly the 6509 MERRA-2-calibrated rows. Note the units are AOD, "
     "not ug/m3 -- this describes error in the AOD fill, not in the PM2.5 estimate.",
     "derived"),
    ("temperature_c", "degrees C", "2 m air temperature, satellite-overpass window mean",
     "ERA5-Land, Copernicus / ECMWF"),
    ("relative_humidity", "percent", "2 m relative humidity, overpass-window mean",
     "ERA5-Land, Copernicus / ECMWF"),
    ("wind_speed", "m/s", "10 m wind speed, overpass-window mean",
     "ERA5-Land, Copernicus / ECMWF"),
    ("boundary_layer_height", "m", "planetary boundary layer height, overpass-window mean",
     "ERA5, Copernicus / ECMWF"),
    ("ndvi_mean", "unitless (-1 to 1)",
     "mean NDVI in a 1 km buffer, 5-day composite", "Sentinel-2, Copernicus"),
    ("ndvi_gap_filled", "0 or 1", "1 if ndvi_mean was filled from a neighbouring period",
     "derived"),
    ("tree_cover_pct", "percent", "share of 1 km buffer pixels in this land-cover class",
     "ESA WorldCover v200 (2021)"),
    ("shrubland_pct", "percent", "share of 1 km buffer pixels in this land-cover class",
     "ESA WorldCover v200 (2021)"),
    ("grassland_pct", "percent", "share of 1 km buffer pixels in this land-cover class",
     "ESA WorldCover v200 (2021)"),
    ("cropland_pct", "percent", "share of 1 km buffer pixels in this land-cover class",
     "ESA WorldCover v200 (2021)"),
    ("built_up_pct", "percent", "share of 1 km buffer pixels in this land-cover class",
     "ESA WorldCover v200 (2021)"),
    ("bare_sparse_veg_pct", "percent", "share of 1 km buffer pixels in this land-cover class",
     "ESA WorldCover v200 (2021)"),
    ("water_pct", "percent", "share of 1 km buffer pixels in this land-cover class",
     "ESA WorldCover v200 (2021)"),
    ("wetland_herbaceous_pct", "percent", "share of 1 km buffer pixels in this land-cover class",
     "ESA WorldCover v200 (2021)"),
    ("elevation_m", "m", "mean elevation in a 1 km buffer", "SRTM, NASA / USGS"),
    ("slope_deg", "degrees", "mean terrain slope in a 1 km buffer", "SRTM, NASA / USGS"),
    ("road_density_km_per_km2", "km/km2", "road length per unit area in a 1 km buffer",
     "OpenStreetMap, ODbL"),
    ("industrial_landuse_fraction", "fraction (0 to 1)",
     "industrial-tagged area divided by 1 km buffer area", "OpenStreetMap, ODbL"),
    ("dist_to_nearest_powerplant_km", "km", "great-circle distance to the nearest power plant",
     "OpenStreetMap, ODbL"),
]

# Columns every grid download carries, whatever resolution.
GRID_ID_COLS = ["location_id", "latitude", "longitude", "date", "season"]
GRID_MODEL_COLS = ["pm25_predicted_ugm3", "pm25_lower_90_ugm3", "pm25_upper_90_ugm3"]

# Resolutions offered. 1 km is the native grid and the only one that ships
# features; the rest are areal means of it.
GRID_RESOLUTIONS = [1, 2, 5, 10]

GRID_DICTIONARY = [
    ("location_id", "-",
     "grid cell identifier. 1 km cells are numbered from 900000; coarse cells "
     "use their row/column in the coarse lattice", "derived"),
    ("coarse_row", "-", "row index of the coarse cell in its own lattice", "derived"),
    ("coarse_col", "-", "column index of the coarse cell in its own lattice", "derived"),
    ("latitude", "degrees north", "cell centre, WGS84", "derived"),
    ("longitude", "degrees east", "cell centre, WGS84", "derived"),
    ("date", "YYYY-MM-DD", "calendar date, India Standard Time", "-"),
    ("season", "-", "summer / monsoon / post_monsoon / winter", "derived"),
    ("pm25_predicted_ugm3", "ug/m3",
     "MODEL ESTIMATE of daily mean PM2.5. Not a measurement. At 1 km this is "
     "the model applied to this cell's own features; at coarser resolutions it "
     "is the MEAN of the constituent 1 km predictions", "LightGBM model"),
    ("pm25_lower_90_ugm3", "ug/m3", "lower bound of the 90% prediction interval",
     "normalized conformal calibration"),
    ("pm25_upper_90_ugm3", "ug/m3", "upper bound of the 90% prediction interval",
     "normalized conformal calibration"),
    ("n_cells_averaged", "-",
     "how many 1 km cells were averaged into this coarse cell. Less than the "
     "full block at the edge of the coverage area, which is why the interval "
     "shrinkage varies", "derived"),
    ("interval_shrinkage", "-",
     "factor the 1 km interval was multiplied by for this cell: "
     "sqrt((1 + (N-1)*rho) / N) with rho = 0.050, the measured same-day "
     "correlation between locations' prediction errors", "derived"),
    ("interval_shrinkage_rho_low", "-",
     "the same factor at the lower bound of rho's 95% CI (0.0219) -- a tighter "
     "interval than served", "derived"),
    ("interval_shrinkage_rho_high", "-",
     "the same factor at the upper bound of rho's 95% CI (0.0852) -- a wider "
     "interval than served. The served value uses the point estimate; these two "
     "show how much the shrinkage itself is uncertain", "derived"),
    ("dist_to_nearest_station_km", "km",
     "distance to the nearest CPCB monitor. Context and provenance, NOT a "
     "confidence measure -- it was tested against per-fold prediction error and "
     "found unpredictive (r = 0.149, p = 0.33)", "derived"),
    ("outside_training_range", "true/false",
     "the cell's features fall outside the range the model was trained on. "
     "LightGBM cannot extrapolate, so it returns its boundary estimate, and the "
     "interval's coverage was measured only at monitored urban locations",
     "derived"),
    ("worst_overshoot_frac", "-",
     "how far outside the training range, as a fraction of that range's width. "
     "0.1 is marginal; 1.0 means a full training-range width beyond, where the "
     "model has no information", "derived"),
    ("worst_feature", "-", "which feature is furthest outside the training range",
     "derived"),
    ("fraction_outside_training_range", "-",
     "share of the constituent 1 km cells that are materially outside the "
     "training range", "derived"),
]

ATTRIBUTION_TEXT = """Delhi PM2.5 satellite-ground calibration -- Phase 1
================================================================

USING THIS DATA

This dataset is free to use for research and educational purposes with
attribution.

Please cite: Muhammed Fiyas, "Delhi PM2.5 satellite-ground calibration", 2026.


DATA SOURCES

PM2.5 ground truth     CPCB (Central Pollution Control Board), accessed via OpenAQ
Aerosol optical depth  MODIS MAIAC (MCD19A2), NASA
AOD gap-fill           MERRA-2, NASA GMAO
Meteorology            ERA5-Land (temperature, humidity, wind), Copernicus
                       Climate Change Service / ECMWF
Boundary layer height  ERA5, Copernicus Climate Change Service / ECMWF
Vegetation (NDVI)      Sentinel-2, Copernicus
Land cover             ESA WorldCover v200 (2021)
Terrain                SRTM, NASA / USGS
Roads, industrial      (c) OpenStreetMap contributors, ODbL
  land use, power
  plants

If you redistribute any part of this data, please carry this file with it.


WHAT THE MODEL NUMBERS MEAN, AND THEIR LIMITS

pm25_observed_ugm3 is MEASURED. pm25_predicted_ugm3 is a MODEL ESTIMATE and
should not be treated as a measurement.

Predictions come from spatial leave-one-station-out cross-validation: each
station's values are predicted by a model fitted without that station (and
without any station within 2 km of it). They are therefore out-of-sample
estimates rather than an in-sample fit.

Validated performance across all 42 stations, spatial LOSO:
  R2    0.803
  RMSE  33.73 ug/m3
  MAE   20.76 ug/m3

The 90% interval is a normalized conformal interval, prediction * (1 +/- 0.5514),
calibrated on those out-of-fold residuals.

IMPORTANT LIMITATION: that interval guarantees MARGINAL coverage -- correct on
average across all predictions -- and not CONDITIONAL coverage at every
location. Pooled held-out coverage is 89.9%, but per station it ranges from
67.4% (Sector-125 Noida) to 98.2%, and 15 of 42 stations fall below the 90%
target. The under-covered stations are those where column AOD tracks surface
PM2.5 weakly. Treat a single station's interval as indicative, not guaranteed.

THE AOD COLUMN HAS THREE PROVENANCES, NOT TWO

aod_gap_filled means "not a MAIAC retrieval", not "a value was imputed":

  aod_gap_filled == 0                        MAIAC retrieval      6985 rows (51.4%)
  aod_gap_filled == 1, aod_055 not null      MERRA-2 calibrated   6509 rows (47.9%)
  aod_gap_filled == 1, aod_055 null          no AOD available      101 rows (0.7%)

Filter on both columns, not the flag alone. aod_055 is the only column in this
file that contains nulls; the 101 unfillable rows are kept because LightGBM
splits on missing values natively, so they still carry a prediction.

AOD-PM2.5 coupling is roughly half as strong on the MERRA-2-calibrated rows
(network-wide r = 0.296) as on the MAIAC-retrieved rows (r = 0.516). If your
analysis depends on the AOD column, this split is worth respecting.


This model is RETROSPECTIVE. Its inputs (ERA5 reanalysis, MAIAC AOD) have
multi-day latency, so it cannot estimate present-day PM2.5.
"""

app = FastAPI(title="Delhi PM2.5 estimator (Phase 1, local demo)")

STATE = {}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_feature_list(df):
    return [col for col in df.columns if col not in ID_COLS and col != TARGET_COL]


def cast_season_categorical(df):
    # Explicit fixed category order, matching the training scripts. A plain
    # astype("category") here would derive codes from whatever rows are
    # present and silently disagree with the booster.
    df = df.copy()
    df[SEASON_COL] = pd.Categorical(df[SEASON_COL], categories=SEASON_CATEGORIES)
    return df


def aqi_band(value):
    for low, high, label in AQI_BANDS:
        if low <= value < high:
            return label
    return "unknown"


def prediction_interval(value):
    # Width scales with the prediction because the model's error is
    # multiplicative: the relative residual is roughly stable across the
    # concentration range while the absolute residual is not.
    q = STATE["conformal_q"]
    lower = max(0.0, value * (1.0 - q))  # negative PM2.5 is meaningless
    upper = value * (1.0 + q)
    return lower, upper


def distance_to_nearest_station(lat, lon):
    s = STATE["stations"]
    d = haversine_km(lat, lon, s["latitude"].values, s["longitude"].values)
    return float(d.min())


def load_state(model_path, dataset_path, station_path, metrics_path, grid_km,
               max_station_distance_km=DEFAULT_MAX_STATION_DISTANCE_KM,
               conformal_path=None, oof_path=None,
               grid_predictions_path=None, grid_cells_path=None,
               grid_features_path=None, aggregated_dir=None,
               aggregation_summary_path=None):
    booster = lgb.Booster(model_file=model_path)

    df = pd.read_csv(dataset_path)
    df = cast_season_categorical(df)
    feature_cols = build_feature_list(df)

    stations = pd.read_csv(station_path)
    stations = stations[stations["status"] == "KEEP"][
        [GROUP_COL, "name", "latitude", "longitude"]].reset_index(drop=True)

    with open(metrics_path) as f:
        metrics = json.load(f)

    oof = pd.read_csv(oof_path)[[GROUP_COL, DATE_COL, "predicted_pm25"]]
    print(f"Loaded out-of-fold predictions: {oof_path} ({len(oof)} rows)")

    with open(conformal_path) as f:
        conformal = json.load(f)
    level = conformal["levels"][SERVED_COVERAGE_LEVEL]
    print(f"Loaded conformal calibration: {SERVED_COVERAGE_LEVEL}% interval, "
          f"q={level['q']:.4f} (+/- {level['width_pct_of_prediction']}% of the prediction), "
          f"held-out coverage {100 * level['coverage_held_out_mean']:.1f}%")

    dates = sorted(df["date"].unique().tolist())

    print(f"Loaded model: {model_path} ({booster.num_trees()} trees)")
    print(f"Loaded features: {dataset_path} ({len(df)} rows, {len(feature_cols)} features)")
    print(f"Loaded stations: {len(stations)} KEEP")
    print(f"Date window: {dates[0]} to {dates[-1]} ({len(dates)} dates)")

    # The real 1 km grid. Predictions are precomputed by
    # scripts/prediction/inference/01_predict_grid.py and validated against the
    # stations (correlation 0.9901, median difference -0.110 ug/m3) before being
    # served -- see reports/prediction/station_cell_agreement.json.
    grid_cells = pd.read_csv(grid_cells_path)
    grid_cells = grid_cells[grid_cells["status"] == "KEEP"][
        [GROUP_COL, "latitude", "longitude", "dist_to_nearest_station_km"]]
    grid_preds = pd.read_parquet(grid_predictions_path)
    print(f"Loaded grid: {len(grid_cells)} cells, {len(grid_preds)} predictions, "
          f"{grid_preds['date'].nunique()} dates")
    n_flagged = int(grid_preds.groupby(GROUP_COL)["materially_outside"].first().sum())
    print(f"  {n_flagged} cells materially outside the training feature range")

    # Indexed by date so a request is a dict lookup rather than a scan of
    # 871,620 rows.
    grid_by_date = {d: g for d, g in grid_preds.groupby("date", sort=False)}

    # Cell coordinates as arrays, for the nearest-cell lookup on every point
    # query.
    cell_lat = grid_cells["latitude"].to_numpy()
    cell_lon = grid_cells["longitude"].to_numpy()
    cell_ids = grid_cells[GROUP_COL].to_numpy()

    # Download frames, assembled once at startup. The 1 km product joins the
    # predictions to the feature table so model(features) reproduces
    # pm25_predicted_ugm3 exactly -- that reproducibility is the whole point of
    # shipping features at native resolution.
    grid_download_1km = None
    grid_download_coarse = {}
    aggregation = None
    if grid_features_path is not None:
        feats = pd.read_csv(grid_features_path)
        base = grid_preds.rename(columns={
            "predicted_pm25_ugm3": "pm25_predicted_ugm3"})
        grid_download_1km = feats.merge(
            base[[GROUP_COL, DATE_COL, "pm25_predicted_ugm3",
                  "pm25_lower_90_ugm3", "pm25_upper_90_ugm3",
                  "outside_training_range", "worst_overshoot_frac",
                  "worst_feature"]],
            on=[GROUP_COL, DATE_COL], how="inner")
        grid_download_1km = grid_download_1km.merge(
            grid_cells[[GROUP_COL, "latitude", "longitude",
                        "dist_to_nearest_station_km"]],
            on=GROUP_COL, how="left")
        print(f"Grid download (1 km): {len(grid_download_1km)} rows, "
              f"{len(grid_download_1km.columns)} columns")

    if aggregated_dir is not None and os.path.isdir(aggregated_dir):
        for km in GRID_RESOLUTIONS:
            if km == 1:
                continue
            path = os.path.join(aggregated_dir,
                                f"grid_predictions_{km}km.parquet")
            if not os.path.exists(path):
                continue
            frame = pd.read_parquet(path)
            frame = frame.rename(columns={"pm25_ugm3": "pm25_predicted_ugm3"})
            # A stable id for the coarse cell, so a download can be joined back
            # to the map and to other dates.
            frame["location_id"] = (frame["coarse_row"].astype(str) + "_"
                                    + frame["coarse_col"].astype(str))
            frame["season"] = frame[DATE_COL].map(
                lambda d: SEASON_FOR_MONTH[int(str(d)[5:7])])
            grid_download_coarse[km] = frame
            print(f"Grid download ({km} km): {len(frame)} rows, "
                  f"{frame.groupby(['coarse_row', 'coarse_col']).ngroups} cells")

    if aggregation_summary_path is not None and os.path.exists(aggregation_summary_path):
        with open(aggregation_summary_path) as f:
            aggregation = json.load(f)

    STATE.update({
        "grid_download_1km": grid_download_1km,
        "grid_download_coarse": grid_download_coarse,
        "aggregation": aggregation,
        "grid_cells": grid_cells,
        "grid_by_date": grid_by_date,
        "cell_lat": cell_lat,
        "cell_lon": cell_lon,
        "cell_ids": cell_ids,
        "grid_dates": sorted(grid_preds["date"].unique().tolist()),
        # Cell size in degrees, derived from the grid itself so the map's
        # rectangles match the cells exactly.
        "grid_lat_step": float(np.diff(np.unique(cell_lat)).min()),
        "grid_lon_step": float(np.diff(np.unique(cell_lon)).min()),
        "booster": booster,
        "df": df,
        "feature_cols": feature_cols,
        "stations": stations,
        "metrics": metrics,
        # What the API can answer for is what the GRID covers, which is 365
        # days. The station dataset has 346: the other 19 are days when no
        # monitor reported a usable PM2.5 value, so they never entered
        # training. The grid has full meteorology and AOD on those days and
        # the model can predict them perfectly well -- refusing them would
        # withhold estimates for a gap in the GROUND TRUTH, which is exactly
        # the gap a satellite model exists to fill.
        "dates": dates,
        "station_dates": dates,
        "grid_km": grid_km,
        "max_station_distance_km": max_station_distance_km,
        "oof": oof,
        "conformal": conformal,
        "conformal_q": level["q"],
        "model_path": model_path,
    })


def predict_rows(rows):
    features = rows[STATE["feature_cols"]]
    return STATE["booster"].predict(features)


def station_predictions_for_date(date):
    # Real model output: the feature rows for this date are the same ones the
    # model was trained and validated on, so these are genuine predictions,
    # not placeholders.
    rows = STATE["df"][STATE["df"]["date"] == date]
    if len(rows) == 0:
        return None
    out = rows[[GROUP_COL, "name", "date"]].copy()
    out["predicted_pm25"] = predict_rows(rows)
    out["observed_pm25"] = rows[TARGET_COL].values
    coords = STATE["stations"].set_index(GROUP_COL)
    out["latitude"] = out[GROUP_COL].map(coords["latitude"])
    out["longitude"] = out[GROUP_COL].map(coords["longitude"])
    return out.dropna(subset=["latitude", "longitude"])


def nearest_station_record(lat, lon, date):
    # Context only: what the closest monitor actually measured that day. Useful
    # for judging an estimate against ground truth; never the estimate itself.
    preds = station_predictions_for_date(date)
    if preds is None or len(preds) == 0:
        return None
    d = haversine_km(lat, lon, preds["latitude"].values, preds["longitude"].values)
    i = int(np.argmin(d))
    row = preds.iloc[i]
    observed = row.get("observed_pm25")
    return {
        "location_id": int(row[GROUP_COL]),
        "name": str(row["name"]),
        "distance_km": round(float(d[i]), 2),
        "observed_pm25_ugm3": (None if observed is None or pd.isna(observed)
                               else round(float(observed), 1)),
        "model_prediction_ugm3": round(float(row["predicted_pm25"]), 1),
    }


def nearest_cell(lat, lon):
    # The cell CONTAINING the point, found as the nearest cell centre. At 1 km
    # spacing that is at most ~700 m away, and the two are the same cell.
    d = haversine_km(lat, lon, STATE["cell_lat"], STATE["cell_lon"])
    i = int(np.argmin(d))
    return int(STATE["cell_ids"][i]), float(d[i])


def build_grid_cells(grid_km, margin_km):
    # Cell centres on a regular lat/lon grid approximating grid_km spacing,
    # seeded over the bbox expanded by margin_km so coverage can extend past
    # the box wherever stations justify it.
    lat_step = grid_km / 110.57
    lon_step = grid_km / (111.32 * math.cos(math.radians((BBOX_MIN_LAT + BBOX_MAX_LAT) / 2)))
    lat_pad = margin_km / 110.57
    lon_pad = margin_km / (111.32 * math.cos(math.radians((BBOX_MIN_LAT + BBOX_MAX_LAT) / 2)))
    lats = np.arange(BBOX_MIN_LAT - lat_pad, BBOX_MAX_LAT + lat_pad, lat_step)
    lons = np.arange(BBOX_MIN_LON - lon_pad, BBOX_MAX_LON + lon_pad, lon_step)
    cells = [(round(float(la), 5), round(float(lo), 5)) for la in lats for lo in lons]
    return cells, lat_step, lon_step


@app.get("/api/model-info")
def model_info():
    m = STATE["metrics"]
    return {
        "model": "delhi_phase1_lightgbm",
        "model_file": STATE["model_path"],
        "n_trees": STATE["booster"].num_trees(),
        "target": "pm25_daily (raw ug/m3, no transform)",
        "n_features": len(STATE["feature_cols"]),
        "features": STATE["feature_cols"],
        "n_training_stations": len(STATE["stations"]),
        "date_window": {"start": STATE["grid_dates"][0],
                        "end": STATE["grid_dates"][-1],
                        "n_dates": len(STATE["grid_dates"]),
                        "n_dates_without_station_data": len(
                            set(STATE["grid_dates"]) - set(STATE["station_dates"]))},
        "validation": {
            "scheme": "spatial leave-one-station-out, 2km buffer exclusion, 42 folds",
            "r2": round(m["r2"], 4),
            "within_r2_kawano": round(m["within_r2"], 4),
            "rmse_ugm3": round(m["rmse_ugm3"], 2),
            "mae_ugm3": round(m["mae_ugm3"], 2),
        },
        "bbox": {"min_lon": BBOX_MIN_LON, "min_lat": BBOX_MIN_LAT,
                 "max_lon": BBOX_MAX_LON, "max_lat": BBOX_MAX_LAT},
        "uncertainty": {
            "method": "normalized conformal prediction, scaled by the prediction",
            "formula": "interval = prediction * (1 +/- q)",
            "coverage_level_pct": int(SERVED_COVERAGE_LEVEL),
            "q": STATE["conformal_q"],
            "width_pct_of_prediction": STATE["conformal"]["levels"][
                SERVED_COVERAGE_LEVEL]["width_pct_of_prediction"],
            "calibrated_on": STATE["conformal"]["calibration_scheme"],
            "n_calibration_rows": STATE["conformal"]["n_calibration_rows"],
            "coverage_held_out_mean": STATE["conformal"]["levels"][
                SERVED_COVERAGE_LEVEL]["coverage_held_out_mean"],
            "caveat": (
                "Conformal prediction guarantees MARGINAL coverage -- correct on "
                "average across all predictions -- not CONDITIONAL coverage at every "
                "location. Pooled coverage is "
                f"{100 * STATE['conformal']['levels'][SERVED_COVERAGE_LEVEL]['coverage_held_out_mean']:.1f}%, "
                "but per station it ranges "
                f"{100 * STATE['conformal']['levels'][SERVED_COVERAGE_LEVEL]['coverage_held_out_min']:.1f}% to "
                f"{100 * STATE['conformal']['levels'][SERVED_COVERAGE_LEVEL]['coverage_held_out_max']:.1f}%, "
                f"with {STATE['conformal']['levels'][SERVED_COVERAGE_LEVEL]['n_stations_below_target']} "
                "of 42 stations below target. The under-covered stations are those "
                "where column AOD tracks surface PM2.5 weakly -- a property that "
                "cannot be computed for an unmonitored location."),
        },
        "coverage_rule": {
            "gate": "distance_to_nearest_training_station",
            "max_station_distance_km": STATE["max_station_distance_km"],
            "note": ("The bounding box frames the map but does not gate coverage -- "
                     "distance to data does. 10.6 km is the largest gap spatial-LOSO "
                     "validation spanned; past that an estimate is unvalidated."),
        },
        "limitations": [
            "Retrospective only. ERA5 reanalysis (~5 day latency) and MAIAC AOD "
            "(1-3 day latency) mean this cannot estimate present-day PM2.5.",
            "Valid only inside the Delhi NCR bounding box.",
            "Validated at 42 station locations; predictions far from any station "
            "are extrapolation beyond what spatial-LOSO tested.",
        ],
    }


@app.get("/api/stations")
def stations():
    s = STATE["stations"]
    return {"n": len(s), "stations": s.to_dict(orient="records")}


@app.get("/api/dates")
def dates():
    servable = STATE["grid_dates"]
    station_only = sorted(set(servable) - set(STATE["station_dates"]))
    return {"start": servable[0], "end": servable[-1],
            "n_dates": len(servable), "dates": servable,
            "n_dates_without_station_data": len(station_only),
            "dates_without_station_data": station_only,
            "note": ("every date the grid covers is servable. The dates listed in "
                     "dates_without_station_data had no usable monitor reading, so "
                     "they are absent from the training set -- but the satellite and "
                     "meteorology inputs exist, so the model predicts them normally")}


@app.get("/api/predict")
def predict(lat: float, lon: float, date: str):
    if date not in STATE["grid_by_date"]:
        return JSONResponse(status_code=422, content={
            "error": "date_out_of_window",
            "message": (f"No satellite or meteorology data for {date}. This model covers "
                        f"{STATE['grid_dates'][0]} to {STATE['grid_dates'][-1]}."),
            "date_window": {"start": STATE["grid_dates"][0],
                            "end": STATE["grid_dates"][-1]},
        })

    gate_km = distance_to_nearest_station(lat, lon)
    if gate_km > STATE["max_station_distance_km"]:
        return JSONResponse(status_code=422, content={
            "error": "outside_coverage",
            "message": (f"The nearest training station is {gate_km:.1f} km away, beyond the "
                        f"{STATE['max_station_distance_km']:.0f} km limit. The model has no "
                        f"basis for an estimate this far from any monitored site."),
            "requested": {"lat": lat, "lon": lon},
            "distance_to_nearest_station_km": round(gate_km, 2),
            "max_station_distance_km": STATE["max_station_distance_km"],
        })

    # The model's prediction for the 1 km cell containing this point -- not
    # the nearest station's value. Serving the nearest station would make the
    # model pointless anywhere between stations, which is most of Delhi.
    cell_id, cell_km = nearest_cell(lat, lon)
    day = STATE["grid_by_date"].get(date)
    if day is None:
        return JSONResponse(status_code=503, content={
            "error": "no_rows_for_date",
            "message": f"No grid predictions available for {date}.",
        })
    row = day[day[GROUP_COL] == cell_id]
    if len(row) == 0:
        return JSONResponse(status_code=503, content={
            "error": "no_prediction_for_cell",
            "message": f"Cell {cell_id} has no prediction for {date}.",
        })
    row = row.iloc[0]
    value = float(row["predicted_pm25_ugm3"])

    station_km = distance_to_nearest_station(lat, lon)
    nearest_station = nearest_station_record(lat, lon, date)

    return {
        "requested": {"lat": lat, "lon": lon, "date": date},
        "predicted_pm25_ugm3": round(value, 1),
        "aqi_band": aqi_band(value),
        "interval": {
            "coverage_pct": int(SERVED_COVERAGE_LEVEL),
            "lower_ugm3": round(float(row["pm25_lower_90_ugm3"]), 1),
            "upper_ugm3": round(float(row["pm25_upper_90_ugm3"]), 1),
            "width_pct_of_prediction": STATE["conformal"]["levels"][
                SERVED_COVERAGE_LEVEL]["width_pct_of_prediction"],
        },
        "cell": {
            "location_id": cell_id,
            "grid_km": 1.0,
            "point_to_cell_centre_km": round(cell_km, 3),
        },
        # Whether this cell's features fall outside the range the model was
        # trained on. LightGBM cannot extrapolate -- it returns the boundary
        # prediction -- and the interval was calibrated only at the 42 urban
        # stations, so neither is validated here. The interval is NOT widened:
        # whether unusualness predicts worse coverage was tested against the
        # stations' measured coverage and found unsupported, so any widening
        # factor would be invented. See reports/prediction/.
        "training_range": {
            "outside": bool(row["outside_training_range"]),
            "materially_outside": bool(row["materially_outside"]),
            "worst_feature": (None if not bool(row["outside_training_range"])
                              else str(row["worst_feature"])),
            "overshoot_range_widths": round(float(row["worst_overshoot_frac"]), 3),
            "note": ("this cell's features fall outside the range the model was "
                     "trained on; the interval's coverage was measured at monitored "
                     "urban locations and is unverified here"
                     if bool(row["materially_outside"]) else
                     "this cell's features fall inside the range the model was "
                     "trained on"),
        },
        # Provenance, NOT uncertainty -- distance does not predict error here
        # (r = 0.149, p = 0.33). Useful for filtering to cells near a monitor
        # or for validation study design; the interval is the uncertainty.
        "provenance": {
            "distance_to_nearest_station_km": round(station_km, 2),
            "beyond_largest_validated_gap": bool(station_km > VALIDATED_GAP_KM),
            "note": ("distance is reported as context, not as a confidence measure -- "
                     "it was tested against per-fold error and found unpredictive"),
        },
        "nearest_station": nearest_station,
        "method": "grid_cell",
        "method_note": ("the model's prediction for the 1 km cell containing this "
                        "point, from its own AOD, meteorology, vegetation and land "
                        "cover"),
    }


@app.get("/api/grid")
def grid(date: str):
    if date not in STATE["grid_by_date"]:
        return JSONResponse(status_code=422, content={
            "error": "date_out_of_window",
            "message": (f"This model covers {STATE['dates'][0]} to {STATE['dates'][-1]}."),
        })

    day = STATE["grid_by_date"].get(date)
    if day is None or len(day) == 0:
        return JSONResponse(status_code=503, content={"error": "no_rows_for_date"})

    # Real per-cell model output. Every cell's value comes from its OWN AOD,
    # meteorology, vegetation and land cover -- not from interpolating station
    # values. Validated against the stations before being served: the cell
    # containing a station predicts what the station predicts, correlation
    # 0.9901, median difference -0.110 ug/m3 across 13,595 station-days.
    cells = STATE["grid_cells"]
    merged = day.merge(cells, on=GROUP_COL, how="inner")

    out = []
    for row in merged.itertuples(index=False):
        out.append({
            "lat": round(float(row.latitude), 5),
            "lon": round(float(row.longitude), 5),
            "pm25_ugm3": round(float(row.predicted_pm25_ugm3), 1),
            "aqi_band": aqi_band(float(row.predicted_pm25_ugm3)),
            "interval_lower_ugm3": round(float(row.pm25_lower_90_ugm3), 1),
            "interval_upper_ugm3": round(float(row.pm25_upper_90_ugm3), 1),
            "nearest_station_km": round(float(row.dist_to_nearest_station_km), 2),
            "beyond_largest_validated_gap": bool(
                row.dist_to_nearest_station_km > VALIDATED_GAP_KM),
            # Cells whose features fall outside the training range. Shown so a
            # viewer can see WHERE the model is extrapolating, rather than
            # being given a uniformly confident surface.
            "outside_training_range": bool(row.materially_outside),
        })

    n_outside = sum(1 for c in out if c["outside_training_range"])
    return {
        "date": date,
        "grid_km": STATE["grid_km"],
        "n_cells": len(out),
        "lat_step": round(STATE["grid_lat_step"], 6),
        "lon_step": round(STATE["grid_lon_step"], 6),
        "interval_coverage_pct": int(SERVED_COVERAGE_LEVEL),
        "n_cells_outside_training_range": n_outside,
        "method": "grid_cell",
        "method_note": ("each cell is the model's own prediction from that cell's "
                        "AOD, meteorology, vegetation and land cover"),
        "training_range_note": (
            f"{n_outside} of {len(out)} cells have features outside the range the "
            f"model was trained on -- every CPCB station is urban, so rural cells "
            f"are extrapolation by construction. LightGBM returns the boundary "
            f"prediction there, and the interval's coverage was measured only at "
            f"stations, so neither is validated for those cells."),
        "cells": out,
    }


def selectable_feature_cols():
    # Everything the model uses, minus the columns that are always included
    # anyway: the AOD provenance pair and `season`, which doubles as an
    # identifier. Leaving `season` selectable would put it in the frame twice.
    always = set(DOWNLOAD_PROVENANCE_COLS) | set(DOWNLOAD_ID_COLS)
    return [c for c in STATE["feature_cols"] if c not in always]


def data_dictionary_text(columns):
    lines = ["COLUMN DICTIONARY", "=" * 70, ""]
    known = {name: (unit, desc, src) for name, unit, desc, src in COLUMN_DICTIONARY}
    for col in columns:
        unit, desc, src = known.get(col, ("-", "(undocumented)", "-"))
        lines.append(f"{col}")
        lines.append(f"  units  : {unit}")
        lines.append(f"  meaning: {desc}")
        lines.append(f"  source : {src}")
        lines.append("")
    return "\n".join(lines)


def build_station_download(date_from, date_to, features):
    # Observed values and features from the modelling dataset, joined to the
    # out-of-fold prediction so the download carries measurement, model
    # estimate and interval side by side.
    df = STATE["df"]
    rows = df[(df[DATE_COL] >= date_from) & (df[DATE_COL] <= date_to)].copy()
    if len(rows) == 0:
        return None

    coords = STATE["stations"].set_index(GROUP_COL)
    rows["latitude"] = rows[GROUP_COL].map(coords["latitude"])
    rows["longitude"] = rows[GROUP_COL].map(coords["longitude"])
    rows[DOWNLOAD_OBSERVED_COL] = rows[TARGET_COL]

    oof = STATE["oof"]
    merged = rows.merge(oof, on=[GROUP_COL, DATE_COL], how="left")
    merged["pm25_predicted_ugm3"] = merged["predicted_pm25"].round(2)
    q = STATE["conformal_q"]
    merged["pm25_lower_90_ugm3"] = (merged["predicted_pm25"] * (1.0 - q)).clip(lower=0).round(2)
    merged["pm25_upper_90_ugm3"] = (merged["predicted_pm25"] * (1.0 + q)).round(2)

    keep = (DOWNLOAD_ID_COLS + [DOWNLOAD_OBSERVED_COL] + DOWNLOAD_MODEL_COLS
            + DOWNLOAD_PROVENANCE_COLS + features)
    # Deduplicate while preserving order -- belt and braces against a column
    # that is both an identifier and a feature.
    seen = set()
    keep = [c for c in keep
            if c in merged.columns and not (c in seen or seen.add(c))]
    out = merged[keep].copy()
    out[SEASON_COL] = out[SEASON_COL].astype(str)
    return out


def parse_feature_selection(features):
    available = selectable_feature_cols()
    if features is None or features.strip() == "" or features.strip().lower() == "all":
        return available, []
    # An explicit "none" means the always-included columns only -- the measured
    # value, the estimate and its interval. Treating it as an unknown column
    # name would work but would report a spurious warning.
    if features.strip().lower() == "none":
        return [], []
    asked = [f.strip() for f in features.split(",") if f.strip()]
    chosen = [f for f in asked if f in available]
    unknown = [f for f in asked if f not in available]
    return chosen, unknown


@app.get("/api/download/stations/columns")
def download_station_columns():
    return {
        "always_included": {
            "identifiers": DOWNLOAD_ID_COLS,
            "observed": [DOWNLOAD_OBSERVED_COL],
            "model": DOWNLOAD_MODEL_COLS,
            "aod_provenance": DOWNLOAD_PROVENANCE_COLS,
            "why_provenance_is_not_optional": (
                "47.9% of aod_055 values are calibrated from MERRA-2 rather than "
                "retrieved by MAIAC, and AOD-PM2.5 coupling halves on those rows "
                "(0.516 vs 0.296), so a column that could not be filtered would "
                "silently change meaning between rows"),
            "aod_provenance_classes": {
                "maiac_observed": "aod_gap_filled == 0 (6985 rows, 51.4%)",
                "merra2_calibrated": "aod_gap_filled == 1 and aod_055 notnull (6509 rows, 47.9%)",
                "unfillable": "aod_gap_filled == 1 and aod_055 isnull (101 rows, 0.7%)",
                "note": ("The flag means 'not a MAIAC retrieval', not 'a value was "
                         "imputed', so filter on both columns. aod_055 is the only "
                         "column in the download that contains nulls."),
            },
        },
        "selectable_features": selectable_feature_cols(),
        "dictionary": [
            {"column": n, "units": u, "meaning": d, "source": s}
            for n, u, d, s in COLUMN_DICTIONARY
        ],
    }


@app.get("/api/download/stations/preview")
def download_station_preview(date_from: str = None, date_to: str = None,
                             features: str = "all"):
    date_from = date_from or STATE["dates"][0]
    date_to = date_to or STATE["dates"][-1]
    chosen, unknown = parse_feature_selection(features)
    out = build_station_download(date_from, date_to, chosen)
    if out is None:
        return JSONResponse(status_code=422, content={
            "error": "no_rows",
            "message": f"No station rows between {date_from} and {date_to}.",
            "date_window": {"start": STATE["dates"][0], "end": STATE["dates"][-1]},
        })
    csv_bytes = len(out.to_csv(index=False).encode("utf-8"))
    return {
        "date_from": date_from, "date_to": date_to,
        "n_rows": len(out), "n_columns": out.shape[1],
        "n_stations": int(out[GROUP_COL].nunique()),
        "columns": out.columns.tolist(),
        "unknown_features_ignored": unknown,
        "estimated_size": {
            "csv_mb": round(csv_bytes / 1e6, 2),
            "parquet_mb": round(csv_bytes / 1e6 * 0.10, 2),
        },
    }


@app.get("/api/download/stations")
def download_stations(date_from: str = None, date_to: str = None,
                      features: str = "all", format: str = "zip"):
    if format not in ("zip", "csv", "parquet"):
        return JSONResponse(status_code=422, content={
            "error": "bad_format", "message": "format must be zip, csv or parquet"})

    date_from = date_from or STATE["dates"][0]
    date_to = date_to or STATE["dates"][-1]
    chosen, _ = parse_feature_selection(features)
    out = build_station_download(date_from, date_to, chosen)
    if out is None:
        return JSONResponse(status_code=422, content={
            "error": "no_rows",
            "message": f"No station rows between {date_from} and {date_to}."})

    stem = f"delhi_pm25_stations_{date_from}_{date_to}"

    if format == "csv":
        return Response(out.to_csv(index=False).encode("utf-8"),
                        media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'})

    if format == "parquet":
        buf = io.BytesIO()
        out.to_parquet(buf, index=False)
        return Response(buf.getvalue(), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.parquet"'})

    # Default: a zip, so the data cannot travel without its dictionary and
    # attribution. A CSV separated from its provenance is the real risk -- the
    # numbers end up in a paper with no way back to their source.
    data_buf = io.BytesIO()
    out.to_parquet(data_buf, index=False)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}.parquet", data_buf.getvalue())
        z.writestr(f"{stem}.csv", out.to_csv(index=False))
        z.writestr("DATA_DICTIONARY.txt", data_dictionary_text(out.columns.tolist()))
        z.writestr("ATTRIBUTION.txt", ATTRIBUTION_TEXT)
    return Response(zip_buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.zip"'})



# ---------------------------------------------------------------------------
# Grid downloads: products 2 and 3.
#
# Product 2 is the 1 km grid WITH features, so model(features) reproduces the
# supplied prediction exactly -- the fully reproducible product.
#
# Product 3 is the coarser grids, which are areal means of product 2 and ship
# WITHOUT features. That is deliberate, not an omission: LightGBM is nonlinear,
# so mean(f(x)) != f(mean(x)). Shipping coarse features beside coarse
# predictions would invite a researcher to re-run the model on them and get a
# different answer, with nothing to say which was right. Not shipping them
# makes the mistake impossible.
# ---------------------------------------------------------------------------

def grid_resolution_frame(resolution_km):
    if resolution_km == 1:
        return STATE["grid_download_1km"]
    return STATE["grid_download_coarse"][resolution_km]


def selectable_grid_features():
    # The model features, minus the ones always included anyway.
    always = set(GRID_ID_COLS) | set(DOWNLOAD_PROVENANCE_COLS)
    return [c for c in STATE["feature_cols"] if c not in always]


def build_grid_download(resolution_km, date_from, date_to, features):
    df = grid_resolution_frame(resolution_km)
    rows = df[(df[DATE_COL] >= date_from) & (df[DATE_COL] <= date_to)].copy()

    if resolution_km == 1:
        keep = list(GRID_ID_COLS) + list(GRID_MODEL_COLS) + [
            "dist_to_nearest_station_km", "outside_training_range",
            "worst_overshoot_frac", "worst_feature"]
        keep = keep + list(DOWNLOAD_PROVENANCE_COLS) + list(features)
    else:
        # No features at coarse resolution -- see the note above.
        keep = ["location_id", "latitude", "longitude", DATE_COL, "season"] + \
               list(GRID_MODEL_COLS) + [
            "n_cells_averaged", "interval_shrinkage",
            "interval_shrinkage_rho_low", "interval_shrinkage_rho_high",
            "dist_to_nearest_station_km", "fraction_outside_training_range"]

    seen = set()
    keep = [c for c in keep if c in rows.columns and not (c in seen or seen.add(c))]
    return rows[keep]


def grid_attribution_text(resolution_km):
    extra = []
    if resolution_km == 1:
        extra.append(
            "RESOLUTION: 1 km, the model's native grid. Every covariate is a\n"
            "1 km-buffer statistic and MAIAC AOD is a 1 km product, so this is the\n"
            "only resolution whose features mean what the model was fitted on.\n"
            "Because the features are included, re-running the model on them\n"
            "reproduces pm25_predicted_ugm3 exactly.")
    else:
        agg = STATE["aggregation"]
        level = next((x for x in agg["levels"] if x["grid_km"] == resolution_km), None)
        extra.append(
            f"RESOLUTION: {resolution_km} km, the MEAN of up to "
            f"{resolution_km * resolution_km} one-kilometre predictions.\n"
            "\n"
            "NO FEATURES ARE INCLUDED, deliberately. LightGBM is nonlinear, so the\n"
            "model applied to averaged features does not equal the average of the\n"
            "model's predictions. Supplying coarse features beside coarse\n"
            "predictions would invite a reproduction attempt that could not\n"
            "succeed. For features, download the 1 km product.\n"
            "\n"
            "THE INTERVAL APPLIES TO THE AREAL MEAN, NOT TO ANY POINT INSIDE THE\n"
            "CELL. This is the easiest thing to misread in this dataset. A 5 km\n"
            f"interval of about +/-{100 * STATE['conformal']['levels']['90']['q'] * (level['shrinkage_full_block'] if level else 1):.0f}% "
            "is NOT a more precise estimate for a street inside\n"
            "that cell -- it is a more precise estimate of the average across the\n"
            "whole cell. For a point, use the 1 km product and its wider interval.")
    return ATTRIBUTION_TEXT + "\n\n" + "\n".join(extra) + "\n"


@app.get("/api/download/grid/columns")
def download_grid_columns(resolution_km: int = 1):
    if resolution_km not in GRID_RESOLUTIONS:
        return JSONResponse(status_code=422, content={
            "error": "bad_resolution",
            "message": f"resolution_km must be one of {GRID_RESOLUTIONS}"})
    agg = STATE["aggregation"]
    level = next((x for x in agg["levels"] if x["grid_km"] == resolution_km), None)
    body = {
        "resolution_km": resolution_km,
        "native_resolution": resolution_km == 1,
        "always_included": {
            "identifiers": GRID_ID_COLS,
            "model": GRID_MODEL_COLS,
            "provenance": ["dist_to_nearest_station_km"],
        },
        "dictionary": [{"column": n, "units": u, "meaning": d, "source": sc}
                       for n, u, d, sc in GRID_DICTIONARY],
    }
    if resolution_km == 1:
        body["selectable_features"] = selectable_grid_features()
        body["features_note"] = (
            "the 1 km product ships features, so model(features) reproduces "
            "pm25_predicted_ugm3 exactly")
        body["always_included"]["aod_provenance"] = list(DOWNLOAD_PROVENANCE_COLS)
        body["always_included"]["extrapolation"] = [
            "outside_training_range", "worst_overshoot_frac", "worst_feature"]
    else:
        body["selectable_features"] = []
        body["features_note"] = (
            "coarse products ship NO features. LightGBM is nonlinear, so "
            "mean(f(x)) != f(mean(x)) -- supplying averaged features beside "
            "averaged predictions would invite a reproduction that cannot "
            "succeed. Use the 1 km product for features.")
        body["always_included"]["aggregation"] = [
            "n_cells_averaged", "interval_shrinkage",
            "interval_shrinkage_rho_low", "interval_shrinkage_rho_high"]
        body["always_included"]["extrapolation"] = [
            "fraction_outside_training_range"]
        body["interval_note"] = (
            "the interval applies to the AREAL MEAN over the cell, not to any "
            "point inside it")
        if level:
            body["interval_shrinkage"] = {
                "full_block": level["shrinkage_full_block"],
                "at_rho_ci_low": level["shrinkage_rho_low"],
                "at_rho_ci_high": level["shrinkage_rho_high"],
                "rho": agg["rho"],
                "rho_ci": [agg["rho_ci_low"], agg["rho_ci_high"]],
            }
    return body


@app.get("/api/download/grid/preview")
def download_grid_preview(resolution_km: int = 1, date_from: str = None,
                          date_to: str = None, features: str = None):
    if resolution_km not in GRID_RESOLUTIONS:
        return JSONResponse(status_code=422, content={
            "error": "bad_resolution",
            "message": f"resolution_km must be one of {GRID_RESOLUTIONS}"})
    dates = STATE["grid_dates"]
    date_from = date_from or dates[0]
    date_to = date_to or dates[-1]
    chosen, unknown = ([], []) if resolution_km != 1 else parse_grid_features(features)
    frame = build_grid_download(resolution_km, date_from, date_to, chosen)
    csv_bytes = estimate_csv_bytes(frame)
    return {
        "resolution_km": resolution_km,
        "date_from": date_from, "date_to": date_to,
        "n_rows": len(frame),
        "n_columns": len(frame.columns),
        "n_cells": int(frame["location_id"].nunique()) if len(frame) else 0,
        "columns": list(frame.columns),
        "unknown_features_ignored": unknown,
        "estimated_size": {
            "csv_mb": round(csv_bytes / 1e6, 2),
            "parquet_mb": round(csv_bytes / 1e6 * 0.10, 2),
        },
    }


def parse_grid_features(features):
    available = selectable_grid_features()
    if features is None or features.strip() == "" or features.strip().lower() == "all":
        return available, []
    if features.strip().lower() == "none":
        return [], []
    asked = [f.strip() for f in features.split(",") if f.strip()]
    chosen = [f for f in asked if f in available]
    unknown = [f for f in asked if f not in available]
    return chosen, unknown


def estimate_csv_bytes(frame):
    if len(frame) == 0:
        return 0
    sample = frame.head(2000)
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    return int(len(buf.getvalue()) / len(sample) * len(frame))


@app.get("/api/download/grid")
def download_grid(resolution_km: int = 1, date_from: str = None,
                  date_to: str = None, features: str = None,
                  format: str = "zip"):
    if resolution_km not in GRID_RESOLUTIONS:
        return JSONResponse(status_code=422, content={
            "error": "bad_resolution",
            "message": f"resolution_km must be one of {GRID_RESOLUTIONS}"})
    if format not in ("zip", "csv", "parquet"):
        return JSONResponse(status_code=422, content={
            "error": "bad_format", "message": "format must be zip, csv or parquet"})
    dates = STATE["grid_dates"]
    date_from = date_from or dates[0]
    date_to = date_to or dates[-1]
    chosen, _ = ([], []) if resolution_km != 1 else parse_grid_features(features)
    frame = build_grid_download(resolution_km, date_from, date_to, chosen)
    stem = f"delhi_pm25_grid_{resolution_km}km_{date_from}_{date_to}"

    if format == "csv":
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"content-disposition":
                                 f'attachment; filename="{stem}.csv"'})
    if format == "parquet":
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False)
        return Response(content=buf.getvalue(), media_type="application/octet-stream",
                        headers={"content-disposition":
                                 f'attachment; filename="{stem}.parquet"'})

    # ZIP is the default so the data cannot travel without its dictionary and
    # attribution -- a CSV separated from its provenance is the real failure
    # mode, and for the coarse products the areal-mean caveat travels with it.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pq = io.BytesIO()
        frame.to_parquet(pq, index=False)
        zf.writestr(f"{stem}.parquet", pq.getvalue())
        csv_buf = io.StringIO()
        frame.to_csv(csv_buf, index=False)
        zf.writestr(f"{stem}.csv", csv_buf.getvalue())
        zf.writestr("DATA_DICTIONARY.txt",
                    grid_data_dictionary_text(list(frame.columns)))
        zf.writestr("ATTRIBUTION.txt", grid_attribution_text(resolution_km))
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"content-disposition":
                             f'attachment; filename="{stem}.zip"'})


def grid_data_dictionary_text(columns):
    lines = ["COLUMN DICTIONARY", "=" * 70, ""]
    known = {n: (u, d, sc) for n, u, d, sc in GRID_DICTIONARY}
    known.update({n: (u, d, sc) for n, u, d, sc in COLUMN_DICTIONARY})
    for col in columns:
        unit, desc, src = known.get(col, ("-", "(undocumented)", "-"))
        lines.append(col)
        lines.append(f"  units  : {unit}")
        lines.append(f"  meaning: {desc}")
        lines.append(f"  source : {src}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/lightgbm/lightgbm_full_model.txt")
    parser.add_argument("--dataset",
                        default="data/training/processed/modeling_datasets/lightgbm_ready_dataset.csv")
    parser.add_argument("--station_file",
                        default="data/stations/cpcb_stations_delhi_status.csv")
    parser.add_argument("--metrics",
                        default="reports/lightgbm/primary/cv_spatial_loso_aggregated.json")
    parser.add_argument("--grid_predictions",
                        default="data/prediction/processed/grid_predictions.parquet",
                        help="output of scripts/prediction/inference/01_predict_grid.py")
    parser.add_argument("--grid_cells",
                        default="data/prediction/processed/grid_1km_delhi.csv")
    parser.add_argument("--grid_features",
                        default="data/prediction/processed/modeling_datasets/"
                                "lightgbm_ready_grid.csv",
                        help="1 km feature table, shipped with the native-resolution "
                             "download so model(features) reproduces the prediction")
    parser.add_argument("--aggregated_dir",
                        default="data/prediction/processed/aggregated")
    parser.add_argument("--aggregation_summary",
                        default="reports/prediction/aggregation_summary.json")
    parser.add_argument("--conformal",
                        default="reports/lightgbm/primary/conformal_calibration.json",
                        help="output of 04_calibrate_conformal_intervals.py")
    parser.add_argument("--oof",
                        default="reports/lightgbm/primary/cv_spatial_loso_oof_predictions.csv",
                        help="per-row out-of-fold predictions, used so the station "
                             "download carries an honest out-of-sample estimate rather "
                             "than an in-sample fit")
    parser.add_argument("--max_station_distance_km",
                        default=DEFAULT_MAX_STATION_DISTANCE_KM, type=float,
                        help="refuse queries further than this from any training station")
    # Reporting only now: the served grid is whatever resolution
    # grid_predictions.parquet was built at, and the actual cell size is read
    # back from the cell coordinates. Kept so /api/grid can state its spacing.
    parser.add_argument("--grid_km", default=1.0, type=float,
                        help="nominal spacing of the served grid, for reporting")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    load_state(args.model, args.dataset, args.station_file, args.metrics, args.grid_km,
               args.max_station_distance_km, args.conformal, args.oof,
               args.grid_predictions, args.grid_cells,
               args.grid_features, args.aggregated_dir,
               args.aggregation_summary)

    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    import uvicorn
    print(f"Serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
