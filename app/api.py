import argparse
import json
import math
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
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
SEASON_COL = "season"
SEASON_CATEGORIES = ["summer", "monsoon", "post_monsoon", "winter"]

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

# Confidence bands keyed to the actual station network, not round numbers.
# Observed station nearest-neighbour distances: median 3.13 km, 90th pct
# 5.91 km, max 10.57 km. So 10.57 km is the largest gap spatial-LOSO
# validation ever had to span -- past that, a query asks for more than the
# validation ever tested. Note the bbox corners are 12.5-23.8 km from any
# station, so a real part of Delhi falls in the lowest band.
CONFIDENCE_NEAR_KM = 3.0
CONFIDENCE_MID_KM = 6.0
CONFIDENCE_FAR_KM = 10.6

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


def confidence_for_distance(distance_km):
    # Reported alongside every prediction because the risk here is specific
    # and documented: the LME produced a 422 ug/m3 RMSE at station 5598 by
    # extrapolating into land-use space it never saw. LightGBM clamps instead
    # of exploding, but a point unlike any training station is still outside
    # what the validation covers.
    if distance_km <= CONFIDENCE_NEAR_KM:
        return "high", "within the median spacing of the station network (3.1 km)"
    if distance_km <= CONFIDENCE_MID_KM:
        return "moderate", "within the usual station spacing (90th percentile 5.9 km)"
    if distance_km <= CONFIDENCE_FAR_KM:
        return "low", "beyond typical spacing but inside the largest validated gap (10.6 km)"
    return "very low", ("further from any station than any station is from its own "
                        "nearest neighbour -- beyond what spatial-LOSO validation tested")


def distance_to_nearest_station(lat, lon):
    s = STATE["stations"]
    d = haversine_km(lat, lon, s["latitude"].values, s["longitude"].values)
    return float(d.min())


def load_state(model_path, dataset_path, station_path, metrics_path, grid_km,
               max_station_distance_km=DEFAULT_MAX_STATION_DISTANCE_KM):
    booster = lgb.Booster(model_file=model_path)

    df = pd.read_csv(dataset_path)
    df = cast_season_categorical(df)
    feature_cols = build_feature_list(df)

    stations = pd.read_csv(station_path)
    stations = stations[stations["status"] == "KEEP"][
        [GROUP_COL, "name", "latitude", "longitude"]].reset_index(drop=True)

    with open(metrics_path) as f:
        metrics = json.load(f)

    dates = sorted(df["date"].unique().tolist())

    print(f"Loaded model: {model_path} ({booster.num_trees()} trees)")
    print(f"Loaded features: {dataset_path} ({len(df)} rows, {len(feature_cols)} features)")
    print(f"Loaded stations: {len(stations)} KEEP")
    print(f"Date window: {dates[0]} to {dates[-1]} ({len(dates)} dates)")

    STATE.update({
        "booster": booster,
        "df": df,
        "feature_cols": feature_cols,
        "stations": stations,
        "metrics": metrics,
        "dates": dates,
        "grid_km": grid_km,
        "max_station_distance_km": max_station_distance_km,
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
        "date_window": {"start": STATE["dates"][0], "end": STATE["dates"][-1],
                        "n_dates": len(STATE["dates"])},
        "validation": {
            "scheme": "spatial leave-one-station-out, 2km buffer exclusion, 42 folds",
            "r2": round(m["r2"], 4),
            "within_r2_kawano": round(m["within_r2"], 4),
            "rmse_ugm3": round(m["rmse_ugm3"], 2),
            "mae_ugm3": round(m["mae_ugm3"], 2),
        },
        "bbox": {"min_lon": BBOX_MIN_LON, "min_lat": BBOX_MIN_LAT,
                 "max_lon": BBOX_MAX_LON, "max_lat": BBOX_MAX_LAT},
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
    return {"start": STATE["dates"][0], "end": STATE["dates"][-1],
            "n_dates": len(STATE["dates"]), "dates": STATE["dates"]}


@app.get("/api/predict")
def predict(lat: float, lon: float, date: str):
    if date not in STATE["dates"]:
        return JSONResponse(status_code=422, content={
            "error": "date_out_of_window",
            "message": (f"No satellite or meteorology data for {date}. This model covers "
                        f"{STATE['dates'][0]} to {STATE['dates'][-1]}."),
            "date_window": {"start": STATE["dates"][0], "end": STATE["dates"][-1]},
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

    preds = station_predictions_for_date(date)
    if preds is None or len(preds) == 0:
        return JSONResponse(status_code=503, content={
            "error": "no_rows_for_date",
            "message": f"No station feature rows available for {date}.",
        })

    distances = haversine_km(lat, lon, preds["latitude"].values, preds["longitude"].values)
    nearest = int(np.argmin(distances))
    distance_km = float(distances[nearest])
    row = preds.iloc[nearest]
    value = float(row["predicted_pm25"])
    level, reason = confidence_for_distance(distance_km)

    return {
        "requested": {"lat": lat, "lon": lon, "date": date},
        "predicted_pm25_ugm3": round(value, 1),
        "aqi_band": aqi_band(value),
        "confidence": {"level": level, "reason": reason,
                       "distance_to_nearest_station_km": round(distance_km, 2)},
        "nearest_station": {
            "location_id": int(row[GROUP_COL]),
            "name": str(row["name"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "observed_pm25_ugm3": round(float(row["observed_pm25"]), 1),
        },
        # Honest about what this endpoint currently is. Once the per-cell
        # feature grid exists this becomes a real grid-cell prediction and the
        # method changes to "grid_cell".
        "method": "nearest_station_proxy",
        "method_note": ("STUB: returns the model's prediction at the nearest station, not "
                        "at the requested point. Real per-point prediction needs the 1km "
                        "feature grid, which is not built yet."),
    }


@app.get("/api/grid")
def grid(date: str):
    if date not in STATE["dates"]:
        return JSONResponse(status_code=422, content={
            "error": "date_out_of_window",
            "message": (f"This model covers {STATE['dates'][0]} to {STATE['dates'][-1]}."),
        })

    preds = station_predictions_for_date(date)
    if preds is None or len(preds) == 0:
        return JSONResponse(status_code=503, content={"error": "no_rows_for_date"})

    # Grid is seeded over the bbox plus a margin, then filtered to cells inside
    # the coverage cutoff -- so the coloured area on the map IS exactly the area
    # the API will answer for, instead of a rectangle that disagrees with it.
    cells, lat_step, lon_step = build_grid_cells(STATE["grid_km"],
                                                 STATE["max_station_distance_km"])
    slat = preds["latitude"].values
    slon = preds["longitude"].values
    svals = preds["predicted_pm25"].values

    out = []
    for lat, lon in cells:
        d = haversine_km(lat, lon, slat, slon)
        nearest_km = float(d.min())
        if nearest_km > STATE["max_station_distance_km"]:
            continue
        # Inverse distance weighting, power 2. A real interpolation method, but
        # NOT the model predicting at this cell -- the model never saw this
        # cell's own AOD, meteorology or land cover.
        w = 1.0 / np.maximum(d, 0.25) ** 2
        value = float(np.sum(w * svals) / np.sum(w))
        level, _ = confidence_for_distance(nearest_km)
        out.append({
            "lat": lat, "lon": lon,
            "pm25_ugm3": round(value, 1),
            "aqi_band": aqi_band(value),
            "nearest_station_km": round(nearest_km, 2),
            "confidence": level,
        })

    return {
        "date": date,
        "grid_km": STATE["grid_km"],
        "n_cells": len(out),
        "lat_step": round(lat_step, 6),
        "lon_step": round(lon_step, 6),
        "synthetic": True,
        "method": "idw_from_station_predictions",
        "method_note": ("SYNTHETIC SURFACE. Inverse-distance weighting of the model's "
                        "42 station predictions -- a placeholder for UI review. It is NOT "
                        "the model predicting per cell; that needs the per-cell feature "
                        "grid (MAIAC AOD, ERA5, NDVI, land cover per 1km cell), which is "
                        "not built yet."),
        "cells": out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/lightgbm/lightgbm_full_model.txt")
    parser.add_argument("--dataset",
                        default="data/processed/modeling_datasets/lightgbm_ready_dataset.csv")
    parser.add_argument("--station_file",
                        default="data/stations/cpcb_stations_delhi_status.csv")
    parser.add_argument("--metrics",
                        default="reports/lightgbm/primary/cv_spatial_loso_aggregated.json")
    parser.add_argument("--max_station_distance_km",
                        default=DEFAULT_MAX_STATION_DISTANCE_KM, type=float,
                        help="refuse queries further than this from any training station")
    parser.add_argument("--grid_km", default=5.0, type=float,
                        help="grid spacing for the /api/grid surface. 5km while the "
                             "surface is synthetic; 1km once the real feature grid exists")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    load_state(args.model, args.dataset, args.station_file, args.metrics, args.grid_km,
               args.max_station_distance_km)

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
