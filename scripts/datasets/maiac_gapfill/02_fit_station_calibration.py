"""
Fit one season-intercept calibration model per station:
  MAIAC_AOD ~ a + b*MERRA2_AOD + c_monsoon*is_monsoon + c_post_monsoon*is_post_monsoon + c_winter*is_winter
(summer is the reference season, folded into "a"). One shared slope b per
station, but the baseline shifts by season. Chosen over a plain pooled line
and over a full per-season a/b fit after comparing CV RMSE in
06_test_calibration_variants.py -- see docs/logs/tasks/7-MAIAC_Gapfill.md.
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd

PARAMS_FILE = "params.yaml"
SEASON_DUMMY_COLUMNS = ["monsoon", "post_monsoon", "winter"]  # summer is the reference


def fit_season_intercept_model(station_df, aod_column, merra2_column):
    n_obs = len(station_df)

    dummy_matrix = np.column_stack([
        (station_df["season"] == season_name).astype(float).to_numpy()
        for season_name in SEASON_DUMMY_COLUMNS
    ])
    feature_matrix = np.column_stack([station_df[merra2_column].to_numpy(), dummy_matrix])
    design_matrix = np.column_stack([np.ones(n_obs), feature_matrix])

    y_values = station_df[aod_column].to_numpy()
    coefficients, residuals_sum, rank, singular_values = np.linalg.lstsq(design_matrix, y_values, rcond=None)
    a, b, c_monsoon, c_post_monsoon, c_winter = coefficients

    predicted = design_matrix @ coefficients
    residuals = y_values - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r_squared = None if ss_tot == 0 else 1 - (ss_res / ss_tot)

    return a, b, c_monsoon, c_post_monsoon, c_winter, r_squared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--input", required=True)
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["maiac_gapfill"]["calibration"]

    aod_column = params["aod_column"]
    merra2_column = params["merra2_column"]
    print("Fitting:", aod_column, "~ a + b *", merra2_column, "+ season intercepts", SEASON_DUMMY_COLUMNS)

    print("=== Loading calibration dataset ===")
    calib_df = pd.read_csv(args.input)
    print("Total overlap rows:", len(calib_df))

    print("=== Loading stations ===")
    stations_df = pd.read_csv(args.station_file)
    stations_df = stations_df[stations_df["status"] == "KEEP"]
    print("KEEP stations:", len(stations_df))

    rows = []
    print("=== Fitting per-station season-intercept calibration ===")
    for index, station_row in stations_df.iterrows():
        location_id = station_row["location_id"]
        name = station_row["name"]

        station_calib = calib_df[calib_df["location_id"] == location_id]
        n_obs = len(station_calib)

        if n_obs < 8:
            print("SKIPPED (not enough overlap days):", location_id, name, "-- n_obs:", n_obs)
            rows.append({
                "location_id": location_id, "name": name,
                "latitude": station_row["latitude"], "longitude": station_row["longitude"],
                "a": None, "b": None, "c_monsoon": None, "c_post_monsoon": None, "c_winter": None,
                "n_obs": n_obs, "r_squared": None,
            })
            continue

        a, b, c_monsoon, c_post_monsoon, c_winter, r_squared = fit_season_intercept_model(station_calib, aod_column, merra2_column)

        rows.append({
            "location_id": location_id, "name": name,
            "latitude": station_row["latitude"], "longitude": station_row["longitude"],
            "a": a, "b": b, "c_monsoon": c_monsoon, "c_post_monsoon": c_post_monsoon, "c_winter": c_winter,
            "n_obs": n_obs, "r_squared": r_squared,
        })
        print("Done:", location_id, name, "-- a:", round(a, 4), "b:", round(b, 4),
              "c_monsoon:", round(c_monsoon, 4), "c_post_monsoon:", round(c_post_monsoon, 4), "c_winter:", round(c_winter, 4),
              "n_obs:", n_obs, "r2:", round(r_squared, 3) if r_squared is not None else None)

    output_df = pd.DataFrame(rows)

    print("=== Saving station calibration output ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("=== Fit Summary ===")
    valid_df = output_df[output_df["b"].notna()]
    print("Stations fitted:", len(valid_df), "of", len(output_df))
    if len(valid_df) > 0:
        print("b (slope) min/median/max:", round(valid_df["b"].min(), 3), round(valid_df["b"].median(), 3), round(valid_df["b"].max(), 3))
        print("r_squared min/median/max:", round(valid_df["r_squared"].min(), 3), round(valid_df["r_squared"].median(), 3), round(valid_df["r_squared"].max(), 3))


if __name__ == "__main__":
    main()
