"""
QC gate for the per-station season-intercept calibration fits (script 02).
Hard-fails the pipeline if any KEEP station has no usable fit, too few
overlap days to trust, or a non-positive MERRA2 slope (b) -- MAIAC and
MERRA2 AOD should move together; a negative or zero slope means the fit
is not usable regardless of the season offsets.
"""

import argparse
import os
import yaml
import pandas as pd

PARAMS_FILE = "params.yaml"
COEFFICIENT_COLUMNS = ["a", "b", "c_monsoon", "c_post_monsoon", "c_winter"]


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
    params = all_params["maiac_gapfill"]["qc"]

    min_obs_for_fit = params["min_obs_for_fit"]
    min_slope = params["min_slope"]
    r_squared_warn_threshold = params["r_squared_warn_threshold"]

    print("Using min_obs_for_fit:", min_obs_for_fit, "min_slope:", min_slope, "r_squared_warn_threshold:", r_squared_warn_threshold)

    print("=== Loading station calibration fits ===")
    calib_df = pd.read_csv(args.input)

    print("=== Loading stations ===")
    stations_df = pd.read_csv(args.station_file)
    keep_df = stations_df[stations_df["status"] == "KEEP"]
    print("Stations expected (KEEP):", len(keep_df))
    print("Stations in calibration output:", len(calib_df))

    missing_ids = set(keep_df["location_id"]) - set(calib_df["location_id"])

    any_coefficient_null = calib_df[COEFFICIENT_COLUMNS].isna().any(axis=1)
    unfitted_df = calib_df[any_coefficient_null]
    unfitted_ids = set(unfitted_df["location_id"])

    low_obs_df = calib_df[calib_df["n_obs"] < min_obs_for_fit]
    low_obs_ids = set(low_obs_df["location_id"])

    bad_slope_df = calib_df[(calib_df["b"].notna()) & (calib_df["b"] <= min_slope)]
    bad_slope_ids = set(bad_slope_df["location_id"])

    low_r_squared_df = calib_df[(calib_df["r_squared"].notna()) & (calib_df["r_squared"] < r_squared_warn_threshold)]

    print("Missing stations (no row at all):", len(missing_ids))
    if missing_ids:
        print(missing_ids)
    print("Unfitted stations (any coefficient null):", len(unfitted_ids))
    if unfitted_ids:
        print(unfitted_ids)
    print("Stations below min_obs_for_fit:", len(low_obs_ids))
    if low_obs_ids:
        print(low_obs_ids)
    print("Stations with non-positive MERRA2 slope (b):", len(bad_slope_ids))
    if bad_slope_ids:
        print(bad_slope_ids)
    print("Stations flagged for low r_squared (warning only, not hard fail):", len(low_r_squared_df))
    for index, row in low_r_squared_df.iterrows():
        print(" ", row["location_id"], row["name"], "-- r_squared:", round(row["r_squared"], 3))

    hard_fail = (
        len(missing_ids) > 0
        or len(unfitted_ids) > 0
        or len(low_obs_ids) > 0
        or len(bad_slope_ids) > 0
    )

    summary_df = pd.DataFrame({
        "n_expected": [len(keep_df)],
        "n_present": [len(calib_df)],
        "n_missing": [len(missing_ids)],
        "n_unfitted": [len(unfitted_ids)],
        "n_below_min_obs": [len(low_obs_ids)],
        "n_bad_slope": [len(bad_slope_ids)],
        "n_low_r_squared_flagged": [len(low_r_squared_df)],
        "hard_fail": [hard_fail],
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print("QC summary saved to:", args.output)

    if hard_fail:
        raise SystemExit("QC HARD FAIL: one or more stations have a missing, unfitted, too-thin, or non-positive-slope calibration")


if __name__ == "__main__":
    main()
