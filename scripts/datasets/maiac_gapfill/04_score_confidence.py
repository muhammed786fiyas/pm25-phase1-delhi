"""
Score confidence in each station's season-intercept gap-fill calibration,
broken out by season. Method: leave-one-out CV over each station's overlap
days (refit the season-intercept model on n-1 days, predict the held-out
day, record the error), then group the held-out errors by season -- this
measures how well the real production calibration (one fit per station,
season-shifted intercept) performs when the true value falls in each
season. Runs before the fill step so confidence can be merged into the
final gap-filled output.

Confidence bucket thresholds (RMSE cutoffs) are read from params.yaml and
start as "NOT_SET" -- run once to see the printed RMSE distribution across
all station x season groups, decide real cutoffs, then rerun. These
thresholds need to be re-derived whenever the underlying calibration model
changes (they were re-derived here after promoting season-intercept).
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

PARAMS_FILE = "params.yaml"
SEASON_DUMMY_COLUMNS = ["monsoon", "post_monsoon", "winter"]  # summer is the reference


def fit_season_intercept_model(x_merra2, dummy_matrix, y_values):
    n_obs = len(y_values)
    feature_matrix = np.column_stack([x_merra2, dummy_matrix])
    design_matrix = np.column_stack([np.ones(n_obs), feature_matrix])
    coefficients, residuals_sum, rank, singular_values = np.linalg.lstsq(design_matrix, y_values, rcond=None)
    return coefficients  # [a, b, c_monsoon, c_post_monsoon, c_winter]


def predict_season_intercept(coefficients, x_merra2_value, season_dummies):
    features = np.array([1.0, x_merra2_value] + season_dummies)
    return features @ coefficients


def run_loocv_for_station(station_df, aod_column, merra2_column):
    station_df = station_df.reset_index(drop=True)
    n_obs = len(station_df)

    dummy_matrix_full = np.column_stack([
        (station_df["season"] == season_name).astype(float).to_numpy()
        for season_name in SEASON_DUMMY_COLUMNS
    ])

    residual_rows = []
    for holdout_index in range(n_obs):
        train_mask = np.ones(n_obs, dtype=bool)
        train_mask[holdout_index] = False

        x_train = station_df.loc[train_mask, merra2_column].to_numpy()
        y_train = station_df.loc[train_mask, aod_column].to_numpy()
        dummy_train = dummy_matrix_full[train_mask]

        coefficients = fit_season_intercept_model(x_train, dummy_train, y_train)

        holdout_row = station_df.loc[holdout_index]
        holdout_dummies = dummy_matrix_full[holdout_index].tolist()
        predicted = predict_season_intercept(coefficients, holdout_row[merra2_column], holdout_dummies)
        actual = holdout_row[aod_column]
        residual = actual - predicted

        residual_rows.append({
            "location_id": holdout_row["location_id"],
            "date": holdout_row["date"],
            "season": holdout_row["season"],
            "actual": actual,
            "predicted": predicted,
            "residual": residual,
        })

    return residual_rows


def assign_bucket(rmse, low_threshold, high_threshold):
    # lower RMSE = higher confidence
    if rmse <= low_threshold:
        return "high"
    elif rmse <= high_threshold:
        return "medium"
    else:
        return "low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--input", required=True)
    parser.add_argument("--residuals_output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    calib_params = all_params["maiac_gapfill"]["calibration"]
    confidence_params = all_params["maiac_gapfill"]["confidence"]

    aod_column = calib_params["aod_column"]
    merra2_column = calib_params["merra2_column"]
    rmse_low_threshold = confidence_params["rmse_low_threshold"]
    rmse_high_threshold = confidence_params["rmse_high_threshold"]

    print("=== Loading calibration dataset ===")
    calib_df = pd.read_csv(args.input)
    print("Total overlap rows:", len(calib_df))

    print("=== Running leave-one-out CV per station (season-intercept model) ===")
    all_residual_rows = []
    for location_id, station_df in calib_df.groupby("location_id"):
        if len(station_df) < 8:
            print("SKIPPED (too few points for LOOCV):", location_id)
            continue
        station_residuals = run_loocv_for_station(station_df, aod_column, merra2_column)
        all_residual_rows.extend(station_residuals)
        print("Done:", location_id, "-- n_obs:", len(station_df))

    residuals_df = pd.DataFrame(all_residual_rows)

    print("=== Saving raw LOOCV residuals ===")
    os.makedirs(os.path.dirname(args.residuals_output), exist_ok=True)
    residuals_df.to_csv(args.residuals_output, index=False)
    print("Saved to:", args.residuals_output)

    print("=== Computing RMSE / MAE per station x season ===")
    score_rows = []
    for (location_id, season), group_df in residuals_df.groupby(["location_id", "season"]):
        rmse = np.sqrt(np.mean(group_df["residual"] ** 2))
        mae = np.mean(np.abs(group_df["residual"]))
        score_rows.append({
            "location_id": location_id,
            "season": season,
            "n_obs": len(group_df),
            "rmse": rmse,
            "mae": mae,
        })

    scores_df = pd.DataFrame(score_rows)

    print("=== RMSE distribution across all station x season groups ===")
    print(scores_df["rmse"].describe())
    print("")
    print("By season:")
    print(scores_df.groupby("season")["rmse"].describe())
    print("")
    print("Tercile cutoffs (33rd, 67th pct):", scores_df["rmse"].quantile([0.333, 0.667]).to_numpy())

    if rmse_low_threshold == "NOT_SET" or rmse_high_threshold == "NOT_SET":
        print("")
        print("ERROR: rmse_low_threshold / rmse_high_threshold are NOT_SET in params.yaml.")
        print("Review the RMSE distribution printed above, decide real cutoffs, then rerun.")
        sys.exit(1)

    print("Using rmse_low_threshold:", rmse_low_threshold, "rmse_high_threshold:", rmse_high_threshold)
    scores_df["confidence_bucket"] = scores_df["rmse"].apply(
        lambda rmse_value: assign_bucket(rmse_value, rmse_low_threshold, rmse_high_threshold)
    )

    print("=== Saving confidence scores ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scores_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("=== Confidence bucket counts by season ===")
    print(scores_df.groupby(["season", "confidence_bucket"]).size())


if __name__ == "__main__":
    main()
