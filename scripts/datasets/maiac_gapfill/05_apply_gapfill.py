"""
Build the full station x date calendar grid for the study window and fill
MAIAC AOD gaps using each station's season-intercept calibration
(a + b*MERRA2 + season offset) on days MAIAC is missing but MERRA-2 has a
value. Every row is tagged gap_filled (0 = observed MAIAC, 1 = filled or
still missing), and every row carries the confidence score for its
station x season (from script 04) -- "observed" / null for real MAIAC
readings, since confidence describes the fill, not a real measurement.
"""

import argparse
import os
import yaml
import pandas as pd

PARAMS_FILE = "params.yaml"

WINDOW_START = "2025-03-01"
WINDOW_END = "2026-02-28"

SEASON_MONTHS = {
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
    12: "winter", 1: "winter", 2: "winter",
}

SEASON_OFFSET_COLUMNS = {
    "monsoon": "c_monsoon",
    "post_monsoon": "c_post_monsoon",
    "winter": "c_winter",
    "summer": None,  # summer is the reference season, offset is 0
}


def build_station_date_grid(stations_df):
    date_range = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    dates_df = pd.DataFrame({"date": date_range.strftime("%Y-%m-%d")})

    stations_df = stations_df[["location_id", "name", "latitude", "longitude"]].copy()
    stations_df["_join_key"] = 1
    dates_df["_join_key"] = 1

    grid_df = pd.merge(stations_df, dates_df, on="_join_key").drop(columns=["_join_key"])
    grid_df["month"] = pd.to_datetime(grid_df["date"]).dt.month
    grid_df["season"] = grid_df["month"].map(SEASON_MONTHS)
    grid_df = grid_df.drop(columns=["month"])
    return grid_df


def season_offset_for_row(row):
    offset_column = SEASON_OFFSET_COLUMNS[row["season"]]
    if offset_column is None:
        return 0.0
    return row[offset_column]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--maiac_input", required=True)
    parser.add_argument("--merra2_input", required=True)
    parser.add_argument("--calibration_input", required=True)
    parser.add_argument("--confidence_input", required=True)
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["maiac_gapfill"]["fill"]

    aod_column = params["aod_column"]
    merra2_column = params["merra2_column"]

    print("=== Loading stations ===")
    stations_df = pd.read_csv(args.station_file)
    stations_df = stations_df[stations_df["status"] == "KEEP"]
    print("KEEP stations:", len(stations_df))

    print("=== Building full station x date grid ===")
    grid_df = build_station_date_grid(stations_df)
    print("Grid rows (stations x days):", len(grid_df))

    print("=== Loading MAIAC observed values ===")
    maiac_df = pd.read_csv(args.maiac_input)
    maiac_slim = maiac_df[["location_id", "date", aod_column]].rename(columns={aod_column: "aod_055_observed"})

    print("=== Loading MERRA-2 fill source ===")
    merra2_df = pd.read_csv(args.merra2_input)
    merra2_slim = merra2_df[["location_id", "date", merra2_column]].rename(columns={merra2_column: "totexttau"})

    print("=== Loading station calibration (season-intercept coefficients) ===")
    calib_df = pd.read_csv(args.calibration_input)
    calib_slim = calib_df[["location_id", "a", "b", "c_monsoon", "c_post_monsoon", "c_winter"]]

    print("=== Loading confidence scores ===")
    confidence_df = pd.read_csv(args.confidence_input)
    confidence_slim = confidence_df[["location_id", "season", "rmse", "confidence_bucket"]].rename(columns={"rmse": "confidence_rmse"})

    print("=== Merging ===")
    merged_df = pd.merge(grid_df, maiac_slim, on=["location_id", "date"], how="left")
    merged_df = pd.merge(merged_df, merra2_slim, on=["location_id", "date"], how="left")
    merged_df = pd.merge(merged_df, calib_slim, on="location_id", how="left")
    merged_df = pd.merge(merged_df, confidence_slim, on=["location_id", "season"], how="left")

    print("=== Applying gap-fill ===")
    aod_values = []
    gap_filled_flags = []
    fill_sources = []
    output_confidence_bucket = []
    output_confidence_rmse = []

    coefficient_columns = ["a", "b", "c_monsoon", "c_post_monsoon", "c_winter"]

    for index, row in merged_df.iterrows():
        if pd.notna(row["aod_055_observed"]):
            aod_values.append(row["aod_055_observed"])
            gap_filled_flags.append(0)
            fill_sources.append("maiac_observed")
            output_confidence_bucket.append("observed")
            output_confidence_rmse.append(None)
        elif pd.notna(row["totexttau"]) and row[coefficient_columns].notna().all():
            season_offset = season_offset_for_row(row)
            filled_value = row["a"] + row["b"] * row["totexttau"] + season_offset
            aod_values.append(filled_value)
            gap_filled_flags.append(1)
            fill_sources.append("merra2_calibrated")
            output_confidence_bucket.append(row["confidence_bucket"] if pd.notna(row["confidence_bucket"]) else "unknown")
            output_confidence_rmse.append(row["confidence_rmse"])
        else:
            aod_values.append(None)
            gap_filled_flags.append(1)
            fill_sources.append("unfillable")
            output_confidence_bucket.append("unfillable")
            output_confidence_rmse.append(None)

    merged_df["aod_055"] = aod_values
    merged_df["gap_filled"] = gap_filled_flags
    merged_df["fill_source"] = fill_sources
    merged_df["confidence_bucket"] = output_confidence_bucket
    merged_df["confidence_rmse"] = output_confidence_rmse

    output_columns = [
        "location_id", "name", "latitude", "longitude", "date", "season",
        "aod_055", "gap_filled", "fill_source", "confidence_bucket", "confidence_rmse",
    ]
    output_df = merged_df[output_columns]

    print("=== Saving gap-filled dataset ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("=== Building per-station fill summary ===")
    summary_rows = []
    for location_id, station_df in output_df.groupby("location_id"):
        n_total = len(station_df)
        n_observed = (station_df["fill_source"] == "maiac_observed").sum()
        n_filled = (station_df["fill_source"] == "merra2_calibrated").sum()
        n_unfillable = (station_df["fill_source"] == "unfillable").sum()
        summary_rows.append({
            "location_id": location_id,
            "n_total": n_total,
            "n_observed": n_observed,
            "n_filled": n_filled,
            "n_unfillable": n_unfillable,
            "pct_filled": round(100.0 * n_filled / n_total, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    print("Summary saved to:", args.summary_output)

    print("=== Fill Summary (all stations) ===")
    n_total = len(output_df)
    n_observed = (output_df["fill_source"] == "maiac_observed").sum()
    n_filled = (output_df["fill_source"] == "merra2_calibrated").sum()
    n_unfillable = (output_df["fill_source"] == "unfillable").sum()
    print("Total station-days:", n_total)
    print("Observed (MAIAC):", n_observed, "-- {:.1f}%".format(100.0 * n_observed / n_total))
    print("Filled (MERRA-2 calibrated):", n_filled, "-- {:.1f}%".format(100.0 * n_filled / n_total))
    print("Unfillable (both missing):", n_unfillable, "-- {:.1f}%".format(100.0 * n_unfillable / n_total))

    print("=== Confidence bucket counts among filled rows ===")
    filled_df = output_df[output_df["fill_source"] == "merra2_calibrated"]
    print(filled_df["confidence_bucket"].value_counts())


if __name__ == "__main__":
    main()
