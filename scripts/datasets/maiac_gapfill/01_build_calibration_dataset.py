"""
Build the MAIAC-MERRA2 calibration dataset: one row per (station, date) where
BOTH MAIAC AOD (aod_055) and MERRA-2 AOD (totexttau) have a value. This is the
sample used downstream to fit each station's calibration line.
"""

import argparse
import os
import pandas as pd


def load_stations(station_file):
    df = pd.read_csv(station_file)
    df = df[df["status"] == "KEEP"].copy()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maiac_input", required=True)
    parser.add_argument("--merra2_input", required=True)
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    print("=== Loading stations ===")
    stations_df = load_stations(args.station_file)
    keep_ids = set(stations_df["location_id"])
    print("KEEP stations:", len(keep_ids))

    print("=== Loading MAIAC daily AOD ===")
    maiac_df = pd.read_csv(args.maiac_input)
    maiac_df = maiac_df[maiac_df["location_id"].isin(keep_ids)].copy()
    print("MAIAC rows (KEEP stations):", len(maiac_df))

    print("=== Loading MERRA-2 daily AOD ===")
    merra2_df = pd.read_csv(args.merra2_input)
    merra2_df = merra2_df[merra2_df["location_id"].isin(keep_ids)].copy()
    print("MERRA-2 rows (KEEP stations):", len(merra2_df))

    print("=== Joining on [location_id, date] ===")
    maiac_slim = maiac_df[["location_id", "date", "season", "aod_055"]]
    merra2_slim = merra2_df[["location_id", "date", "cell_id", "totexttau"]]

    joined_df = pd.merge(maiac_slim, merra2_slim, on=["location_id", "date"], how="inner")
    print("Overlap rows (both products present):", len(joined_df))

    joined_df = pd.merge(
        joined_df,
        stations_df[["location_id", "name", "latitude", "longitude"]],
        on="location_id",
        how="left",
    )

    column_order = [
        "location_id", "name", "latitude", "longitude", "cell_id",
        "date", "season", "aod_055", "totexttau",
    ]
    joined_df = joined_df[column_order]

    print("=== Saving calibration dataset ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joined_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("=== Building per-station overlap summary ===")
    summary_rows = []
    for location_id in sorted(keep_ids):
        station_rows = joined_df[joined_df["location_id"] == location_id]
        summary_rows.append({
            "location_id": location_id,
            "n_overlap_days": len(station_rows),
        })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    print("Summary saved to:", args.summary_output)

    print("=== Overlap Summary ===")
    print("Stations with zero overlap:", (summary_df["n_overlap_days"] == 0).sum())
    print("Min overlap days:", summary_df["n_overlap_days"].min())
    print("Median overlap days:", summary_df["n_overlap_days"].median())
    print("Max overlap days:", summary_df["n_overlap_days"].max())


if __name__ == "__main__":
    main()
