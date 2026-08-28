import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading raw industrial area ===")
    raw_df = pd.read_csv(args.input)
    stations_df = pd.read_csv(args.station_file)
    keep_df = stations_df[stations_df["status"] == "KEEP"]

    print("Stations expected (KEEP):", len(keep_df))
    print("Stations in raw output:", len(raw_df))

    missing_ids = set(keep_df["location_id"]) - set(raw_df["location_id"])

    over_area_ids = []
    for index, row in raw_df.iterrows():
        if row["industrial_area_m2"] > row["buffer_area_m2"] * 1.01:
            over_area_ids.append(row["location_id"])

    zero_industrial_ids = raw_df[raw_df["industrial_area_m2"] == 0]["location_id"].tolist()

    print("Missing stations:", len(missing_ids))
    if missing_ids:
        print(missing_ids)
    print("Stations with industrial area > buffer area (geometry bug):", len(over_area_ids))
    if over_area_ids:
        print(over_area_ids)
    print("Stations with zero industrial area (expected for many residential stations):", len(zero_industrial_ids))

    hard_fail = len(missing_ids) > 0 or len(over_area_ids) > 0

    summary_df = pd.DataFrame({
        "n_expected": [len(keep_df)],
        "n_present": [len(raw_df)],
        "n_missing": [len(missing_ids)],
        "n_over_area_bug": [len(over_area_ids)],
        "n_zero_industrial": [len(zero_industrial_ids)],
        "hard_fail": [hard_fail],
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print("QC summary saved to:", args.output)

    if hard_fail:
        raise SystemExit("QC HARD FAIL: missing stations or geometry-area bug in industrial output")


if __name__ == "__main__":
    main()
