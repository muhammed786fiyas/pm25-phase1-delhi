import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--edge_effect_distance_km_threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading raw power plant distances ===")
    raw_df = pd.read_csv(args.input)
    stations_df = pd.read_csv(args.station_file)
    keep_df = stations_df[stations_df["status"] == "KEEP"]

    print("Stations expected (KEEP):", len(keep_df))
    print("Stations in raw output:", len(raw_df))

    missing_ids = set(keep_df["location_id"]) - set(raw_df["location_id"])

    threshold_m = args.edge_effect_distance_km_threshold * 1000
    edge_flagged = raw_df[raw_df["dist_to_nearest_powerplant_m"] > threshold_m][
        ["location_id", "name", "dist_to_nearest_powerplant_m"]
    ].to_dict("records")

    print("Missing stations:", len(missing_ids))
    if missing_ids:
        print(missing_ids)
    print("Stations near/beyond the download extent (possible edge effect -- true nearest plant may be outside the downloaded area):", len(edge_flagged))
    for f in edge_flagged:
        print(f)

    hard_fail = len(missing_ids) > 0

    summary_df = pd.DataFrame({
        "n_expected": [len(keep_df)],
        "n_present": [len(raw_df)],
        "n_missing": [len(missing_ids)],
        "n_edge_effect_flagged": [len(edge_flagged)],
        "hard_fail": [hard_fail],
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print("QC summary saved to:", args.output)

    if hard_fail:
        raise SystemExit("QC HARD FAIL: missing stations in power plant distance output")


if __name__ == "__main__":
    main()
