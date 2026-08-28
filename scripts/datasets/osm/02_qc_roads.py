import argparse
import os
import math
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--max_plausible_density_km_per_km2", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading raw road lengths ===")
    raw_df = pd.read_csv(args.input)
    stations_df = pd.read_csv(args.station_file)
    keep_df = stations_df[stations_df["status"] == "KEEP"]

    print("Stations expected (KEEP):", len(keep_df))
    print("Stations in raw output:", len(raw_df))

    missing_ids = set(keep_df["location_id"]) - set(raw_df["location_id"])
    zero_length_ids = raw_df[raw_df["total_road_length_m"] <= 0]["location_id"].tolist()

    flagged_high_density = []
    for index, row in raw_df.iterrows():
        buffer_radius_m = row["buffer_radius_m"]
        buffer_area_km2 = math.pi * (buffer_radius_m / 1000.0) ** 2
        density_km_per_km2 = (row["total_road_length_m"] / 1000.0) / buffer_area_km2
        if density_km_per_km2 > args.max_plausible_density_km_per_km2:
            flagged_high_density.append({
                "location_id": row["location_id"],
                "name": row["name"],
                "density_km_per_km2": round(density_km_per_km2, 2),
            })

    print("Missing stations:", len(missing_ids))
    if missing_ids:
        print(missing_ids)
    print("Zero-length stations:", len(zero_length_ids))
    if zero_length_ids:
        print(zero_length_ids)
    print("Implausibly high density stations:", len(flagged_high_density))
    for f in flagged_high_density:
        print(f)

    hard_fail = len(missing_ids) > 0

    summary_df = pd.DataFrame({
        "n_expected": [len(keep_df)],
        "n_present": [len(raw_df)],
        "n_missing": [len(missing_ids)],
        "n_zero_length": [len(zero_length_ids)],
        "n_high_density_flagged": [len(flagged_high_density)],
        "hard_fail": [hard_fail],
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print("QC summary saved to:", args.output)

    if hard_fail:
        raise SystemExit("QC HARD FAIL: missing stations in road length output")


if __name__ == "__main__":
    main()
