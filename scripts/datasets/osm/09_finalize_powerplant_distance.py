import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading QC-passed raw power plant distances ===")
    raw_df = pd.read_csv(args.input)
    print("Stations loaded:", len(raw_df))

    output_df = raw_df.copy()
    output_df["dist_to_nearest_powerplant_km"] = output_df["dist_to_nearest_powerplant_m"] / 1000.0
    output_df = output_df[[
        "location_id", "name", "latitude", "longitude",
        "dist_to_nearest_powerplant_km", "nearest_powerplant_name",
    ]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)
    print("Total stations:", len(output_df))


if __name__ == "__main__":
    main()
