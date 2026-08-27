import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading QC-passed raw data ===")
    raw_df = pd.read_csv(args.input)
    print("Stations loaded:", len(raw_df))

    output_df = raw_df[["location_id", "name", "latitude", "longitude", "elevation_m", "slope_deg"]].copy()

    print("=== Saving processed output ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)
    print("Total stations:", len(output_df))


if __name__ == "__main__":
    main()