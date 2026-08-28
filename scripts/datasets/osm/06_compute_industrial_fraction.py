import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading raw industrial area ===")
    raw_df = pd.read_csv(args.input)
    print("Stations loaded:", len(raw_df))

    rows = []
    print("=== Converting to industrial land-use fraction ===")
    for index, row in raw_df.iterrows():
        fraction = row["industrial_area_m2"] / row["buffer_area_m2"]
        rows.append({
            "location_id": row["location_id"],
            "name": row["name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "industrial_landuse_fraction": fraction,
        })
        print("Done:", row["location_id"], row["name"], "-- fraction:", round(fraction, 4))

    output_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)
    print("Total stations:", len(output_df))


if __name__ == "__main__":
    main()
