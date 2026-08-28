import argparse
import os
import math
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading raw road lengths ===")
    raw_df = pd.read_csv(args.input)
    print("Stations loaded:", len(raw_df))

    rows = []
    print("=== Converting to road density (km per km^2) ===")
    for index, row in raw_df.iterrows():
        buffer_radius_m = row["buffer_radius_m"]
        buffer_area_km2 = math.pi * (buffer_radius_m / 1000.0) ** 2
        total_length_km = row["total_road_length_m"] / 1000.0
        density = total_length_km / buffer_area_km2

        rows.append({
            "location_id": row["location_id"],
            "name": row["name"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "road_density_km_per_km2": density,
        })
        print("Done:", row["location_id"], row["name"], "-- density:", round(density, 3))

    output_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)
    print("Total stations:", len(output_df))


if __name__ == "__main__":
    main()
