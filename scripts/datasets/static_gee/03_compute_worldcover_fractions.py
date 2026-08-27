import argparse
import os
import pandas as pd

WORLDCOVER_CLASS_NAMES = {
    "10": "tree_cover",
    "20": "shrubland",
    "30": "grassland",
    "40": "cropland",
    "50": "built_up",
    "60": "bare_sparse_veg",
    "70": "snow_ice",
    "80": "water",
    "90": "wetland_herbaceous",
    "95": "mangroves",
    "100": "moss_lichen",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading raw histogram ===")
    raw_df = pd.read_csv(args.input)
    print("Stations loaded:", len(raw_df))

    rows = []

    print("=== Converting counts to % fractions ===")
    for index, row in raw_df.iterrows():
        location_id = row["location_id"]
        name = row["name"]
        total_pixels = row["total_pixels"]

        output_row = {}
        output_row["location_id"] = location_id
        output_row["name"] = name
        output_row["latitude"] = row["latitude"]
        output_row["longitude"] = row["longitude"]

        for class_code in WORLDCOVER_CLASS_NAMES:
            class_name = WORLDCOVER_CLASS_NAMES[class_code]
            column_name = "class_" + class_code + "_count"
            pct_column_name = class_name + "_pct"

            count = row[column_name]
            if total_pixels > 0:
                output_row[pct_column_name] = (count / total_pixels) * 100
            else:
                output_row[pct_column_name] = 0

        rows.append(output_row)
        print("Done:", location_id, name)

    print("=== Saving processed output ===")
    output_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)
    print("Total stations:", len(output_df))


if __name__ == "__main__":
    main()