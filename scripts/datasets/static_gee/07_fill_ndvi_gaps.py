import argparse
import os
import sys
import yaml
import pandas as pd

PARAMS_FILE = "params.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--raw_input", required=True)
    parser.add_argument("--distance_input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["static_gee_layers"]["ndvi"]["fill"]

    max_distance = params["max_fill_distance_periods"]

    if max_distance == "NOT_SET":
        print("ERROR: max_fill_distance_periods is NOT_SET in params.yaml.")
        print("Review data/interim/static_gee/ndvi/ndvi_fill_distance_analysis.csv and set a cap before running this stage.")
        sys.exit(1)

    print("Using max_fill_distance_periods:", max_distance)

    print("=== Loading raw NDVI periods ===")
    raw_df = pd.read_csv(args.raw_input)
    print("Total rows:", len(raw_df))

    print("=== Loading fill distance analysis ===")
    distance_df = pd.read_csv(args.distance_input)
    print("Total nulls in analysis:", len(distance_df))

    # Build a lookup: (location_id, period_index) -> ndvi_mean, for all valid values
    lookup = {}
    for index, row in raw_df.iterrows():
        if pd.notna(row["ndvi_mean"]):
            key = (row["location_id"], row["period_index"])
            lookup[key] = row["ndvi_mean"]

    filled_values = {}
    over_cap_count = 0
    unfillable_count = 0

    print("=== Applying fill ===")
    for index, row in distance_df.iterrows():
        location_id = row["location_id"]
        period_index = row["period_index"]

        if row["unfillable"] == True:
            unfillable_count = unfillable_count + 1
            continue

        distance = row["fill_distance_periods"]
        direction = row["fill_direction"]

        if distance > max_distance:
            over_cap_count = over_cap_count + 1
            continue

        if direction == "backward":
            source_period = period_index - distance
        else:
            source_period = period_index + distance

        source_key = (location_id, source_period)
        if source_key in lookup:
            filled_values[(location_id, period_index)] = {
                "ndvi_mean": lookup[source_key],
                "fill_distance_periods": distance,
            }

    print("Nulls filled:", len(filled_values))
    print("Nulls skipped (over cap of", max_distance, "):", over_cap_count)
    print("Nulls unfillable (no valid period at all):", unfillable_count)

    print("=== Building final output ===")
    output_rows = []
    for index, row in raw_df.iterrows():
        location_id = row["location_id"]
        period_index = row["period_index"]

        output_row = {}
        output_row["location_id"] = location_id
        output_row["name"] = row["name"]
        output_row["latitude"] = row["latitude"]
        output_row["longitude"] = row["longitude"]
        output_row["period_index"] = period_index
        output_row["period_start"] = row["period_start"]
        output_row["period_end"] = row["period_end"]

        if pd.notna(row["ndvi_mean"]):
            output_row["ndvi_mean"] = row["ndvi_mean"]
            output_row["gap_filled"] = 0
            output_row["fill_distance_periods"] = 0
        else:
            key = (location_id, period_index)
            if key in filled_values:
                output_row["ndvi_mean"] = filled_values[key]["ndvi_mean"]
                output_row["gap_filled"] = 1
                output_row["fill_distance_periods"] = filled_values[key]["fill_distance_periods"]
            else:
                output_row["ndvi_mean"] = None
                output_row["gap_filled"] = 1
                output_row["fill_distance_periods"] = None

        output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)

    print("=== Saving final NDVI output ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    remaining_nulls = output_df["ndvi_mean"].isna().sum()
    print("Remaining nulls after fill:", remaining_nulls)
    print("Total rows:", len(output_df))


if __name__ == "__main__":
    main()