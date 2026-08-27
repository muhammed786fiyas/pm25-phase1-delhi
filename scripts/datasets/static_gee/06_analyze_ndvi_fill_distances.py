import argparse
import os
import pandas as pd


def find_fill_distance(period_index, valid_periods_for_station):
    backward_candidates = [p for p in valid_periods_for_station if p <= period_index]
    forward_candidates = [p for p in valid_periods_for_station if p >= period_index]

    backward_distance = None
    if len(backward_candidates) > 0:
        nearest_backward = max(backward_candidates)
        backward_distance = period_index - nearest_backward

    forward_distance = None
    if len(forward_candidates) > 0:
        nearest_forward = min(forward_candidates)
        forward_distance = nearest_forward - period_index

    if backward_distance is None and forward_distance is None:
        return None, None

    if backward_distance is None:
        return forward_distance, "forward"

    if forward_distance is None:
        return backward_distance, "backward"

    if backward_distance <= forward_distance:
        return backward_distance, "backward"
    else:
        return forward_distance, "forward"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading raw NDVI periods ===")
    raw_df = pd.read_csv(args.input)
    print("Total rows:", len(raw_df))

    analysis_rows = []
    unfillable_count = 0

    print("=== Analyzing fill distance per station ===")
    station_ids = raw_df["location_id"].unique()

    for location_id in station_ids:
        station_df = raw_df[raw_df["location_id"] == location_id].sort_values("period_index")

        valid_periods = station_df[station_df["ndvi_mean"].notna()]["period_index"].tolist()
        null_periods = station_df[station_df["ndvi_mean"].isna()]["period_index"].tolist()

        for period_index in null_periods:
            distance, direction = find_fill_distance(period_index, valid_periods)

            if distance is None:
                unfillable_count = unfillable_count + 1
                analysis_rows.append({
                    "location_id": location_id,
                    "period_index": period_index,
                    "fill_distance_periods": None,
                    "fill_direction": None,
                    "unfillable": True,
                })
            else:
                analysis_rows.append({
                    "location_id": location_id,
                    "period_index": period_index,
                    "fill_distance_periods": distance,
                    "fill_direction": direction,
                    "unfillable": False,
                })

    analysis_df = pd.DataFrame(analysis_rows)

    print("=== Saving analysis output ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    analysis_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("=== Fill Distance Summary ===")
    print("Total nulls analyzed:", len(analysis_df))
    print("Unfillable (no valid period at all for that station):", unfillable_count)

    fillable_df = analysis_df[analysis_df["unfillable"] == False]
    if len(fillable_df) > 0:
        print("Fill distance min:", fillable_df["fill_distance_periods"].min())
        print("Fill distance max:", fillable_df["fill_distance_periods"].max())
        print("Fill distance mean:", round(fillable_df["fill_distance_periods"].mean(), 2))
        print("")
        print("Distribution (periods away : count of nulls):")
        distance_counts = fillable_df["fill_distance_periods"].value_counts().sort_index()
        for distance_value, count in distance_counts.items():
            print(" ", distance_value, "period(s) away:", count, "nulls")


if __name__ == "__main__":
    main()