import argparse
import os
import pandas as pd
import yaml


def aggregate_to_daily(windowed_df, band, min_hours_required):
    daily = windowed_df.groupby(["cell_id", "date"])[band].agg(["mean", "count"]).reset_index()
    daily = daily.rename(columns={"mean": "totexttau", "count": "n_hours_available"})

    before = len(daily)
    daily = daily[daily["n_hours_available"] >= min_hours_required]
    print(f"Dropped {before - len(daily)} cell-days with fewer than {min_hours_required} overpass hours")

    return daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windowed", required=True, help="Path to merra2_aod_windowed_qc.csv")
    parser.add_argument("--mapping", required=True, help="Path to station_cell_mapping.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)["merra2_aggregation"]

    aot_band = params["aot_band"]
    min_hours_required = params["min_hours_required"]

    windowed_df = pd.read_csv(args.windowed)
    print(f"Loaded {len(windowed_df)} windowed/QC'd hourly rows")

    daily_df = aggregate_to_daily(windowed_df, aot_band, min_hours_required)

    mapping_df = pd.read_csv(args.mapping)
    station_daily_df = mapping_df.merge(daily_df, on="cell_id")

    out_path = os.path.join(args.outdir, "merra2_aod_daily.csv")
    station_daily_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"{station_daily_df['location_id'].nunique()} stations, {len(station_daily_df)} station-day rows")


if __name__ == "__main__":
    main()