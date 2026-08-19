import argparse
import os
import yaml
import pandas as pd

with open("params.yaml") as f:
    params = yaml.safe_load(f)

DROP_STATIONS = params["cpcb_filter_aggregate"]["drop_stations"]
MIN_HOURS = params["cpcb_filter_aggregate"]["min_hours"]

SEASON_MONTHS = {
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
    12: "winter", 1: "winter", 2: "winter",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Window-filtered PM2.5 CSV")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    df["dt_local"] = pd.to_datetime(df["datetime_from_local"])

    print(f"Loaded {len(df)} rows, {df['location_id'].nunique()} stations")

    # drop bad stations
    before_rows = len(df)
    before_stations = df["location_id"].nunique()
    df = df[~df["location_id"].isin(DROP_STATIONS)]
    print(f"Dropped stations {DROP_STATIONS}")
    print(f"{before_rows} -> {len(df)} rows, {before_stations} -> {df['location_id'].nunique()} stations")

    # aggregate to daily
    df["date"] = df["dt_local"].dt.date
    daily = df.groupby(["location_id", "date"])["value"].agg(["mean", "count"])
    daily = daily.reset_index()
    daily = daily.rename(columns={"mean": "pm25_daily", "count": "hours_used"})

    print(f"\nTotal station-days before hour filter: {len(daily)}")

    # compare min_hours = 1 vs min_hours = 2
    at_1 = daily[daily["hours_used"] >= 1]
    at_2 = daily[daily["hours_used"] >= 2]
    print(f"station-days with hours_used >= 1: {len(at_1)}")
    print(f"station-days with hours_used >= 2: {len(at_2)}")
    print(f"days lost by requiring 2 instead of 1: {len(at_1) - len(at_2)}")

    print("\nstation-days lost per station (only had 1 hour):")
    only_one = daily[daily["hours_used"] == 1]
    print(only_one.groupby("location_id").size())

    # apply the chosen threshold
    daily_final = daily[daily["hours_used"] >= MIN_HOURS].copy()

    # assign season based on the date's month
    daily_final["month"] = pd.to_datetime(daily_final["date"]).dt.month
    daily_final["season"] = daily_final["month"].map(SEASON_MONTHS)

    print(f"\n=== Station-days per season ===")
    pivot = daily_final.pivot_table(index="location_id", columns="season",
                                     values="pm25_daily", aggfunc="count", fill_value=0)
    season_order = ["summer", "monsoon", "post_monsoon", "winter"]
    pivot = pivot[[c for c in season_order if c in pivot.columns]]
    pivot["total"] = pivot.sum(axis=1)
    pivot_path = os.path.join(args.outdir, "daily_rows_per_station_per_season.csv")
    pivot.to_csv(pivot_path)
    print(f"Wrote {pivot_path}")
    print(pivot)

    daily_final = daily_final.drop(columns=["month"])

    out_path = os.path.join(args.outdir, "pm25_daily.csv")
    daily_final.to_csv(out_path, index=False)
    print(f"\nUsed MIN_HOURS = {MIN_HOURS}")
    print(f"Wrote {len(daily_final)} station-days to {out_path}")

if __name__ == "__main__":
    main()