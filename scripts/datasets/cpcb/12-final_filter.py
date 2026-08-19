import argparse
import os
import pandas as pd

SEASON_MONTHS = {
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
    12: "winter", 1: "winter", 2: "winter",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", required=True, help="Aggregated daily PM2.5 CSV")
    parser.add_argument("--stations", required=True, help="Station status CSV (status column is authoritative)")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    daily = pd.read_csv(args.daily)
    stations = pd.read_csv(args.stations)

    print(f"Loaded {len(daily)} station-days, {daily['location_id'].nunique()} stations in daily file")

    keep_stations = stations[stations["status"] == "KEEP"]["location_id"].tolist()
    print(f"Station status CSV says KEEP for {len(keep_stations)} stations")

    before_rows = len(daily)
    before_stations = daily["location_id"].nunique()

    daily_final = daily[daily["location_id"].isin(keep_stations)]

    print(f"{before_rows} -> {len(daily_final)} rows")
    print(f"{before_stations} -> {daily_final['location_id'].nunique()} stations")

    dropped = set(daily["location_id"].unique()) - set(keep_stations)
    if len(dropped) > 0:
        print(f"Dropped station_ids not marked KEEP: {sorted(dropped)}")

    out_path = os.path.join(args.outdir, "pm25_daily_final.csv")
    daily_final.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    daily_final = daily_final.copy()
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

if __name__ == "__main__":
    main()