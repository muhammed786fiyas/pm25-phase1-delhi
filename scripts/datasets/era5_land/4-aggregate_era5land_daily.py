import argparse
import os
import yaml
import pandas as pd


def aggregate_daily(windowed):
    daily = windowed.groupby(["location_id", "name", "date"]).agg(
        temperature_c=("temperature_c", "mean"),
        dewpoint_c=("dewpoint_c", "mean"),
        wind_speed=("wind_speed", "mean"),
        relative_humidity=("relative_humidity", "mean"),
        n_hours_used=("hour", "count"),
    ).reset_index()
    return daily


def apply_min_hours_gate(daily, min_hours_required):
    before = len(daily)
    thin_days = daily[daily["n_hours_used"] < min_hours_required]
    daily = daily[daily["n_hours_used"] >= min_hours_required]

    if len(thin_days) > 0:
        print(f"Dropped {len(thin_days)} station-days below min_hours_required={min_hours_required}")
    print(f"After min_hours_required gate: {before} -> {len(daily)} station-days")
    return daily


def build_station_summary(daily):
    rows = []
    for location_id in sorted(daily["location_id"].unique()):
        station_data = daily[daily["location_id"] == location_id]
        name = station_data["name"].iloc[0]
        rows.append({
            "location_id": location_id,
            "name": name,
            "n_days": len(station_data),
            "date_min": station_data["date"].min(),
            "date_max": station_data["date"].max(),
            "temp_mean_c": round(station_data["temperature_c"].mean(), 2),
            "temp_min_c": round(station_data["temperature_c"].min(), 2),
            "temp_max_c": round(station_data["temperature_c"].max(), 2),
            "rh_mean": round(station_data["relative_humidity"].mean(), 2),
            "rh_min": round(station_data["relative_humidity"].min(), 2),
            "rh_max": round(station_data["relative_humidity"].max(), 2),
            "wind_speed_mean": round(station_data["wind_speed"].mean(), 2),
            "wind_speed_min": round(station_data["wind_speed"].min(), 2),
            "wind_speed_max": round(station_data["wind_speed"].max(), 2),
            "n_hours_used_min": station_data["n_hours_used"].min(),
            "n_hours_used_max": station_data["n_hours_used"].max(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windowed", required=True, help="era5_land_windowed.csv")
    parser.add_argument("--params", required=True, help="params.yaml path")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)

    min_hours_required = params["era5_land"]["daily_aggregation"]["min_hours_required"]

    windowed = pd.read_csv(args.windowed)
    print(f"Loaded {len(windowed)} station-hour rows, {windowed['location_id'].nunique()} stations")

    daily = aggregate_daily(windowed)
    print(f"Aggregated to {len(daily)} station-days")

    daily = apply_min_hours_gate(daily, min_hours_required)

    out_path = os.path.join(args.outdir, "era5_land_daily.csv")
    daily.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    print("=== n_hours_used breakdown ===")
    for n_hours in sorted(daily["n_hours_used"].unique()):
        count = len(daily[daily["n_hours_used"] == n_hours])
        print(f"n_hours_used={n_hours}: {count} station-days")

    summary = build_station_summary(daily)
    summary_path = os.path.join(args.outdir, "era5_land_daily_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Total stations: {summary['location_id'].nunique()}")
    print(f"Expected station-days (365 x 42): {365 * 42}")


if __name__ == "__main__":
    main()


