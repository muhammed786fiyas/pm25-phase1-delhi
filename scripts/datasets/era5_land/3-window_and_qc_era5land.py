import argparse
import os
import yaml
import pandas as pd
import numpy as np


def check_coordinate_consistency(merged, tolerance):
    lon_diff = (merged["longitude"] - merged["cell_lon"]).abs()
    lat_diff = (merged["latitude"] - merged["cell_lat"]).abs()
    bad_rows = merged[(lon_diff > tolerance) | (lat_diff > tolerance)]

    if len(bad_rows) > 0:
        print(f"Found {len(bad_rows)} rows exceeding {tolerance} degree tolerance")
        raise ValueError("Coordinate mismatch - some rows drifted from their assigned cell center")

    print(f"Coordinate check passed: all rows within {tolerance} degrees of cell center")


def run_qc_checks(merged, temp_min_c, temp_max_c, wind_component_max_ms):
    temp_bad = merged[(merged["temperature_c"] < temp_min_c) | (merged["temperature_c"] > temp_max_c)]
    if len(temp_bad) > 0:
        print(f"Found {len(temp_bad)} rows with temperature outside {temp_min_c}-{temp_max_c} C")
        raise ValueError("Temperature QC failed")

    dewpoint_bad = merged[merged["dewpoint_c"] > merged["temperature_c"]]
    if len(dewpoint_bad) > 0:
        print(f"Found {len(dewpoint_bad)} rows where dewpoint exceeds temperature")
        raise ValueError("Dewpoint QC failed")

    wind_bad = merged[
        (merged["u_component_of_wind_10m"].abs() > wind_component_max_ms)
        | (merged["v_component_of_wind_10m"].abs() > wind_component_max_ms)
    ]
    if len(wind_bad) > 0:
        print(f"Found {len(wind_bad)} rows with wind component exceeding {wind_component_max_ms} m/s")
        raise ValueError("Wind QC failed")

    print("All QC checks passed")


def compute_derived_features(merged):
    merged["wind_speed"] = np.sqrt(
        merged["u_component_of_wind_10m"] ** 2 + merged["v_component_of_wind_10m"] ** 2
    )

    merged["relative_humidity"] = 100 * (
        np.exp((17.625 * merged["dewpoint_c"]) / (243.04 + merged["dewpoint_c"]))
        / np.exp((17.625 * merged["temperature_c"]) / (243.04 + merged["temperature_c"]))
    )
    return merged


def build_station_summary(final):
    # one row per station so you can eyeball whether any single station
    # looks off (wrong cell, weird QC pattern) before moving on
    rows = []
    for location_id in sorted(final["location_id"].unique()):
        station_data = final[final["location_id"] == location_id]
        name = station_data["name"].iloc[0]
        rows.append({
            "location_id": location_id,
            "name": name,
            "n_rows": len(station_data),
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
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="era5_land_raw.csv")
    parser.add_argument("--mapping", required=True, help="station_cell_mapping.csv")
    parser.add_argument("--params", required=True, help="params.yaml path")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)

    qc_params = params["era5_land"]["window_qc"]
    tolerance = qc_params["tolerance_degrees"]
    window_hours = qc_params["window_hours"]
    temp_min_c = qc_params["temp_min_c"]
    temp_max_c = qc_params["temp_max_c"]
    wind_component_max_ms = qc_params["wind_component_max_ms"]

    raw = pd.read_csv(args.raw)
    mapping = pd.read_csv(args.mapping)
    print(f"Loaded {len(raw)} raw cell-hour rows, {mapping['location_id'].nunique()} stations")

    mapping_cols = mapping[["location_id", "name", "cell_id", "cell_lat", "cell_lon"]]
    merged = raw.merge(mapping_cols, on="cell_id")
    print(f"After fan-out to station level: {len(merged)} rows")

    check_coordinate_consistency(merged, tolerance)

    merged["hour"] = pd.to_datetime(merged["datetime_utc"]).dt.hour
    before = len(merged)
    merged = merged[merged["hour"].isin(window_hours)]
    print(f"Filtered to overpass window hours {window_hours}: {before} -> {len(merged)} rows")

    merged["temperature_c"] = merged["temperature_2m"] - 273.15
    merged["dewpoint_c"] = merged["dewpoint_temperature_2m"] - 273.15

    run_qc_checks(merged, temp_min_c, temp_max_c, wind_component_max_ms)

    merged = compute_derived_features(merged)
    merged["date"] = pd.to_datetime(merged["datetime_utc"]).dt.date

    output_cols = [
        "location_id", "name", "cell_id", "date", "hour",
        "temperature_c", "dewpoint_c", "wind_speed", "relative_humidity",
    ]
    final = merged[output_cols]

    out_path = os.path.join(args.outdir, "era5_land_windowed.csv")
    final.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Total rows: {len(final)}")
    print(f"Stations: {final['location_id'].nunique()}")

    summary = build_station_summary(final)
    summary_path = os.path.join(args.outdir, "era5_land_window_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"=== Overall ranges across all stations ===")
    print(f"Temp: {summary['temp_min_c'].min()} to {summary['temp_max_c'].max()} C")
    print(f"RH: {summary['rh_min'].min()} to {summary['rh_max'].max()}")
    print(f"Wind speed: {summary['wind_speed_min'].min()} to {summary['wind_speed_max'].max()} m/s")


if __name__ == "__main__":
    main()