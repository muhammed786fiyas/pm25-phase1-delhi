import argparse
import os
import yaml
import pandas as pd


def check_coordinate_consistency(merged, tolerance):
    lon_diff = (merged["longitude"] - merged["cell_lon"]).abs()
    lat_diff = (merged["latitude"] - merged["cell_lat"]).abs()
    bad_rows = merged[(lon_diff > tolerance) | (lat_diff > tolerance)]

    if len(bad_rows) > 0:
        print(f"Found {len(bad_rows)} rows exceeding {tolerance} degree tolerance")
        raise ValueError("Coordinate mismatch - some rows drifted from their assigned cell center")

    print(f"Coordinate check passed: all rows within {tolerance} degrees of cell center")


def run_qc_checks(merged, blh_min_m, blh_max_m):
    blh_bad = merged[(merged["boundary_layer_height"] < blh_min_m) | (merged["boundary_layer_height"] > blh_max_m)]
    if len(blh_bad) > 0:
        print(f"Found {len(blh_bad)} rows with BLH outside {blh_min_m}-{blh_max_m} m")
        raise ValueError("BLH QC failed")

    print("All QC checks passed")


def build_station_summary(final):
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
            "blh_mean_m": round(station_data["boundary_layer_height"].mean(), 2),
            "blh_min_m": round(station_data["boundary_layer_height"].min(), 2),
            "blh_max_m": round(station_data["boundary_layer_height"].max(), 2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="era5_blh_raw.csv")
    parser.add_argument("--mapping", required=True, help="station_cell_mapping.csv")
    parser.add_argument("--params", required=True, help="params.yaml path")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)

    qc_params = params["era5_blh"]["window_qc"]
    tolerance = qc_params["tolerance_degrees"]
    window_hours = qc_params["window_hours"]
    blh_min_m = qc_params["blh_min_m"]
    blh_max_m = qc_params["blh_max_m"]

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

    run_qc_checks(merged, blh_min_m, blh_max_m)

    merged["date"] = pd.to_datetime(merged["datetime_utc"]).dt.date

    output_cols = [
        "location_id", "name", "cell_id", "date", "hour", "boundary_layer_height",
    ]
    final = merged[output_cols]

    out_path = os.path.join(args.outdir, "era5_blh_windowed.csv")
    final.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Total rows: {len(final)}")
    print(f"Stations: {final['location_id'].nunique()}")

    summary = build_station_summary(final)
    summary_path = os.path.join(args.outdir, "era5_blh_window_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"=== Overall BLH range across all stations ===")
    print(f"BLH: {summary['blh_min_m'].min()} to {summary['blh_max_m'].max()} m")


if __name__ == "__main__":
    main()