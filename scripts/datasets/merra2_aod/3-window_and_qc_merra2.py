import argparse
import os
import pandas as pd
import yaml


def check_coordinate_consistency(hourly_df, mapping_df, tolerance):
    # Confirms the point actually queried in script 2 (longitude/latitude)
    # matches the point script 1 intended (cell_lat/cell_lon) for that cell_id.
    # Catches silent query-point drift (wrong scale, wrong geometry, etc.)
    # before it produces subtly-wrong AOT values downstream.
    cell_coords = mapping_df[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates()
    merged = hourly_df.merge(cell_coords, on="cell_id")

    lat_diff = (merged["latitude"] - merged["cell_lat"]).abs()
    lon_diff = (merged["longitude"] - merged["cell_lon"]).abs()
    mismatches = merged[(lat_diff > tolerance) | (lon_diff > tolerance)]

    if len(mismatches) > 0:
        raise ValueError(
            f"{len(mismatches)} rows have sampled coordinates that do not match "
            f"their intended cell center (tolerance={tolerance} degrees). "
            f"Affected cell_ids: {sorted(mismatches['cell_id'].unique())}"
        )

    print(f"Coordinate consistency check: all {len(merged)} rows match their intended cell center")


def apply_overpass_window(hourly_df, overpass_hours):
    hourly_df["datetime_utc"] = pd.to_datetime(hourly_df["time"], unit="ms", utc=True)
    hourly_df["date"] = hourly_df["datetime_utc"].dt.date
    hourly_df["hour"] = hourly_df["datetime_utc"].dt.hour

    before = len(hourly_df)
    hourly_df = hourly_df[hourly_df["hour"].isin(overpass_hours)]
    print(f"Overpass window filter: {before} -> {len(hourly_df)} rows")
    return hourly_df


def check_duplicates(hourly_df):
    before = len(hourly_df)
    hourly_df = hourly_df.drop_duplicates(subset=["cell_id", "time"])
    dropped = before - len(hourly_df)
    if dropped > 0:
        print(f"Duplicate check: dropped {dropped} duplicate cell_id+time rows")
    else:
        print("Duplicate check: no duplicates found")
    return hourly_df


def check_value_range(hourly_df, band, value_min, value_max):
    before = len(hourly_df)
    out_of_range = hourly_df[(hourly_df[band] < value_min) | (hourly_df[band] > value_max)]
    if len(out_of_range) > 0:
        print(f"Range check: {len(out_of_range)} rows outside [{value_min}, {value_max}], dropping")
    hourly_df = hourly_df[(hourly_df[band] >= value_min) & (hourly_df[band] <= value_max)]
    print(f"Range check: {before} -> {len(hourly_df)} rows")
    return hourly_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Path to merra2_aod_raw.csv")
    parser.add_argument("--mapping", required=True, help="Path to station_cell_mapping.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)["merra2_windowing_qc"]

    aot_band = params["aot_band"]
    overpass_hours = params["overpass_hours_utc"]
    value_min = params["totexttau_min"]
    value_max = params["totexttau_max"]
    coord_tolerance = params["coord_tolerance_degrees"]

    raw_df = pd.read_csv(args.raw)
    mapping_df = pd.read_csv(args.mapping)
    print(f"Loaded {len(raw_df)} raw hourly rows across {raw_df['cell_id'].nunique()} cells")

    check_coordinate_consistency(raw_df, mapping_df, coord_tolerance)

    windowed_df = apply_overpass_window(raw_df, overpass_hours)
    windowed_df = check_duplicates(windowed_df)
    windowed_df = check_value_range(windowed_df, aot_band, value_min, value_max)

    out_path = os.path.join(args.outdir, "merra2_aod_windowed_qc.csv")
    windowed_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()