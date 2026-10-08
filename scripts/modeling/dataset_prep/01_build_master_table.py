import argparse
import os
import pandas as pd

# columns to drop from each source table before merging, so we don't carry
# duplicate name/latitude/longitude columns from every table into the master
STATION_META_COLS = ["name", "latitude", "longitude"]

# Copied verbatim from scripts/datasets/cpcb/6-trim_and_season.py, which is
# where season is assigned for the training data. Duplicated rather than
# imported, per the repo's no-shared-utils convention -- but it MUST stay in
# sync with that file, because season is a model feature and a different
# month boundary would be a different feature. Only used when building from
# --base_table (the prediction grid); the training path inherits season from
# the PM2.5 table, which that script already produced.
SEASON_MONTHS = {
    3: "summer", 4: "summer", 5: "summer",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
    12: "winter", 1: "winter", 2: "winter",
}

def load_station_list(station_file):
    stations = pd.read_csv(station_file)
    stations = stations[stations["status"] == "KEEP"]
    return stations[["location_id", "name", "latitude", "longitude"]]

def merge_daily_table(master, path, keep_cols, rename_map=None):
    # keep_cols are the non-key columns we actually want from this table
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["location_id", "date"] + keep_cols]
    if rename_map:
        df = df.rename(columns=rename_map)
    before = len(master)
    master = master.merge(df, on=["location_id", "date"], how="left")
    print(f"Merged {path}: {before} -> {len(master)} rows")
    return master

def merge_static_table(master, path, keep_cols):
    df = pd.read_csv(path)
    df = df[["location_id"] + keep_cols]
    before = len(master)
    master = master.merge(df, on="location_id", how="left")
    print(f"Merged {path}: {before} -> {len(master)} rows")
    return master

def merge_ndvi_periods(master, path):
    # NDVI is on 5-day periods, not daily -- match each station-day to the
    # period whose [period_start, period_end] range contains that date
    ndvi = pd.read_csv(path)
    ndvi["period_start"] = pd.to_datetime(ndvi["period_start"])
    ndvi["period_end"] = pd.to_datetime(ndvi["period_end"])
    ndvi = ndvi[["location_id", "period_start", "period_end", "period_index",
                 "ndvi_mean", "gap_filled", "fill_distance_periods"]]
    ndvi = ndvi.rename(columns={"gap_filled": "ndvi_gap_filled"})
    # merge_asof requires the "on" column sorted globally (not just within
    # each "by" group), so sort by date/period_start alone here
    ndvi = ndvi.sort_values("period_start")

    master_sorted = master.sort_values("date")
    merged = pd.merge_asof(master_sorted, ndvi, left_on="date", right_on="period_start",
                            by="location_id", direction="backward")
    merged = merged.sort_values(["location_id", "date"]).reset_index(drop=True)

    # every date should land inside its matched period -- if not, the period
    # table has a gap and this needs investigating before trusting the join
    outside_period = merged[merged["date"] > merged["period_end"]]
    print(f"NDVI period join: {len(outside_period)} rows fell outside their matched period (expected 0)")

    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station_file", required=True)
    # Optional, because the PREDICTION grid has no ground truth -- PM2.5 is the
    # thing being predicted. One of --pm25 or --base_table must be given; the
    # base simply decides which (location, date) rows exist. Everything after
    # that is identical, which is the point: the NDVI period join in particular
    # is a feature-defining operation (merge_asof onto 5-day period ranges) and
    # a second copy of it for the grid would be free to drift.
    parser.add_argument("--pm25", default=None)
    parser.add_argument("--base_table", default=None,
                        help="CSV of location_id,date to use as the row set "
                             "instead of the PM2.5 table (prediction grid)")
    parser.add_argument("--aod", required=True)
    parser.add_argument("--era5_land", required=True)
    parser.add_argument("--era5_blh", required=True)
    parser.add_argument("--ndvi", required=True)
    parser.add_argument("--worldcover", required=True)
    parser.add_argument("--srtm", required=True)
    parser.add_argument("--roads", required=True)
    parser.add_argument("--industrial", required=True)
    parser.add_argument("--powerplants", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # start from the PM2.5 target table -- this defines which station-days
    # are in the master table at all
    if args.pm25 is None and args.base_table is None:
        raise SystemExit("ERROR: give either --pm25 (training) or --base_table (prediction grid)")
    if args.pm25 is not None and args.base_table is not None:
        raise SystemExit("ERROR: --pm25 and --base_table are alternatives, not both")
    if args.pm25 is not None:
        master = pd.read_csv(args.pm25)
        master["date"] = pd.to_datetime(master["date"])
        print(f"Base PM2.5 table: {len(master)} station-days, "
              f"{master['location_id'].nunique()} stations")
    else:
        base = pd.read_csv(args.base_table)
        master = base[["location_id", "date"]].drop_duplicates()
        master["date"] = pd.to_datetime(master["date"])
        master = master.sort_values(["location_id", "date"]).reset_index(drop=True)
        master["season"] = master["date"].dt.month.map(SEASON_MONTHS)
        print(f"Base grid table: {len(master)} cell-days, "
              f"{master['location_id'].nunique()} cells, "
              f"{master['date'].nunique()} dates (no PM2.5 -- prediction grid)")
        print("Season assigned from month:",
              master["season"].value_counts().to_dict())

    stations = load_station_list(args.station_file)
    master = master.merge(stations, on="location_id", how="left")
    print(f"Attached station name/lat/lon: {len(master)} rows")

    master = merge_daily_table(
        master, args.aod,
        keep_cols=["aod_055", "gap_filled", "fill_source", "confidence_bucket", "confidence_rmse"],
        rename_map={"gap_filled": "aod_gap_filled"},
    )

    master = merge_daily_table(
        master, args.era5_land,
        keep_cols=["temperature_c", "wind_speed", "relative_humidity", "n_hours_used"],
        rename_map={"n_hours_used": "n_hours_met"},
    )
    # dewpoint_c deliberately excluded -- redundant with relative_humidity

    master = merge_daily_table(
        master, args.era5_blh,
        keep_cols=["boundary_layer_height", "n_hours_used"],
        rename_map={"n_hours_used": "n_hours_blh"},
    )

    master = merge_ndvi_periods(master, args.ndvi)

    master = merge_static_table(
        master, args.worldcover,
        keep_cols=["tree_cover_pct", "shrubland_pct", "grassland_pct", "cropland_pct",
                   "built_up_pct", "bare_sparse_veg_pct", "snow_ice_pct", "water_pct",
                   "wetland_herbaceous_pct", "mangroves_pct", "moss_lichen_pct"],
    )

    master = merge_static_table(master, args.srtm, keep_cols=["elevation_m", "slope_deg"])

    master = merge_static_table(master, args.roads, keep_cols=["road_density_km_per_km2"])

    master = merge_static_table(master, args.industrial, keep_cols=["industrial_landuse_fraction"])

    master = merge_static_table(
        master, args.powerplants,
        keep_cols=["dist_to_nearest_powerplant_km", "nearest_powerplant_name"],
    )

    master["date"] = master["date"].dt.strftime("%Y-%m-%d")
    master["period_start"] = master["period_start"].dt.strftime("%Y-%m-%d")
    master["period_end"] = master["period_end"].dt.strftime("%Y-%m-%d")

    master.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(master)} rows, {len(master.columns)} columns")

    print("=== NaN counts per column ===")
    for col in master.columns:
        n_missing = master[col].isna().sum()
        if n_missing > 0:
            print(f"{col}: {n_missing} missing")

if __name__ == "__main__":
    main()
