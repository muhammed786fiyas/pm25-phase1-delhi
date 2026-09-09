import argparse
import os
import pandas as pd

# LightGBM is unaffected by collinearity, so we only drop the 3 WorldCover
# columns that are literally 0% at every one of the 42 Delhi stations (no
# information at all) -- water_pct and wetland_herbaceous_pct are kept since
# they carry real, if modest, variation across stations
WORLDCOVER_DROP_COLS = ["snow_ice_pct", "mangroves_pct", "moss_lichen_pct"]

MET_COLS = ["temperature_c", "relative_humidity", "wind_speed", "boundary_layer_height"]
TERRAIN_COLS = ["elevation_m", "slope_deg"]
OSM_COLS = ["road_density_km_per_km2", "industrial_landuse_fraction", "dist_to_nearest_powerplant_km"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="master_feature_table.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded master table: {len(df)} rows")

    worldcover_keep_cols = [col for col in df.columns if col.endswith("_pct")
                            and col not in WORLDCOVER_DROP_COLS]

    lgbm_cols = ["location_id", "name", "date", "season", "pm25_daily",
                 "aod_055", "aod_gap_filled", "confidence_rmse"] + \
                MET_COLS + ["ndvi_mean", "ndvi_gap_filled"] + \
                worldcover_keep_cols + TERRAIN_COLS + OSM_COLS
    df = df[lgbm_cols]

    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(df)} rows, {len(df.columns)} columns")
    print("Note: location_id is kept as an ID/grouping column only (e.g. for "
          "station-based CV splits) -- exclude it from the training feature list.")
    print("Note: season is a plain string column in this CSV -- cast it to a "
          "pandas category dtype (or pass categorical_feature=['season']) when "
          "loading this file for training.")

    print("=== Rows with missing AOD (native NaN, not dropped) ===")
    print(f"aod_055 missing: {df['aod_055'].isna().sum()} rows")

    print("=== Rows per season ===")
    print(df["season"].value_counts(dropna=False))

    print("=== Confidence bucket source (confidence_rmse present only for gap-filled AOD rows) ===")
    print(f"confidence_rmse missing (observed AOD, not filled): {df['confidence_rmse'].isna().sum()} rows")

if __name__ == "__main__":
    main()
