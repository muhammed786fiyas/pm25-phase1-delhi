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

def drop_rows_with_zero_target(df):
    # EDA found 46 station-days with pm25_daily == 0, an implausible ground-
    # truth reading regardless of model type (a wrong label is a wrong label,
    # whether a linear model or a tree learns from it)
    before = len(df)
    df = df[df["pm25_daily"] > 0].copy()
    print(f"Dropped {before - len(df)} rows with pm25_daily <= 0 (before: {before}, after: {len(df)})")
    return df

def clip_negative_aod(df):
    # EDA found 11 station-days with a physically impossible negative
    # aod_055 (a gap-fill artifact) -- fixing this is a data-quality decision,
    # not a model-specific one, even though trees would not be numerically
    # broken by a negative value the way a linear model's scaling would be
    n_negative = (df["aod_055"] < 0).sum()
    df["aod_055"] = df["aod_055"].clip(lower=0)
    print(f"Clipped {n_negative} negative aod_055 values to 0")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="master_feature_table.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop_zero_target_rows", required=True, choices=["true", "false"],
                         help="drop rows where pm25_daily <= 0 before modeling")
    parser.add_argument("--clip_negative_aod_to_zero", required=True, choices=["true", "false"],
                         help="clip negative aod_055 values to 0")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded master table: {len(df)} rows")

    if args.drop_zero_target_rows == "true":
        df = drop_rows_with_zero_target(df)

    if args.clip_negative_aod_to_zero == "true":
        df = clip_negative_aod(df)

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
    print("Note: no target_transform here -- tree splits are invariant to any "
          "monotonic transform of the target, so log(pm25_daily) would not "
          "change LightGBM's learned structure, only complicate reporting.")

    print("=== Rows with missing AOD (native NaN, not dropped) ===")
    print(f"aod_055 missing: {df['aod_055'].isna().sum()} rows")

    print("=== Rows per season ===")
    print(df["season"].value_counts(dropna=False))

    print("=== Confidence bucket source (confidence_rmse present only for gap-filled AOD rows) ===")
    print(f"confidence_rmse missing (observed AOD, not filled): {df['confidence_rmse'].isna().sum()} rows")

if __name__ == "__main__":
    main()
