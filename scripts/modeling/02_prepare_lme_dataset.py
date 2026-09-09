import argparse
import os
import pandas as pd

REFERENCE_SEASON = "summer"
OTHER_SEASONS = ["monsoon", "post_monsoon", "winter"]

# WorldCover columns dropped entirely for LME: snow_ice/mangroves/moss_lichen are
# 0% at every Delhi station (no information), wetland_herbaceous_pct is nonzero at
# only 3 of 42 stations (negligible), and water_pct is dropped as the reference
# category so the remaining land-use % columns don't sum to a constant 100%
WORLDCOVER_DROP_COLS = ["snow_ice_pct", "mangroves_pct", "moss_lichen_pct",
                         "wetland_herbaceous_pct", "water_pct"]

WORLDCOVER_KEEP_COLS = ["tree_cover_pct", "shrubland_pct", "grassland_pct",
                         "cropland_pct", "built_up_pct", "bare_sparse_veg_pct"]

MET_COLS = ["temperature_c", "relative_humidity", "wind_speed", "boundary_layer_height"]
TERRAIN_COLS = ["elevation_m", "slope_deg"]
OSM_COLS = ["road_density_km_per_km2", "industrial_landuse_fraction", "dist_to_nearest_powerplant_km"]

# every continuous regressor gets centered and scaled for numerical stability
# and so the fitted coefficients are comparable to each other
SCALE_COLS = ["aod_055", "ndvi_mean"] + MET_COLS + TERRAIN_COLS + OSM_COLS + WORLDCOVER_KEEP_COLS

# these are the columns that must be non-missing for a row to enter the LME
# dataset at all (complete-case filtering) -- this is what drops the 113
# station-days with no AOD value at all
REQUIRED_COLS = ["pm25_daily", "aod_055", "season", "location_id"] + MET_COLS + \
                ["ndvi_mean"] + WORLDCOVER_KEEP_COLS + TERRAIN_COLS + OSM_COLS

def scale_columns(df, cols):
    scaling_stats = []
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
        scaling_stats.append({"column": col, "mean": mean, "std": std})
    return df, pd.DataFrame(scaling_stats)

def build_season_dummies(df):
    for season in OTHER_SEASONS:
        df[f"season_{season}"] = (df["season"] == season).astype(int)
    return df

def build_aod_interactions(df):
    # AOD's total effect is (b1 + b_interaction*season) -- summer is the
    # reference season, so its interaction is implicitly the plain aod_055
    # coefficient and needs no separate column
    for season in OTHER_SEASONS:
        df[f"aod_x_{season}"] = df["aod_055"] * df[f"season_{season}"]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="master_feature_table.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    outdir = os.path.dirname(args.output)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded master table: {len(df)} rows")

    print("=== Missing values in required columns (before complete-case filter) ===")
    for col in REQUIRED_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"{col}: {n_missing} missing")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    print(f"Complete-case filter: {before} -> {len(df)} rows ({before - len(df)} dropped)")

    df = df.drop(columns=WORLDCOVER_DROP_COLS)

    df = build_season_dummies(df)

    # scale first, then build interactions from the *scaled* AOD so the
    # interaction terms are on the same scale as the main aod_055 effect
    df, scaling_stats = scale_columns(df, SCALE_COLS)
    df = build_aod_interactions(df)

    lme_cols = ["location_id", "name", "date", "season", "pm25_daily",
                "aod_055", "aod_x_monsoon", "aod_x_post_monsoon", "aod_x_winter",
                "season_monsoon", "season_post_monsoon", "season_winter"] + \
               MET_COLS + ["ndvi_mean"] + WORLDCOVER_KEEP_COLS + TERRAIN_COLS + OSM_COLS + \
               ["confidence_bucket"]
    df = df[lme_cols]

    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(df)} rows, {len(df.columns)} columns")

    scaling_path = os.path.join(outdir, "lme_scaling_params.csv")
    scaling_stats.to_csv(scaling_path, index=False)
    print(f"Wrote {scaling_path} (mean/std used to scale each column, for back-transforming coefficients)")

    print("=== Rows per season (after complete-case filter) ===")
    for season in [REFERENCE_SEASON] + OTHER_SEASONS:
        sub = df[df["season"] == season]
        print(f"{season}: {len(sub)} rows, {sub['location_id'].nunique()} stations")

    print("=== Rows per confidence_bucket (AOD gap-fill quality, for sensitivity analysis) ===")
    print(df["confidence_bucket"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
