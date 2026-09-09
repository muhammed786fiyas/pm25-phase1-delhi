import argparse
import os
import numpy as np
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

def check_target_transform_is_set(target_transform):
    # this stage refuses to run until a human has looked at the EDA notebook
    # and typed in a real choice -- same NOT_SET guard pattern already used
    # in scripts/maiac_gapfill/04_score_confidence.py
    if target_transform == "NOT_SET":
        print("=== STOPPING: modeling.lme_prep.target_transform is still NOT_SET ===")
        print("EDA (notebooks/01_eda_master_feature_table.ipynb, Phase 4) found pm25_daily")
        print("is right-skewed: skew = 1.89 on the raw scale, skew = -0.41 after a log")
        print("transform. Review the notebook, then set target_transform in params.yaml")
        print("to \"raw\" or \"log\" (and update the matching --target_transform value in")
        print("dvc.yaml's cmd, per this repo's convention of keeping both in sync).")
        raise SystemExit("target_transform is NOT_SET -- EDA must be reviewed and a real choice made first")

def drop_rows_with_zero_target(df):
    # EDA (Phase 2) found 46 station-days with pm25_daily == 0, which is an
    # implausible ground-truth reading and also breaks the log transform
    before = len(df)
    df = df[df["pm25_daily"] > 0].copy()
    print(f"Dropped {before - len(df)} rows with pm25_daily <= 0 (before: {before}, after: {len(df)})")
    return df

def clip_negative_aod(df):
    # EDA (Phase 2) found 11 station-days with a physically impossible
    # negative AOD value (an artifact of the gap-filling step, not real haze)
    n_negative = (df["aod_055"] < 0).sum()
    df["aod_055"] = df["aod_055"].clip(lower=0)
    print(f"Clipped {n_negative} negative aod_055 values to 0")
    return df

def add_modeling_target(df, target_transform):
    if target_transform == "raw":
        df["modeling_target"] = df["pm25_daily"]
    elif target_transform == "log":
        n_non_positive = (df["pm25_daily"] <= 0).sum()
        if n_non_positive > 0:
            raise SystemExit(f"Cannot log-transform: {n_non_positive} rows still have "
                              f"pm25_daily <= 0. Set drop_zero_target_rows to true.")
        df["modeling_target"] = np.log(df["pm25_daily"])
    else:
        raise SystemExit(f"Unknown target_transform: {target_transform}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="master_feature_table.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_transform", required=True, choices=["NOT_SET", "raw", "log"],
                         help="raw or log transform of pm25_daily -- set only after reviewing the EDA notebook")
    parser.add_argument("--drop_zero_target_rows", required=True, choices=["true", "false"],
                         help="drop rows where pm25_daily <= 0 before modeling")
    parser.add_argument("--clip_negative_aod_to_zero", required=True, choices=["true", "false"],
                         help="clip negative aod_055 values to 0 before scaling")
    args = parser.parse_args()

    check_target_transform_is_set(args.target_transform)

    outdir = os.path.dirname(args.output)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded master table: {len(df)} rows")

    print("=== Missing values in required columns (before complete-case filter) ===")
    for col in REQUIRED_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"{col}: {n_missing} missing")

    if args.drop_zero_target_rows == "true":
        df = drop_rows_with_zero_target(df)

    if args.clip_negative_aod_to_zero == "true":
        df = clip_negative_aod(df)

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    print(f"Complete-case filter: {before} -> {len(df)} rows ({before - len(df)} dropped)")

    df = df.drop(columns=WORLDCOVER_DROP_COLS)

    df = build_season_dummies(df)

    # scale first, then build interactions from the *scaled* AOD so the
    # interaction terms are on the same scale as the main aod_055 effect
    df, scaling_stats = scale_columns(df, SCALE_COLS)
    df = build_aod_interactions(df)

    df = add_modeling_target(df, args.target_transform)

    lme_cols = ["location_id", "name", "date", "season", "pm25_daily", "modeling_target",
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
