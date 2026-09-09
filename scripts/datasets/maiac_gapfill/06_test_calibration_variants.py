"""
OUTCOME (2026-09-09): season-intercept was promoted to production based on
this comparison (see docs/logs/tasks/7-MAIAC_Gapfill.md) -- it captured
nearly all of full season-specific fit's gain (0.2669 vs 0.2659 overall
LOOCV RMSE) without the small-sample monsoon slope fragility. Static
covariates showed no benefit and were not pursued further. Scripts
02/04/05 now implement the season-intercept model this script tested.

EXPLORATORY / standalone -- not wired into dvc.yaml or params.yaml.

Tests three richer calibration variants against the production baseline
(one pooled-per-year, per-station a + b*MERRA2 line):

  1. Season-intercept: shared slope b per station, but the intercept a shifts
     by season (summer/monsoon/post_monsoon/winter). Still one fit per
     station, using all of that station's overlap days together.
  2. Full season interaction: a separate a AND b fit per station x season,
     using only that season's (small) overlap days -- "how bad does it get"
     if we take "different relationship per season" completely literally.
  3. Static-covariate-augmented pooled model: for the 38 stations that share
     one MERRA-2 grid cell (so MERRA2 alone can't tell them apart on a given
     day), pools their overlap days into one regression and adds station-level
     static covariates (NDVI, land cover, road density, industrial fraction,
     elevation) as extra predictors, compared against a plain pooled
     MERRA2-only model for the same cluster.

All four (including the baseline) are scored with cross-validated RMSE,
broken out by season as well as overall, and saved to one comparison CSV so
the full table doesn't have to be reconstructed by hand.
"""

import argparse
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_FOLDS = 5

SEASONS_IN_ORDER = ["summer", "monsoon", "post_monsoon", "winter"]


def fit_ols_line(x_values, y_values):
    slope, intercept = np.polyfit(x_values, y_values, 1)
    return intercept, slope


def fit_multiple_ols(feature_matrix, y_values):
    n_rows = feature_matrix.shape[0]
    design_matrix = np.column_stack([np.ones(n_rows), feature_matrix])
    coefficients, residuals_sum, rank, singular_values = np.linalg.lstsq(design_matrix, y_values, rcond=None)
    return coefficients


def predict_multiple_ols(coefficients, feature_matrix):
    n_rows = feature_matrix.shape[0]
    design_matrix = np.column_stack([np.ones(n_rows), feature_matrix])
    return design_matrix @ coefficients


def rmse_of(residuals):
    return np.sqrt(np.mean(np.array(residuals) ** 2))


def summarize_by_season(rows, candidate_name, scope_name):
    # rows: list of {"season": ..., "residual": ...}
    rows_df = pd.DataFrame(rows)
    summary = []
    for season_name in SEASONS_IN_ORDER:
        season_rows = rows_df[rows_df["season"] == season_name]
        if len(season_rows) == 0:
            continue
        summary.append({
            "candidate": candidate_name,
            "scope": scope_name,
            "season": season_name,
            "rmse": rmse_of(season_rows["residual"]),
            "n_obs": len(season_rows),
        })
    summary.append({
        "candidate": candidate_name,
        "scope": scope_name,
        "season": "overall",
        "rmse": rmse_of(rows_df["residual"]),
        "n_obs": len(rows_df),
    })
    return summary


def loocv_baseline(station_df):
    station_df = station_df.reset_index(drop=True)
    rows = []
    for holdout_index in range(len(station_df)):
        train_df = station_df.drop(index=holdout_index)
        a, b = fit_ols_line(train_df["totexttau"].to_numpy(), train_df["aod_055"].to_numpy())
        holdout_row = station_df.loc[holdout_index]
        predicted = a + b * holdout_row["totexttau"]
        rows.append({"season": holdout_row["season"], "residual": holdout_row["aod_055"] - predicted})
    return rows


def loocv_season_intercept(station_df):
    station_df = station_df.reset_index(drop=True)
    season_dummy_cols = SEASONS_IN_ORDER[1:]  # summer is the reference season

    rows = []
    for holdout_index in range(len(station_df)):
        train_df = station_df.drop(index=holdout_index)

        dummy_matrix = np.column_stack([
            (train_df["season"] == season_name).astype(float).to_numpy()
            for season_name in season_dummy_cols
        ])
        feature_matrix = np.column_stack([train_df["totexttau"].to_numpy(), dummy_matrix])
        coefficients = fit_multiple_ols(feature_matrix, train_df["aod_055"].to_numpy())

        holdout_row = station_df.loc[holdout_index]
        holdout_dummies = [1.0 if holdout_row["season"] == season_name else 0.0 for season_name in season_dummy_cols]
        holdout_features = np.array([[holdout_row["totexttau"]] + holdout_dummies])
        predicted = predict_multiple_ols(coefficients, holdout_features)[0]
        rows.append({"season": holdout_row["season"], "residual": holdout_row["aod_055"] - predicted})
    return rows


def loocv_season_specific(station_df):
    rows = []
    for season_name, season_df in station_df.groupby("season"):
        season_df = season_df.reset_index(drop=True)
        if len(season_df) < 3:
            continue
        for holdout_index in range(len(season_df)):
            train_df = season_df.drop(index=holdout_index)
            a, b = fit_ols_line(train_df["totexttau"].to_numpy(), train_df["aod_055"].to_numpy())
            holdout_row = season_df.loc[holdout_index]
            predicted = a + b * holdout_row["totexttau"]
            rows.append({"season": holdout_row["season"], "residual": holdout_row["aod_055"] - predicted})
    return rows


def kfold_row_indices(n_rows, n_folds, seed):
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(n_rows)
    folds = np.array_split(shuffled_indices, n_folds)
    return folds


def kfold_cv_pooled(feature_matrix, y_values, season_values, n_folds, seed):
    n_rows = feature_matrix.shape[0]
    folds = kfold_row_indices(n_rows, n_folds, seed)
    rows = []
    for fold_index in range(n_folds):
        test_idx = folds[fold_index]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold_index])
        coefficients = fit_multiple_ols(feature_matrix[train_idx], y_values[train_idx])
        predicted = predict_multiple_ols(coefficients, feature_matrix[test_idx])
        residuals = y_values[test_idx] - predicted
        for season_name, residual in zip(season_values[test_idx], residuals):
            rows.append({"season": season_name, "residual": residual})
    return rows


def find_shared_cell_stations(cell_mapping_df):
    cell_counts = cell_mapping_df.groupby("cell_id").size()
    biggest_cell_id = cell_counts.idxmax()
    station_ids = cell_mapping_df[cell_mapping_df["cell_id"] == biggest_cell_id]["location_id"].tolist()
    return biggest_cell_id, station_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_input", default="data/raw/maiac_gapfill/calibration_dataset.csv")
    parser.add_argument("--cell_mapping", default="data/raw/merra2_aod/station_cell_mapping.csv")
    parser.add_argument("--worldcover_input", default="data/processed/static_gee/worldcover_landuse.csv")
    parser.add_argument("--ndvi_input", default="data/processed/static_gee/ndvi_gapfilled.csv")
    parser.add_argument("--srtm_input", default="data/processed/static_gee/srtm_terrain.csv")
    parser.add_argument("--road_density_input", default="data/processed/osm/road_density.csv")
    parser.add_argument("--industrial_input", default="data/processed/osm/industrial_fraction.csv")
    parser.add_argument("--output", default="data/interim/maiac_gapfill/calibration_variant_comparison.csv")
    args = parser.parse_args()

    print("=== Loading calibration dataset ===")
    calib_df = pd.read_csv(args.calibration_input)
    print("Total overlap rows:", len(calib_df))

    all_summary_rows = []

    print("")
    print("=== 1 & 2: per-station variants (baseline / season-intercept / season-specific) ===")
    baseline_rows_all = []
    season_intercept_rows_all = []
    season_specific_rows_all = []

    for location_id, station_df in calib_df.groupby("location_id"):
        if len(station_df) < 8:
            print("SKIPPED (too few points):", location_id)
            continue
        baseline_rows_all.extend(loocv_baseline(station_df))
        season_intercept_rows_all.extend(loocv_season_intercept(station_df))
        season_specific_rows_all.extend(loocv_season_specific(station_df))

    all_summary_rows.extend(summarize_by_season(baseline_rows_all, "baseline_pooled_per_station", "all_stations"))
    all_summary_rows.extend(summarize_by_season(season_intercept_rows_all, "season_intercept", "all_stations"))
    all_summary_rows.extend(summarize_by_season(season_specific_rows_all, "season_specific_fit", "all_stations"))

    print("")
    print("=== 3: static-covariate-augmented pooled model (shared MERRA-2 cell cluster) ===")
    cell_mapping_df = pd.read_csv(args.cell_mapping)
    biggest_cell_id, cluster_station_ids = find_shared_cell_stations(cell_mapping_df)
    print("Largest shared MERRA-2 cell:", biggest_cell_id, "-- stations:", len(cluster_station_ids))

    cluster_calib_df = calib_df[calib_df["location_id"].isin(cluster_station_ids)].copy()
    print("Pooled overlap rows for cluster:", len(cluster_calib_df))

    worldcover_df = pd.read_csv(args.worldcover_input)[["location_id", "tree_cover_pct", "built_up_pct"]]
    ndvi_df = pd.read_csv(args.ndvi_input).groupby("location_id")["ndvi_mean"].mean().reset_index().rename(columns={"ndvi_mean": "ndvi_mean_station"})
    srtm_df = pd.read_csv(args.srtm_input)[["location_id", "elevation_m"]]
    road_df = pd.read_csv(args.road_density_input)[["location_id", "road_density_km_per_km2"]]
    industrial_df = pd.read_csv(args.industrial_input)[["location_id", "industrial_landuse_fraction"]]

    covariates_df = worldcover_df.merge(ndvi_df, on="location_id").merge(srtm_df, on="location_id").merge(road_df, on="location_id").merge(industrial_df, on="location_id")
    cluster_calib_df = cluster_calib_df.merge(covariates_df, on="location_id", how="left")

    y_values = cluster_calib_df["aod_055"].to_numpy()
    season_values = cluster_calib_df["season"].to_numpy()
    merra2_feature = cluster_calib_df[["totexttau"]].to_numpy()

    covariate_columns = ["tree_cover_pct", "built_up_pct", "ndvi_mean_station", "elevation_m", "road_density_km_per_km2", "industrial_landuse_fraction"]
    augmented_feature = cluster_calib_df[["totexttau"] + covariate_columns].to_numpy()

    pooled_baseline_rows = kfold_cv_pooled(merra2_feature, y_values, season_values, N_FOLDS, RANDOM_SEED)
    pooled_augmented_rows = kfold_cv_pooled(augmented_feature, y_values, season_values, N_FOLDS, RANDOM_SEED)

    all_summary_rows.extend(summarize_by_season(pooled_baseline_rows, "pooled_cluster_merra2_only", "shared_cell_cluster"))
    all_summary_rows.extend(summarize_by_season(pooled_augmented_rows, "pooled_cluster_covariate_augmented", "shared_cell_cluster"))

    cluster_baseline_rows = []
    for location_id, station_df in calib_df[calib_df["location_id"].isin(cluster_station_ids)].groupby("location_id"):
        if len(station_df) < 8:
            continue
        cluster_baseline_rows.extend(loocv_baseline(station_df))
    all_summary_rows.extend(summarize_by_season(cluster_baseline_rows, "baseline_pooled_per_station", "shared_cell_cluster"))

    print("")
    print("=== Saving comparison table ===")
    comparison_df = pd.DataFrame(all_summary_rows)
    comparison_df = comparison_df[["candidate", "scope", "season", "rmse", "n_obs"]]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    comparison_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("")
    print("=== Comparison table (all_stations scope) ===")
    pivot = comparison_df[comparison_df["scope"] == "all_stations"].pivot(index="candidate", columns="season", values="rmse")
    pivot = pivot[[c for c in SEASONS_IN_ORDER + ["overall"] if c in pivot.columns]]
    print(pivot.round(4))

    print("")
    print("=== Comparison table (shared_cell_cluster scope) ===")
    pivot2 = comparison_df[comparison_df["scope"] == "shared_cell_cluster"].pivot(index="candidate", columns="season", values="rmse")
    pivot2 = pivot2[[c for c in SEASONS_IN_ORDER + ["overall"] if c in pivot2.columns]]
    print(pivot2.round(4))


if __name__ == "__main__":
    main()
