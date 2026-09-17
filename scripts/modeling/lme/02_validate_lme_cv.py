import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score

# Same fixed-effect list as scripts/modeling/lme/01_fit_lme_model.py -- kept as
# a separate copy here (repo convention: no shared utils module across
# scripts), must be kept in sync if the equation changes.
AOD_COLS = ["aod_055", "aod_x_monsoon", "aod_x_post_monsoon", "aod_x_winter"]
SEASON_COLS = ["season_monsoon", "season_post_monsoon", "season_winter"]
MET_COLS = ["temperature_c", "relative_humidity", "wind_speed", "boundary_layer_height"]
LANDUSE_COLS = ["ndvi_mean", "tree_cover_pct", "shrubland_pct", "grassland_pct",
                "cropland_pct", "built_up_pct", "bare_sparse_veg_pct",
                "elevation_m", "slope_deg",
                "road_density_km_per_km2", "industrial_landuse_fraction",
                "dist_to_nearest_powerplant_km"]

FIXED_EFFECTS = AOD_COLS + SEASON_COLS + MET_COLS + LANDUSE_COLS

TARGET_COL = "modeling_target"
RAW_TARGET_COL = "pm25_daily"
GROUP_COL = "location_id"
SEASON_COL = "season"

EARTH_RADIUS_KM = 6371.0

MLFLOW_EXPERIMENT_FULL = "delhi_phase1_lme"
MLFLOW_EXPERIMENT_GAPFILL_ROBUSTNESS = "delhi_phase1_lme_gapfill_robustness"

# Same re_formula as scripts/modeling/lme/01_fit_lme_model.py's
# RE_FORMULA_RANDOM_SLOPE_AOD -- adds a per-station random slope for aod_055
# alongside the random intercept, instead of the primary random-intercept-
# only model.
RE_FORMULA_RANDOM_SLOPE_AOD = "~ aod_055"

# 2026-09-09: within-R2 switched from a shared-demeaning classic-R2 formula to
# Kawano et al.'s literal formula (independent demeaning + corr^2). See
# compute_within_r2_kawano() below. Historical MLflow runs logged before this
# date used the old formula under the same metric name ("within_r2" /
# "within_r2_pooled") -- they were deleted from the tracking store rather than
# left to be confused with the new numbers (see docs/logs/tasks/9-LME_Model.md).
WITHIN_R2_FORMULA = "kawano_independent_demeaning_corr2"


def build_formula():
    return TARGET_COL + " ~ " + " + ".join(FIXED_EFFECTS)


def fit_lme(df, reml, re_formula=None):
    formula = build_formula()
    model = smf.mixedlm(formula, data=df, groups=df[GROUP_COL], re_formula=re_formula)
    result = model.fit(reml=reml)
    return result


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def compute_buffer_exclusions(station_df, buffer_km):
    # station_df must already be filtered to the KEEP stations. Recomputes
    # pairwise station distances directly from lat/lon every time this runs,
    # so it stays correct if the station roster ever changes -- nothing here
    # is hardcoded to specific station names or ids.
    ids = station_df[GROUP_COL].values
    names = station_df["name"].values
    lats = station_df["latitude"].values
    lons = station_df["longitude"].values
    n = len(station_df)

    exclusions = {}
    report_rows = []
    for i in range(n):
        distances = haversine_km(lats[i], lons[i], lats, lons)
        neighbor_mask = (distances <= buffer_km) & (np.arange(n) != i)
        neighbor_ids = ids[neighbor_mask]
        exclusions[ids[i]] = set(neighbor_ids)
        report_rows.append({
            "location_id": ids[i],
            "name": names[i],
            "n_excluded_neighbors": len(neighbor_ids),
            "excluded_location_ids": ",".join(str(x) for x in neighbor_ids),
        })

    n_affected = sum(1 for v in exclusions.values() if len(v) > 0)
    print(f"Buffer exclusion ({buffer_km} km): {n_affected} of {n} stations have "
          f"at least one neighbor within the buffer")
    return exclusions, pd.DataFrame(report_rows)


def predict_fixed_effects(result, df):
    return result.predict(exog=df[FIXED_EFFECTS])


def predict_with_random_effects(result, df, random_slope_aod):
    # Used for the random-CV scheme: if a station's other days appear in the
    # training fold, statsmodels has already fit a BLUP random intercept (and,
    # for the random-slope-AOD variant, a BLUP random AOD slope too) for it
    # (result.random_effects), so add that in on top of the fixed-effect
    # prediction. A station entirely absent from the training fold (should be
    # rare with a random row split) falls back to fixed-effects-only, same as
    # the spatial-LOSO scheme. With random_slope_aod=True, random_effects[loc]
    # is a 2-entry Series indexed ["Group", "aod_055"] -- the station's own
    # random slope multiplies that row's own aod_055 value (each row can have
    # a different AOD reading, unlike the intercept term).
    fixed_pred = predict_fixed_effects(result, df)
    random_effects = result.random_effects
    re_values = []
    for loc_id, aod_val in zip(df[GROUP_COL], df["aod_055"]):
        if loc_id in random_effects:
            re = random_effects[loc_id]
            re_total = re.iloc[0]
            if random_slope_aod:
                re_total += re["aod_055"] * aod_val
            re_values.append(re_total)
        else:
            re_values.append(0.0)
    return fixed_pred.values + np.array(re_values)


def compute_point_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae}


def group_key(df):
    return df[GROUP_COL].astype(str) + "_" + df[SEASON_COL]


def compute_group_means(df, values):
    # Station x season group means of the TRUE target, computed once from the
    # full dataset being validated (not from a single fold's test rows). A
    # single random-CV fold has only ~321 rows spread across up to 168
    # station x season groups, so most groups would have 1-2 rows if the mean
    # were computed per fold -- a singleton group's "mean" is just its own
    # value, which would zero out its contribution while still letting its
    # residual swing the metric wildly. Using one fixed, dataset-wide
    # reference mean per group avoids that and matches the literal
    # definition -- "that location's own seasonal average" -- rather than an
    # arbitrary CV split's average. Used as the TRUE-side reference for
    # compute_within_r2_kawano() below.
    #
    # Takes `values` rather than reading TARGET_COL directly so the same
    # helper can build both the modelling-scale means (log target) and the
    # raw ug/m3 means needed for the raw-scale within-R2 that the LightGBM
    # head-to-head compares against.
    keys = group_key(df)
    return pd.Series(values, index=keys.index).groupby(keys).mean()


def compute_pred_group_means(y_pred, group_keys):
    # Predicted-value counterpart to compute_group_means(): station x season
    # group means of the model's own out-of-fold predictions, computed once
    # from the FULL pooled OOF prediction set across every fold -- never from
    # a single fold's own rows, for the same singleton-group instability
    # reason as compute_group_means(). This is the PRED-side reference for
    # compute_within_r2_kawano() -- Kawano et al. demean the true series by
    # its own group's TRUE mean and the predicted series by its own group's
    # PREDICTED mean, independently.
    return pd.Series(y_pred, index=group_keys.index).groupby(group_keys).mean()


def compute_within_r2_kawano(y_true, y_pred, group_keys, true_group_means, pred_group_means):
    # Kawano et al.'s within-R2: Corr^2[ y_hat_it - y_hat_i , y_it - y_bar_i ].
    # De-mean the TRUE series by its own station x season group's TRUE mean,
    # and the PREDICTED series by its own group's PREDICTED mean -- two
    # different reference values, applied independently -- then take the
    # squared Pearson correlation of the two demeaned series. Independent
    # demeaning cancels out any constant per-group bias in the predictions
    # before scoring (a station the model over/under-predicts on average is
    # not penalized for that here), isolating whether the model tracks the
    # day-to-day/seasonal anomaly pattern rather than also requiring it to
    # get each group's absolute level right. This replaced an earlier
    # shared-demeaning classic-R2 formula on 2026-09-09 (see WITHIN_R2_FORMULA
    # above and docs/logs/tasks/9-LME_Model.md) -- the two formulas can swing
    # from strongly negative to positive on the same predictions whenever the
    # model's error is dominated by a per-station offset rather than
    # mistracked day-to-day anomalies.
    true_means = true_group_means.reindex(group_keys).values
    pred_means = pred_group_means.reindex(group_keys).values
    y_true_demeaned = y_true - true_means
    y_pred_demeaned = y_pred - pred_means
    if np.std(y_true_demeaned) == 0 or np.std(y_pred_demeaned) == 0:
        # No within-group variance to correlate against (can happen on a
        # small/degenerate fold slice) -- undefined, not zero.
        return np.nan
    corr = np.corrcoef(y_true_demeaned, y_pred_demeaned)[0, 1]
    return corr ** 2


def naive_backtransform_metrics(y_true_log, y_pred_log):
    # modeling_target is log(pm25_daily) (target_transform=log, decided in the
    # dataset-prep stage). This is a *naive* exp() back-transform with no bias
    # correction -- kept alongside the Duan-corrected metrics below purely so
    # the size of the correction is visible; not the recommended number to
    # report as-is (see duan_backtransform_metrics()).
    #
    # r2_ugm3_naive is reported alongside RMSE/MAE because R2 is NOT invariant
    # under a nonlinear transform: the log-scale r2 this script also reports is
    # a different quantity from the raw-scale one, computed against a different
    # denominator, and only the raw-scale one is comparable to LightGBM's.
    y_true_raw = np.exp(y_true_log)
    y_pred_raw = np.exp(y_pred_log)
    residuals = y_true_raw - y_pred_raw
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2 = r2_score(y_true_raw, y_pred_raw)
    return {"rmse_ugm3_naive": rmse, "mae_ugm3_naive": mae, "r2_ugm3_naive": r2}


def compute_smearing_factor(y_true_log, y_pred_log):
    # Duan (1983) smearing estimator. The naive exp() back-transform of a
    # log-scale prediction is biased low by Jensen's inequality -- E[exp(e)]
    # > exp(E[e]) whenever the residuals e have any spread at all, even if
    # E[e] = 0 -- so exp(fitted_log) systematically underpredicts the raw-
    # scale mean. The smearing factor S = mean(exp(residuals)), computed over
    # TRAINING residuals only, is a nonparametric correction (no normality
    # assumption on the residuals, unlike the classic exp(sigma^2/2)
    # lognormal correction): the corrected back-transform is
    # exp(fitted_log) * S. Must be computed from the training fold only and
    # never from the held-out rows being scored, or the correction leaks test
    # information into itself.
    residuals = y_true_log - y_pred_log
    return np.mean(np.exp(residuals))


def duan_backtransform_metrics(y_true_log, y_pred_log, smearing_factor):
    y_true_raw = np.exp(y_true_log)
    y_pred_raw = np.exp(y_pred_log) * smearing_factor
    residuals = y_true_raw - y_pred_raw
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2 = r2_score(y_true_raw, y_pred_raw)
    return {"rmse_ugm3_duan": rmse, "mae_ugm3_duan": mae, "r2_ugm3_duan": r2}


def raw_scale_predictions(y_pred_log, smearing_factor, target_transform):
    # The prediction vector in ug/m3, whichever scale the model was fit on.
    # For a log-target fit that is the Duan-corrected back-transform (the
    # statistically correct one of the two, per 9-LME_Model.md section 9);
    # for a raw-target fit the predictions are already in ug/m3 and must be
    # left alone -- np.exp() on a raw PM2.5 value would overflow.
    if target_transform == "raw":
        return y_pred_log
    return np.exp(y_pred_log) * smearing_factor


def raw_scale_truth(y_true, target_transform):
    if target_transform == "raw":
        return y_true
    return np.exp(y_true)


def compute_cv_results(folds_df, oof_true_list, oof_pred_list, oof_group_keys_list,
                        oof_fold_id_list, oof_smearing_list, true_group_means,
                        true_group_means_ugm3, target_transform):
    # Pools every fold's out-of-fold rows together, then computes the
    # within-R2 (Kawano) reference means and metrics from that pooled set --
    # both per-fold (sliced back out of the pooled set) and for the headline
    # aggregated/pooled number. This two-pass structure is required because
    # the predicted-side group mean (compute_pred_group_means) can only be
    # computed once every fold's predictions are known, unlike the true-side
    # group mean which is fixed upfront from the raw data.
    y_true = np.concatenate(oof_true_list)
    y_pred = np.concatenate(oof_pred_list)
    group_keys = pd.concat(oof_group_keys_list, ignore_index=True)
    fold_id = np.concatenate(oof_fold_id_list)
    # Each fold has its own Duan smearing factor (fit on that fold's own
    # training rows) -- broadcast per-row here so the pooled Duan-corrected
    # metric applies each row's own fold's factor, not a single global one.
    smearing_arr = np.concatenate(oof_smearing_list)

    pred_group_means = compute_pred_group_means(y_pred, group_keys)

    within_r2_by_fold = {}
    for f in np.unique(fold_id):
        mask = fold_id == f
        within_r2_by_fold[f] = compute_within_r2_kawano(
            y_true[mask], y_pred[mask], group_keys[mask], true_group_means, pred_group_means)
    folds_df = folds_df.copy()
    folds_df["within_r2"] = folds_df["fold"].map(within_r2_by_fold)

    point_metrics = compute_point_metrics(y_true, y_pred)
    within_r2_pooled = compute_within_r2_kawano(
        y_true, y_pred, group_keys, true_group_means, pred_group_means)

    # Raw ug/m3 within-R2 -- the one comparable to LightGBM's. Computed from
    # the raw-scale predictions (Duan-corrected for a log fit, as-is for a raw
    # fit) against raw-scale group means. Only the Duan flavour is computed:
    # Kawano's within-R2 is a squared Pearson correlation, so it is invariant
    # to multiplying the predictions by a positive constant, and Duan's
    # correction is exactly that -- a naive-flavoured version would differ
    # only through the small between-fold variation in the smearing factor
    # (~1.12-1.13), i.e. in about the 4th decimal.
    y_true_ugm3 = raw_scale_truth(y_true, target_transform)
    y_pred_ugm3 = raw_scale_predictions(y_pred, smearing_arr, target_transform)
    pred_group_means_ugm3 = compute_pred_group_means(y_pred_ugm3, group_keys)
    within_r2_ugm3 = compute_within_r2_kawano(
        y_true_ugm3, y_pred_ugm3, group_keys, true_group_means_ugm3, pred_group_means_ugm3)

    aggregated = {
        "target_transform": target_transform,
        "n_oof_rows": len(y_true),
        "r2": point_metrics["r2"],
        "within_r2": within_r2_pooled,
        "rmse": point_metrics["rmse"],
        "mae": point_metrics["mae"],
        "within_r2_ugm3": within_r2_ugm3,
    }

    if target_transform == "raw":
        # Already in ug/m3 -- no back-transform, no smearing. The *_ugm3 keys
        # are what the cross-model comparison reads, so mirror the native
        # metrics into them rather than leaving them absent.
        aggregated["r2_ugm3"] = point_metrics["r2"]
        aggregated["rmse_ugm3"] = point_metrics["rmse"]
        aggregated["mae_ugm3"] = point_metrics["mae"]
    else:
        backtransform = naive_backtransform_metrics(y_true, y_pred)
        duan_backtransform = duan_backtransform_metrics(y_true, y_pred, smearing_arr)
        aggregated.update({
            "r2_ugm3": duan_backtransform["r2_ugm3_duan"],
            "rmse_ugm3": duan_backtransform["rmse_ugm3_duan"],
            "mae_ugm3": duan_backtransform["mae_ugm3_duan"],
            "rmse_ugm3_naive": backtransform["rmse_ugm3_naive"],
            "mae_ugm3_naive": backtransform["mae_ugm3_naive"],
            "r2_ugm3_naive": backtransform["r2_ugm3_naive"],
            "rmse_ugm3_duan": duan_backtransform["rmse_ugm3_duan"],
            "mae_ugm3_duan": duan_backtransform["mae_ugm3_duan"],
            "r2_ugm3_duan": duan_backtransform["r2_ugm3_duan"],
            "smearing_factor_mean": float(np.mean(smearing_arr)),
            "smearing_factor_min": float(np.min(smearing_arr)),
            "smearing_factor_max": float(np.max(smearing_arr)),
        })
    return folds_df, aggregated


def fold_backtransform_rows(y_true, y_pred, train_true, train_pred, target_transform):
    # Per-fold raw-scale reporting. For a log-target fit this is the smearing
    # factor plus the naive and Duan back-transform metrics. For a raw-target
    # fit there is nothing to back-transform -- the fold's own r2/rmse/mae are
    # already ug/m3 -- so the smearing factor is reported as a neutral 1.0 and
    # no back-transform columns are emitted.
    if target_transform == "raw":
        return {"smearing_factor": 1.0}, 1.0
    smearing_factor = compute_smearing_factor(train_true, train_pred)
    rows = {"smearing_factor": smearing_factor}
    rows.update(naive_backtransform_metrics(y_true, y_pred))
    rows.update(duan_backtransform_metrics(y_true, y_pred, smearing_factor))
    return rows, smearing_factor


def run_spatial_loso_cv(df, exclusions, true_group_means, true_group_means_ugm3,
                        re_formula, target_transform):
    # Fixed-effects-only prediction here regardless of re_formula: a held-out
    # station gets NO random-effect BLUP of any kind -- not an intercept, and
    # not an AOD slope either -- since statsmodels has never seen that
    # station's data. This is why the random-slope-AOD variant does not, by
    # itself, fix the land-use-extrapolation outlier problem on stations
    # 5598/6934 (see docs/logs/tasks/9-LME_Model.md "Ideas" section) -- it's
    # a fixed-effects/land-use problem, and spatial LOSO prediction never
    # touches random effects at all.
    #
    # Duan smearing factor here is computed from the TRAIN split's own
    # fixed-effects-only residuals (predict_fixed_effects on train_df), to
    # match exactly what predict_fixed_effects produces on the held-out
    # station -- using result.resid instead would mix in the training
    # stations' own random-intercept BLUPs, which the held-out prediction
    # never gets, and would bias the correction.
    station_ids = sorted(df[GROUP_COL].unique())
    fold_rows = []
    oof_true = []
    oof_pred = []
    oof_group_keys = []
    oof_fold_id = []
    oof_smearing = []

    for i, held_out_id in enumerate(station_ids):
        excluded_ids = exclusions.get(held_out_id, set())
        drop_ids = excluded_ids | {held_out_id}
        train_df = df[~df[GROUP_COL].isin(drop_ids)]
        test_df = df[df[GROUP_COL] == held_out_id]

        result = fit_lme(train_df, reml=True, re_formula=re_formula)

        train_pred_log = predict_fixed_effects(result, train_df).values

        y_pred = predict_fixed_effects(result, test_df).values
        y_true = test_df[TARGET_COL].values

        point_metrics = compute_point_metrics(y_true, y_pred)
        backtransform_rows, smearing_factor = fold_backtransform_rows(
            y_true, y_pred, train_df[TARGET_COL].values, train_pred_log, target_transform)

        fold_row = {
            "fold": i + 1,
            "held_out_station": held_out_id,
            "n_excluded_buffer_stations": len(excluded_ids),
            "n_train_stations": train_df[GROUP_COL].nunique(),
            "n_test_rows": len(test_df),
            "r2": point_metrics["r2"],
            "rmse": point_metrics["rmse"],
            "mae": point_metrics["mae"],
        }
        fold_row.update(backtransform_rows)
        fold_rows.append(fold_row)
        oof_true.append(y_true)
        oof_pred.append(y_pred)
        oof_group_keys.append(group_key(test_df))
        oof_fold_id.append(np.full(len(test_df), i + 1))
        oof_smearing.append(np.full(len(test_df), smearing_factor))

        print(f"Spatial LOSO fold {i + 1}/{len(station_ids)} station {held_out_id}: "
              f"R2={point_metrics['r2']:.3f} "
              f"RMSE={point_metrics['rmse']:.3f} MAE={point_metrics['mae']:.3f}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df, aggregated = compute_cv_results(
        folds_df, oof_true, oof_pred, oof_group_keys, oof_fold_id, oof_smearing,
        true_group_means, true_group_means_ugm3, target_transform)
    print("Spatial LOSO within_R2 (Kawano) by fold:")
    for row in folds_df.itertuples(index=False):
        print(f"  fold {row.fold} (station {row.held_out_station}): within_R2={row.within_r2}")
    return folds_df, aggregated


def run_random_cv(df, n_folds, seed, true_group_means, true_group_means_ugm3,
                  re_formula, random_slope_aod, target_transform):
    # Duan smearing factor here is computed from the TRAIN split's own
    # predict_with_random_effects residuals (fixed effects + BLUP), to match
    # exactly what predict_with_random_effects produces on the held-out rows
    # -- a station seen during training keeps its random-effect BLUP for
    # random CV, unlike spatial LOSO above.
    rng = np.random.RandomState(seed)
    shuffled_index = rng.permutation(df.index.values)
    fold_assignment = pd.Series(np.arange(len(df)) % n_folds, index=shuffled_index).sort_index()

    fold_rows = []
    oof_true = []
    oof_pred = []
    oof_group_keys = []
    oof_fold_id = []
    oof_smearing = []

    for fold_id in range(n_folds):
        test_mask = fold_assignment == fold_id
        train_df = df[~test_mask]
        test_df = df[test_mask]

        result = fit_lme(train_df, reml=True, re_formula=re_formula)

        train_pred_log = predict_with_random_effects(result, train_df, random_slope_aod)

        y_pred = predict_with_random_effects(result, test_df, random_slope_aod)
        y_true = test_df[TARGET_COL].values

        point_metrics = compute_point_metrics(y_true, y_pred)
        backtransform_rows, smearing_factor = fold_backtransform_rows(
            y_true, y_pred, train_df[TARGET_COL].values, train_pred_log, target_transform)

        fold_row = {
            "fold": fold_id + 1,
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "r2": point_metrics["r2"],
            "rmse": point_metrics["rmse"],
            "mae": point_metrics["mae"],
        }
        fold_row.update(backtransform_rows)
        fold_rows.append(fold_row)
        oof_true.append(y_true)
        oof_pred.append(y_pred)
        oof_group_keys.append(group_key(test_df))
        oof_fold_id.append(np.full(len(test_df), fold_id + 1))
        oof_smearing.append(np.full(len(test_df), smearing_factor))

        print(f"Random CV fold {fold_id + 1}/{n_folds}: "
              f"R2={point_metrics['r2']:.3f} "
              f"RMSE={point_metrics['rmse']:.3f} MAE={point_metrics['mae']:.3f}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df, aggregated = compute_cv_results(
        folds_df, oof_true, oof_pred, oof_group_keys, oof_fold_id, oof_smearing,
        true_group_means, true_group_means_ugm3, target_transform)
    print("Random CV within_R2 (Kawano) by fold:")
    for row in folds_df.itertuples(index=False):
        print(f"  fold {row.fold}: within_R2={row.within_r2}")
    return folds_df, aggregated


def log_cv_run_to_mlflow(run_name, cv_scheme, folds_df, aggregated, folds_csv_path, extra_params):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("cv_scheme", cv_scheme)
        mlflow.log_param("formula", build_formula())
        mlflow.log_param("reml", True)
        mlflow.log_param("within_r2_formula", WITHIN_R2_FORMULA)
        for key, value in extra_params.items():
            mlflow.log_param(key, value)

        for row in folds_df.itertuples(index=False):
            mlflow.log_metric("fold_r2", row.r2, step=row.fold)
            if not pd.isna(row.within_r2):
                mlflow.log_metric("fold_within_r2", row.within_r2, step=row.fold)
            mlflow.log_metric("fold_rmse", row.rmse, step=row.fold)
            mlflow.log_metric("fold_mae", row.mae, step=row.fold)
            mlflow.log_metric("fold_smearing_factor", row.smearing_factor, step=row.fold)

        mlflow.log_metric("r2_pooled", aggregated["r2"])
        mlflow.log_metric("within_r2_pooled", aggregated["within_r2"])
        mlflow.log_metric("rmse_pooled", aggregated["rmse"])
        mlflow.log_metric("mae_pooled", aggregated["mae"])
        # The raw ug/m3 block -- what the LightGBM head-to-head compares
        # against, present for both target transforms.
        mlflow.log_metric("r2_ugm3_pooled", aggregated["r2_ugm3"])
        mlflow.log_metric("within_r2_ugm3_pooled", aggregated["within_r2_ugm3"])
        mlflow.log_metric("rmse_ugm3_pooled", aggregated["rmse_ugm3"])
        mlflow.log_metric("mae_ugm3_pooled", aggregated["mae_ugm3"])
        if "smearing_factor_mean" in aggregated:
            mlflow.log_metric("rmse_ugm3_naive_pooled", aggregated["rmse_ugm3_naive"])
            mlflow.log_metric("mae_ugm3_naive_pooled", aggregated["mae_ugm3_naive"])
            mlflow.log_metric("r2_ugm3_naive_pooled", aggregated["r2_ugm3_naive"])
            mlflow.log_metric("rmse_ugm3_duan_pooled", aggregated["rmse_ugm3_duan"])
            mlflow.log_metric("mae_ugm3_duan_pooled", aggregated["mae_ugm3_duan"])
            mlflow.log_metric("r2_ugm3_duan_pooled", aggregated["r2_ugm3_duan"])
            mlflow.log_metric("smearing_factor_mean", aggregated["smearing_factor_mean"])
            mlflow.log_metric("smearing_factor_min", aggregated["smearing_factor_min"])
            mlflow.log_metric("smearing_factor_max", aggregated["smearing_factor_max"])
        mlflow.log_metric("r2_fold_mean", folds_df["r2"].mean())
        mlflow.log_metric("r2_fold_std", folds_df["r2"].std())
        mlflow.log_artifact(folds_csv_path)

        print(f"Logged MLflow run '{run_name}' (cv_scheme={cv_scheme})")


def print_aggregated(label, aggregated):
    # The native-scale block is whatever modeling_target is on; the ug/m3
    # block is always present and is the one comparable to LightGBM.
    print(f"{label} aggregated ({aggregated['target_transform']} target): "
          f"R2={aggregated['r2']:.3f} within_R2={aggregated['within_r2']:.3f} "
          f"RMSE={aggregated['rmse']:.3f} MAE={aggregated['mae']:.3f}")
    print(f"  Raw ug/m3 (comparable to LightGBM): R2={aggregated['r2_ugm3']:.3f} "
          f"within_R2={aggregated['within_r2_ugm3']:.3f} "
          f"RMSE={aggregated['rmse_ugm3']:.3f} MAE={aggregated['mae_ugm3']:.3f}")
    if "smearing_factor_mean" in aggregated:
        print(f"  Back-transform detail: naive R2={aggregated['r2_ugm3_naive']:.3f} "
              f"RMSE={aggregated['rmse_ugm3_naive']:.3f} "
              f"MAE={aggregated['mae_ugm3_naive']:.3f} | Duan-corrected "
              f"R2={aggregated['r2_ugm3_duan']:.3f} "
              f"RMSE={aggregated['rmse_ugm3_duan']:.3f} "
              f"MAE={aggregated['mae_ugm3_duan']:.3f} "
              f"(mean smearing factor={aggregated['smearing_factor_mean']:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="lme_ready_dataset.csv")
    parser.add_argument("--station_file", required=True, help="cpcb_stations_delhi_status.csv")
    parser.add_argument("--buffer_km", required=True, type=float)
    parser.add_argument("--n_random_folds", required=True, type=int)
    parser.add_argument("--random_seed", required=True, type=int)
    parser.add_argument("--exclude_low_confidence", required=True, choices=["true", "false"],
                         help="true runs only the spatial-LOSO gap-fill robustness check "
                              "(excludes confidence_bucket == low), logged to a separate "
                              "MLflow experiment. false runs the primary spatial-LOSO and "
                              "random-CV comparison on all data.")
    parser.add_argument("--exclude_stations", default="",
                         help="Comma-separated location_ids to drop entirely from this "
                              "validation run (e.g. for a with/without outlier-station "
                              "comparison). Default: none excluded.")
    parser.add_argument("--random_slope_aod", default="false", choices=["true", "false"],
                         help="true validates the random-slope-AOD equation variant "
                              "(re_formula='~ aod_055', matching 01_fit_lme_model.py), "
                              "instead of the primary random-intercept-only model. "
                              "Default: false.")
    parser.add_argument("--target_transform", default="log", choices=["log", "raw"],
                         help="Which scale modeling_target is on in --input. 'log' is the "
                              "primary dataset and reports both the log-scale metrics and a "
                              "back-transformed ug/m3 block (naive + Duan). 'raw' is the "
                              "raw-target comparison variant: predictions are already in "
                              "ug/m3, so the back-transform and Duan smearing are skipped "
                              "entirely (np.exp() on a raw PM2.5 value would overflow). "
                              "Default: log.")
    parser.add_argument("--run_tag", default="",
                         help="Optional tag appended to MLflow run names, so a diagnostic "
                              "variant (e.g. an --exclude_stations run, or a run against a "
                              "different --input dataset) doesn't collide with the primary "
                              "run's names in the same MLflow experiment. Output filenames "
                              "don't need this -- give each variant its own --output_dir "
                              "(e.g. reports/lme/cv_<variant>/) instead. Default: no tag.")
    parser.add_argument("--output_dir", required=True,
                         help="Give each run variant its own directory (e.g. "
                              "reports/lme/cv_primary/, reports/lme/cv_aod_winsorized/) so "
                              "output filenames can stay plain and reports/lme/ stays "
                              "organized by run rather than by suffixed filename.")
    args = parser.parse_args()

    load_dotenv()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("No MLFLOW_TRACKING_URI found. Set it in .env")
    mlflow.set_tracking_uri(tracking_uri)

    os.makedirs(args.output_dir, exist_ok=True)

    stations = pd.read_csv(args.station_file)
    keep_stations = stations[stations["status"] == "KEEP"].reset_index(drop=True)
    exclusions, exclusion_report = compute_buffer_exclusions(keep_stations, args.buffer_km)

    df = pd.read_csv(args.input)
    print(f"Loaded {args.input}: {len(df)} rows, {df[GROUP_COL].nunique()} stations")

    exclude_station_ids = set()
    if args.exclude_stations:
        exclude_station_ids = {int(x.strip()) for x in args.exclude_stations.split(",") if x.strip()}
        before = len(df)
        df = df[~df[GROUP_COL].isin(exclude_station_ids)].reset_index(drop=True)
        print(f"Excluded stations {sorted(exclude_station_ids)}: {before} -> {len(df)} rows, "
              f"{df[GROUP_COL].nunique()} stations remaining")

    exclude_low_confidence = args.exclude_low_confidence == "true"
    if exclude_low_confidence:
        before = len(df)
        df = df[df["confidence_bucket"] != "low"].reset_index(drop=True)
        print(f"Excluded confidence_bucket == 'low' rows: {before} -> {len(df)} rows")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_GAPFILL_ROBUSTNESS)
        run_name_suffix = "_excl_low_confidence"
    else:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_FULL)
        run_name_suffix = ""

    random_slope_aod = args.random_slope_aod == "true"
    re_formula = RE_FORMULA_RANDOM_SLOPE_AOD if random_slope_aod else None
    if random_slope_aod:
        run_name_suffix += "_random_slope_aod"

    if args.run_tag:
        run_name_suffix += "_" + args.run_tag

    if not exclude_low_confidence:
        # Plain filename -- each run variant gets its own --output_dir (see
        # dvc.yaml), so no suffix is needed to keep variants from colliding.
        exclusion_path = os.path.join(args.output_dir, "station_buffer_exclusions.csv")
        exclusion_report.to_csv(exclusion_path, index=False)
        print(f"Wrote {exclusion_path}")

    true_group_means = compute_group_means(df, df[TARGET_COL].values)
    # Raw ug/m3 group means for the raw-scale within-R2. pm25_daily is carried
    # in every lme_ready dataset variant regardless of what modeling_target
    # holds, so this is the same reference for both transforms.
    true_group_means_ugm3 = compute_group_means(df, df[RAW_TARGET_COL].values)

    print("=== Spatial LOSO CV (primary, out-of-site, 2km buffer) ===")
    spatial_folds_df, spatial_aggregated = run_spatial_loso_cv(
        df, exclusions, true_group_means, true_group_means_ugm3, re_formula,
        args.target_transform)
    spatial_folds_path = os.path.join(args.output_dir, "cv_spatial_loso_folds.csv")
    spatial_folds_df.to_csv(spatial_folds_path, index=False)
    print(f"Wrote {spatial_folds_path}")
    spatial_agg_path = os.path.join(args.output_dir, "cv_spatial_loso_aggregated.json")
    with open(spatial_agg_path, "w") as f:
        json.dump(spatial_aggregated, f, indent=2)
    print(f"Wrote {spatial_agg_path}")
    print_aggregated("Spatial LOSO", spatial_aggregated)

    log_cv_run_to_mlflow(
        run_name="spatial_loso_cv" + run_name_suffix,
        cv_scheme="spatial_loso",
        folds_df=spatial_folds_df,
        aggregated=spatial_aggregated,
        folds_csv_path=spatial_folds_path,
        extra_params={
            "buffer_km": args.buffer_km,
            "n_folds": spatial_folds_df.shape[0],
            "exclude_low_confidence": exclude_low_confidence,
            "exclude_stations": sorted(exclude_station_ids) if exclude_station_ids else "none",
            "random_slope_aod": random_slope_aod,
            "target_transform": args.target_transform,
            "run_tag": args.run_tag or "none",
            "n_rows": len(df),
        },
    )

    if exclude_low_confidence:
        print("exclude_low_confidence=true: skipping random CV (spatial-LOSO "
              "robustness check only, per gap-fill validation scope).")
        return

    print("=== Random CV (comparison, ignores station grouping) ===")
    random_folds_df, random_aggregated = run_random_cv(
        df, args.n_random_folds, args.random_seed, true_group_means, true_group_means_ugm3,
        re_formula, random_slope_aod, args.target_transform)
    random_folds_path = os.path.join(args.output_dir, "cv_random_folds.csv")
    random_folds_df.to_csv(random_folds_path, index=False)
    print(f"Wrote {random_folds_path}")
    random_agg_path = os.path.join(args.output_dir, "cv_random_aggregated.json")
    with open(random_agg_path, "w") as f:
        json.dump(random_aggregated, f, indent=2)
    print(f"Wrote {random_agg_path}")
    print_aggregated("Random CV", random_aggregated)

    log_cv_run_to_mlflow(
        run_name="random_cv" + run_name_suffix,
        cv_scheme="random_cv",
        folds_df=random_folds_df,
        aggregated=random_aggregated,
        folds_csv_path=random_folds_path,
        extra_params={
            "n_folds": args.n_random_folds,
            "random_seed": args.random_seed,
            "exclude_low_confidence": exclude_low_confidence,
            "exclude_stations": sorted(exclude_station_ids) if exclude_station_ids else "none",
            "random_slope_aod": random_slope_aod,
            "target_transform": args.target_transform,
            "run_tag": args.run_tag or "none",
            "n_rows": len(df),
        },
    )

    print("=== Spatial LOSO vs random CV (inflation check) ===")
    print(f"Spatial LOSO pooled R2: {spatial_aggregated['r2']:.3f}, "
          f"within_R2: {spatial_aggregated['within_r2']:.3f}")
    print(f"Random CV pooled R2: {random_aggregated['r2']:.3f}, "
          f"within_R2: {random_aggregated['within_r2']:.3f}")
    print("Random CV is expected to show higher R2 than spatial LOSO: a station's "
          "random intercept u_i is already fit from its other training days, so "
          "its held-out days become easier to predict than a station the model "
          "has never seen at all (spatial LOSO) -- this gap demonstrates the "
          "spatial-leakage inflation, matching Kawano et al. and the other "
          "benchmark papers in the project's literature docs.")


if __name__ == "__main__":
    main()
