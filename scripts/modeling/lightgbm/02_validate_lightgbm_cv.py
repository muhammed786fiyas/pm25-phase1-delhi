import argparse
import json
import os

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score

# Same feature set as scripts/modeling/lightgbm/00_tune_lightgbm_hyperparameters.py
# and 01_fit_lightgbm_model.py -- kept as a separate copy here (repo
# convention: no shared utils module across scripts), must be kept in sync if
# the feature set changes.
ID_COLS = ["location_id", "name", "date"]
TARGET_COL = "pm25_daily"
GROUP_COL = "location_id"

SEASON_COL = "season"
SEASON_CATEGORIES = ["summer", "monsoon", "post_monsoon", "winter"]

FIXED_PARAMS = {"objective": "regression", "metric": "rmse", "verbosity": -1}

EARTH_RADIUS_KM = 6371.0

MLFLOW_EXPERIMENT = "delhi_phase1_lightgbm"

# Same Kawano-style within-R2 as the LME module's 02_validate_lme_cv.py, so
# the two models' within-R2 numbers are computed identically -- duplicated by
# hand here rather than imported (repo convention: no shared utils module).
WITHIN_R2_FORMULA = "kawano_independent_demeaning_corr2"

# LightGBM models pm25_daily raw, so its R2 / within-R2 / RMSE / MAE are all
# already in ug/m3 and no Duan smearing back-transform applies (that was purely
# a consequence of the LME's log target). The LME side now reports its own raw
# ug/m3 block too -- see 02_validate_lme_cv.py's --target_transform flag and
# the raw-target model variant -- so the head-to-head happens entirely in
# physical units and this script needs no log-scale counterpart. An earlier
# version reported a parallel log-scale metric set as a bridge to the LME's
# log-scale numbers; removed 2026-09-17 once the LME reported raw directly.


def build_feature_list(df):
    return [col for col in df.columns if col not in ID_COLS and col != TARGET_COL]


def cast_season_categorical(df):
    # Explicit fixed category order so the integer codes LightGBM sees are
    # identical in every fold -- a per-fold astype("category") would derive
    # them from whichever seasons that fold happens to contain.
    df = df.copy()
    df[SEASON_COL] = pd.Categorical(df[SEASON_COL], categories=SEASON_CATEGORIES)
    return df


def build_params(tuned_params, learning_rate, max_boosting_rounds, seed):
    params = dict(FIXED_PARAMS)
    params.update(tuned_params)
    params["learning_rate"] = learning_rate
    params["n_estimators"] = max_boosting_rounds
    params["random_state"] = seed
    return params


def check_boosting_settings_match(tuned, args):
    # Same drift guard as 01_fit_lightgbm_model.py: a tuned hyperparameter set
    # is only valid for the learning rate it was searched under, and these
    # three values live in params.yaml once per stage, so they could otherwise
    # diverge from the tuning run without anything noticing.
    mismatched = []
    for key, value in [("learning_rate", args.learning_rate),
                       ("max_boosting_rounds", args.max_boosting_rounds),
                       ("early_stopping_rounds", args.early_stopping_rounds)]:
        if tuned[key] != value:
            mismatched.append(f"{key}: tuned under {tuned[key]}, got {value}")
    if mismatched:
        raise SystemExit("Boosting settings do not match the tuning run: "
                         + "; ".join(mismatched)
                         + " -- align params.yaml or re-run "
                           "00_tune_lightgbm_hyperparameters.py")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def compute_buffer_exclusions(station_df, buffer_km):
    # station_df must already be filtered to the KEEP stations. Identical
    # logic to the LME's 02_validate_lme_cv.py so both models' spatial-LOSO
    # folds are built the same way -- pairwise station distances recomputed
    # from lat/lon every run, nothing hardcoded to specific station names or
    # ids.
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
        exclusions[int(ids[i])] = set(int(x) for x in neighbor_ids)
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


def block_lookup(tuned):
    # Maps each station to the tuning block it belongs to, so a spatial-LOSO
    # fold can pick up the hyperparameter set that was chosen without that
    # station's data ever entering the search.
    lookup = {}
    for label, station_ids in tuned["blocks"].items():
        for station_id in station_ids:
            lookup[station_id] = label
    return lookup


def split_validation_stations(df, validation_station_fraction, seed):
    # Early-stopping slice for the spatial-LOSO folds, split off by STATION.
    # 13 of the 23 features are constant within a station, so a random row
    # split would fill the validation slice with near-copies of training rows
    # and early stopping would not trigger until the model had memorized
    # station baselines -- which is exactly the behaviour spatial LOSO exists
    # to penalize.
    station_ids = np.array(sorted(df[GROUP_COL].unique()))
    n_validation = max(1, int(round(len(station_ids) * validation_station_fraction)))
    rng = np.random.RandomState(seed)
    validation_ids = set(rng.choice(station_ids, size=n_validation, replace=False))
    train_df = df[~df[GROUP_COL].isin(validation_ids)]
    validation_df = df[df[GROUP_COL].isin(validation_ids)]
    return train_df, validation_df


def split_validation_rows(df, validation_row_fraction, seed):
    # Early-stopping slice for the random-CV folds, split off by ROW. Random
    # CV is a row-level scheme by construction (that is the whole point of the
    # comparison), and holding out whole stations here would instead remove
    # them from training entirely, breaking the "every station is represented
    # in training" premise that produces the leakage effect being measured.
    rng = np.random.RandomState(seed)
    n_validation = max(1, int(round(len(df) * validation_row_fraction)))
    validation_positions = rng.choice(len(df), size=n_validation, replace=False)
    mask = np.zeros(len(df), dtype=bool)
    mask[validation_positions] = True
    return df[~mask], df[mask]


def find_best_iteration(params, train_df, validation_df, feature_cols,
                        early_stopping_rounds):
    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[feature_cols], train_df[TARGET_COL],
              eval_X=validation_df[feature_cols], eval_y=validation_df[TARGET_COL],
              eval_metric="rmse",
              callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                         lgb.log_evaluation(0)])
    return model.best_iteration_


def fit_fold_model(params, probe_train_df, probe_validation_df, full_train_df,
                   feature_cols, early_stopping_rounds):
    # Two-step, mirroring 01_fit_lightgbm_model.py's recipe exactly: train a
    # throwaway probe on probe_train_df and early-stop it against
    # probe_validation_df (a slice carved out of this fold's own training set,
    # and held out of probe_train_df -- otherwise early stopping would be
    # scoring rows it had just trained on and would never trigger), then refit
    # on the fold's FULL training set with the round count that probe found.
    # Keeping the probe instead would validate a model trained on ~85% of the
    # stations the production model gets, understating what the deliverable
    # actually does. The held-out test rows are never involved in either step.
    best_iteration = find_best_iteration(params, probe_train_df, probe_validation_df,
                                         feature_cols, early_stopping_rounds)
    refit_params = dict(params)
    refit_params["n_estimators"] = best_iteration
    model = lgb.LGBMRegressor(**refit_params)
    model.fit(full_train_df[feature_cols], full_train_df[TARGET_COL])
    return model, best_iteration


def compute_point_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae}


def group_key(df):
    return df[GROUP_COL].astype(str) + "_" + df[SEASON_COL].astype(str)


def compute_group_means(df, values):
    # Station x season group means, computed once from the full dataset being
    # validated -- never from a single fold's test rows. A single random-CV
    # fold has only ~324 rows spread across up to 168 station x season groups,
    # so most groups would have 1-2 rows if the mean were computed per fold; a
    # singleton group's "mean" is just its own value, contributing nothing to
    # the true-side spread while its prediction error still swings the metric.
    # Same reasoning and same fix as the LME module's compute_group_means()
    # (see docs/logs/tasks/9-LME_Model.md, "Bug caught and fixed").
    keys = group_key(df)
    return pd.Series(values, index=keys.index).groupby(keys).mean()


def compute_pred_group_means(y_pred, group_keys):
    # Predicted-value counterpart to compute_group_means(): computed once from
    # the FULL pooled out-of-fold prediction set across every fold, for the
    # same singleton-group instability reason. This is the PRED-side reference
    # for compute_within_r2_kawano() -- Kawano et al. demean the true series by
    # its own group's TRUE mean and the predicted series by its own group's
    # PREDICTED mean, independently.
    return pd.Series(y_pred, index=group_keys.index).groupby(group_keys).mean()


def compute_within_r2_kawano(y_true, y_pred, group_keys, true_group_means, pred_group_means):
    # Kawano et al.'s within-R2: Corr^2[ y_hat_it - y_hat_i , y_it - y_bar_i ].
    # De-mean the TRUE series by its own station x season group's TRUE mean and
    # the PREDICTED series by its own group's PREDICTED mean -- two different
    # reference values, applied independently -- then take the squared Pearson
    # correlation of the two demeaned series. Independent demeaning cancels any
    # constant per-group bias before scoring, isolating whether the model
    # tracks the day-to-day/seasonal anomaly pattern rather than also requiring
    # it to get each group's absolute level right. Byte-for-byte the same
    # formula as the LME module's copy, so the two models' within-R2 numbers
    # are directly comparable.
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


def compute_cv_results(folds_df, oof_true_list, oof_pred_list, oof_group_keys_list,
                       oof_fold_id_list, true_group_means_raw):
    # Pools every fold's out-of-fold rows together, then computes the within-R2
    # reference means and metrics from that pooled set -- both per-fold (sliced
    # back out of the pooled set) and for the headline aggregated number. This
    # two-pass structure is required because the predicted-side group mean can
    # only be computed once every fold's predictions are known, unlike the
    # true-side group mean which is fixed upfront from the raw data. Same
    # structure as the LME module's compute_cv_results().
    y_true = np.concatenate(oof_true_list)
    y_pred = np.concatenate(oof_pred_list)
    group_keys = pd.concat(oof_group_keys_list, ignore_index=True)
    fold_id = np.concatenate(oof_fold_id_list)

    pred_group_means_raw = compute_pred_group_means(y_pred, group_keys)

    within_r2_by_fold = {}
    for f in np.unique(fold_id):
        mask = fold_id == f
        within_r2_by_fold[f] = compute_within_r2_kawano(
            y_true[mask], y_pred[mask], group_keys[mask],
            true_group_means_raw, pred_group_means_raw)
    folds_df = folds_df.copy()
    folds_df["within_r2"] = folds_df["fold"].map(within_r2_by_fold)

    point_metrics = compute_point_metrics(y_true, y_pred)
    within_r2_pooled = compute_within_r2_kawano(
        y_true, y_pred, group_keys, true_group_means_raw, pred_group_means_raw)

    aggregated = {
        "n_oof_rows": len(y_true),
        "r2": point_metrics["r2"],
        "within_r2": within_r2_pooled,
        "rmse_ugm3": point_metrics["rmse"],
        "mae_ugm3": point_metrics["mae"],
    }
    return folds_df, aggregated


def run_spatial_loso_cv(df, exclusions, blocks_by_station, block_params, feature_cols,
                        true_group_means_raw, validation_station_fraction,
                        early_stopping_rounds, seed):
    # Holds out each station in turn, and also drops any station within the
    # buffer from that fold's training set -- identical fold construction to
    # the LME's spatial LOSO. The hyperparameters for station i come from the
    # Optuna search that excluded i's whole block, so no station's own data
    # ever influenced the hyperparameters used to evaluate it.
    station_ids = sorted(int(x) for x in df[GROUP_COL].unique())
    fold_rows = []
    oof_true = []
    oof_pred = []
    oof_group_keys = []
    oof_fold_id = []

    for i, held_out_id in enumerate(station_ids):
        excluded_ids = exclusions.get(held_out_id, set())
        drop_ids = excluded_ids | {held_out_id}
        train_df = df[~df[GROUP_COL].isin(drop_ids)].reset_index(drop=True)
        test_df = df[df[GROUP_COL] == held_out_id]

        block_label = blocks_by_station[held_out_id]
        params = block_params[block_label]

        inner_train_df, inner_validation_df = split_validation_stations(
            train_df, validation_station_fraction, seed + i)
        model, best_iteration = fit_fold_model(params, inner_train_df, inner_validation_df,
                                               train_df, feature_cols, early_stopping_rounds)

        y_pred = model.predict(test_df[feature_cols])
        y_true = test_df[TARGET_COL].values

        point_metrics = compute_point_metrics(y_true, y_pred)

        fold_rows.append({
            "fold": i + 1,
            "held_out_station": held_out_id,
            "tuning_block": block_label,
            "n_excluded_buffer_stations": len(excluded_ids),
            "n_train_stations": train_df[GROUP_COL].nunique(),
            "n_test_rows": len(test_df),
            "best_iteration": best_iteration,
            "r2": point_metrics["r2"],
            "rmse_ugm3": point_metrics["rmse"],
            "mae_ugm3": point_metrics["mae"],
        })
        oof_true.append(y_true)
        oof_pred.append(y_pred)
        oof_group_keys.append(group_key(test_df))
        oof_fold_id.append(np.full(len(test_df), i + 1))

        print(f"Spatial LOSO fold {i + 1}/{len(station_ids)} station {held_out_id} "
              f"({block_label}): R2={point_metrics['r2']:.3f} "
              f"RMSE={point_metrics['rmse']:.3f} MAE={point_metrics['mae']:.3f} ug/m3 "
              f"rounds={best_iteration}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df, aggregated = compute_cv_results(
        folds_df, oof_true, oof_pred, oof_group_keys, oof_fold_id, true_group_means_raw)
    print("Spatial LOSO within_R2 (Kawano) by fold:")
    for row in folds_df.itertuples(index=False):
        print(f"  fold {row.fold} (station {row.held_out_station}): "
              f"within_R2={row.within_r2}")
    return folds_df, aggregated


def run_random_cv(df, params, feature_cols, n_folds, true_group_means_raw,
                  validation_row_fraction, early_stopping_rounds, seed):
    # Row-level folds, ignoring station grouping -- the leakage comparison arm.
    # Every station is represented in every fold's training set, and 13 of the
    # 23 features are constant within a station, so the trees can recover a
    # per-station baseline from the land-use fingerprint alone. That is the
    # LightGBM analog of the LME's fitted random intercept, and the reason this
    # arm is expected to score higher than spatial LOSO on identical data.
    #
    # All 42 folds use the full-data hyperparameter set rather than the
    # block-nested ones: a row-level fold contains rows from all 42 stations,
    # so the "station i gets its own block's Hk" rule has nothing to key on,
    # and guarding this arm's hyperparameters against leakage would be
    # incoherent when its folds are deliberately leaky by construction. Noted
    # here because it means the two arms do NOT share hyperparameters, unlike
    # the LME comparison where only the fold-assignment method differed.
    rng = np.random.RandomState(seed)
    shuffled_index = rng.permutation(df.index.values)
    fold_assignment = pd.Series(np.arange(len(df)) % n_folds, index=shuffled_index).sort_index()

    fold_rows = []
    oof_true = []
    oof_pred = []
    oof_group_keys = []
    oof_fold_id = []

    for fold_id in range(n_folds):
        test_mask = fold_assignment == fold_id
        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask]

        inner_train_df, inner_validation_df = split_validation_rows(
            train_df, validation_row_fraction, seed + fold_id)
        model, best_iteration = fit_fold_model(params, inner_train_df, inner_validation_df,
                                               train_df, feature_cols, early_stopping_rounds)

        y_pred = model.predict(test_df[feature_cols])
        y_true = test_df[TARGET_COL].values

        point_metrics = compute_point_metrics(y_true, y_pred)

        fold_rows.append({
            "fold": fold_id + 1,
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "best_iteration": best_iteration,
            "r2": point_metrics["r2"],
            "rmse_ugm3": point_metrics["rmse"],
            "mae_ugm3": point_metrics["mae"],
        })
        oof_true.append(y_true)
        oof_pred.append(y_pred)
        oof_group_keys.append(group_key(test_df))
        oof_fold_id.append(np.full(len(test_df), fold_id + 1))

        print(f"Random CV fold {fold_id + 1}/{n_folds}: R2={point_metrics['r2']:.3f} "
              f"RMSE={point_metrics['rmse']:.3f} MAE={point_metrics['mae']:.3f} ug/m3 "
              f"rounds={best_iteration}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df, aggregated = compute_cv_results(
        folds_df, oof_true, oof_pred, oof_group_keys, oof_fold_id, true_group_means_raw)
    print("Random CV within_R2 (Kawano) by fold:")
    for row in folds_df.itertuples(index=False):
        print(f"  fold {row.fold}: within_R2={row.within_r2}")
    return folds_df, aggregated


def log_cv_run_to_mlflow(run_name, cv_scheme, folds_df, aggregated, folds_csv_path,
                         extra_params):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "validation")
        mlflow.set_tag("cv_scheme", cv_scheme)
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("target_transform", "none (raw ug/m3)")
        mlflow.log_param("within_r2_formula", WITHIN_R2_FORMULA)
        for key, value in extra_params.items():
            mlflow.log_param(key, value)

        for row in folds_df.itertuples(index=False):
            mlflow.log_metric("fold_r2", row.r2, step=row.fold)
            if not pd.isna(row.within_r2):
                mlflow.log_metric("fold_within_r2", row.within_r2, step=row.fold)
            mlflow.log_metric("fold_rmse_ugm3", row.rmse_ugm3, step=row.fold)
            mlflow.log_metric("fold_mae_ugm3", row.mae_ugm3, step=row.fold)
            mlflow.log_metric("fold_best_iteration", row.best_iteration, step=row.fold)

        mlflow.log_metric("r2_pooled", aggregated["r2"])
        mlflow.log_metric("within_r2_pooled", aggregated["within_r2"])
        mlflow.log_metric("rmse_ugm3_pooled", aggregated["rmse_ugm3"])
        mlflow.log_metric("mae_ugm3_pooled", aggregated["mae_ugm3"])
        mlflow.log_metric("r2_fold_mean", folds_df["r2"].mean())
        mlflow.log_metric("r2_fold_std", folds_df["r2"].std())
        mlflow.log_artifact(folds_csv_path)

        print(f"Logged MLflow run '{run_name}' (cv_scheme={cv_scheme})")


def print_aggregated(label, aggregated):
    print(f"{label} aggregated: R2={aggregated['r2']:.3f} "
          f"within_R2={aggregated['within_r2']:.3f} "
          f"RMSE={aggregated['rmse_ugm3']:.3f} MAE={aggregated['mae_ugm3']:.3f} ug/m3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="lightgbm_ready_dataset.csv")
    parser.add_argument("--station_file", required=True, help="cpcb_stations_delhi_status.csv")
    parser.add_argument("--hyperparameters", required=True,
                         help="tuned hyperparameters json from "
                              "00_tune_lightgbm_hyperparameters.py")
    parser.add_argument("--buffer_km", required=True, type=float)
    parser.add_argument("--n_random_folds", required=True, type=int)
    parser.add_argument("--random_seed", required=True, type=int)
    parser.add_argument("--learning_rate", required=True, type=float)
    parser.add_argument("--max_boosting_rounds", required=True, type=int)
    parser.add_argument("--early_stopping_rounds", required=True, type=int)
    parser.add_argument("--validation_station_fraction", required=True, type=float,
                         help="fraction of each spatial-LOSO fold's training stations "
                              "held out purely to early-stop on")
    parser.add_argument("--validation_row_fraction", required=True, type=float,
                         help="fraction of each random-CV fold's training rows held out "
                              "purely to early-stop on (rows, not stations -- random CV "
                              "needs every station represented in training)")
    parser.add_argument("--run_tag", default="",
                         help="Optional tag appended to MLflow run names, so a diagnostic "
                              "variant does not collide with the primary run's names in "
                              "the same MLflow experiment. Output filenames do not need "
                              "this -- give each variant its own --output_dir instead. "
                              "Default: no tag.")
    parser.add_argument("--output_dir", required=True,
                         help="Give each run variant its own directory (e.g. "
                              "reports/lightgbm/primary/) so output filenames can stay "
                              "plain, matching reports/lme/'s per-run-subfolder layout.")
    args = parser.parse_args()

    load_dotenv()
    # MLflow 3.x refuses the plain filesystem tracking backend by default
    # unless this is set -- same gotcha the LME scripts hit and document.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("No MLFLOW_TRACKING_URI found. Set it in .env")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.hyperparameters) as f:
        tuned = json.load(f)
    print(f"Loaded {args.hyperparameters}")

    stations = pd.read_csv(args.station_file)
    keep_stations = stations[stations["status"] == "KEEP"].reset_index(drop=True)
    exclusions, exclusion_report = compute_buffer_exclusions(keep_stations, args.buffer_km)

    df = pd.read_csv(args.input)
    df = cast_season_categorical(df)
    feature_cols = build_feature_list(df)
    print(f"Loaded {args.input}: {len(df)} rows, {df[GROUP_COL].nunique()} stations, "
          f"{len(feature_cols)} features")

    if feature_cols != tuned["feature_cols"]:
        raise SystemExit("Feature list does not match the one the hyperparameters were "
                         "tuned on -- re-run 00_tune_lightgbm_hyperparameters.py")

    check_boosting_settings_match(tuned, args)

    blocks_by_station = block_lookup(tuned)
    missing = [s for s in sorted(int(x) for x in df[GROUP_COL].unique())
               if s not in blocks_by_station]
    if missing:
        raise SystemExit(f"Stations {missing} have no tuning block -- re-run "
                         "00_tune_lightgbm_hyperparameters.py against this dataset")

    block_params = {}
    for label in tuned["block_hyperparameters"]:
        block_params[label] = build_params(tuned["block_hyperparameters"][label],
                                           args.learning_rate, args.max_boosting_rounds,
                                           args.random_seed)
    full_data_params = build_params(tuned["full_data_hyperparameters"], args.learning_rate,
                                    args.max_boosting_rounds, args.random_seed)

    exclusion_path = os.path.join(args.output_dir, "station_buffer_exclusions.csv")
    exclusion_report.to_csv(exclusion_path, index=False)
    print(f"Wrote {exclusion_path}")

    true_group_means_raw = compute_group_means(df, df[TARGET_COL].values)

    run_name_suffix = ""
    if args.run_tag:
        run_name_suffix = "_" + args.run_tag

    print("=== Spatial LOSO CV (primary, out-of-site, block-nested hyperparameters) ===")
    spatial_folds_df, spatial_aggregated = run_spatial_loso_cv(
        df, exclusions, blocks_by_station, block_params, feature_cols,
        true_group_means_raw, args.validation_station_fraction,
        args.early_stopping_rounds, args.random_seed)
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
            "hyperparameter_source": "block-nested optuna (per held-out station's block)",
            "n_blocks": len(tuned["blocks"]),
            "learning_rate": args.learning_rate,
            "early_stopping_rounds": args.early_stopping_rounds,
            "validation_station_fraction": args.validation_station_fraction,
            "run_tag": args.run_tag or "none",
            "n_rows": len(df),
        },
    )

    print("=== Random CV (comparison, ignores station grouping) ===")
    random_folds_df, random_aggregated = run_random_cv(
        df, full_data_params, feature_cols, args.n_random_folds, true_group_means_raw,
        args.validation_row_fraction, args.early_stopping_rounds, args.random_seed)
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
            "hyperparameter_source": "full_data optuna search (see run_random_cv docstring)",
            "learning_rate": args.learning_rate,
            "early_stopping_rounds": args.early_stopping_rounds,
            "validation_row_fraction": args.validation_row_fraction,
            "run_tag": args.run_tag or "none",
            "n_rows": len(df),
        },
    )

    print("=== Spatial LOSO vs random CV (inflation check) ===")
    print(f"Spatial LOSO pooled R2: {spatial_aggregated['r2']:.3f}, "
          f"within_R2: {spatial_aggregated['within_r2']:.3f}")
    print(f"Random CV pooled R2: {random_aggregated['r2']:.3f}, "
          f"within_R2: {random_aggregated['within_r2']:.3f}")
    print("Random CV is expected to show higher R2 than spatial LOSO: 13 of the 23 "
          "features are constant within a station, so once a station has any rows in "
          "the training fold the trees can recover its own baseline from that land-use "
          "fingerprint -- the LightGBM analog of the LME's fitted random intercept. "
          "The gap between the two arms is the spatial-leakage inflation, the same "
          "effect the LME module measured. For a like-for-like comparison use the "
          "raw-target LME (reports/lme/raw_target/, fit AND scored on ug/m3 exactly "
          "as this model is): R2 0.437 spatial LOSO vs 0.567 random CV.")


if __name__ == "__main__":
    main()
