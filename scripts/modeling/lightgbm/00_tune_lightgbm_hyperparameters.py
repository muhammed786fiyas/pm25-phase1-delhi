import argparse
import json
import os

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold

# Feature set for the LightGBM comparison model: every column in
# lightgbm_ready_dataset.csv except the 3 ID/meta columns and the target.
# Unlike the LME equation (which hand-dropped water_pct /
# wetland_herbaceous_pct for VIF reasons and modelled log(pm25_daily)), trees
# are unaffected by collinearity and by near-constant columns, so nothing is
# dropped here -- aod_gap_filled, confidence_rmse, ndvi_gap_filled and the
# full WorldCover set all stay in. Kept as a separate copy in each of the 3
# lightgbm scripts (repo convention: no shared utils module across scripts),
# must be kept in sync if the feature set changes.
ID_COLS = ["location_id", "name", "date"]
TARGET_COL = "pm25_daily"
GROUP_COL = "location_id"

# season is a plain string column in lightgbm_ready_dataset.csv. Cast once,
# with an explicit fixed category order, so the integer codes LightGBM sees
# are identical in every fold -- a per-fold astype("category") would derive
# the codes from whichever seasons that fold happens to contain, silently
# remapping them between train and test.
SEASON_COL = "season"
SEASON_CATEGORIES = ["summer", "monsoon", "post_monsoon", "winter"]

EARTH_RADIUS_KM = 6371.0

# Search space per the kickoff brief: num_leaves rather than max_depth (the
# bigger lever in LightGBM), plus min_child_samples, the two subsampling
# fractions, and both regularization terms. learning_rate is fixed and the
# number of boosting rounds comes from early stopping, so n_estimators is
# never tuned directly.
NUM_LEAVES_RANGE = (15, 255)
MIN_CHILD_SAMPLES_RANGE = (5, 200)
FEATURE_FRACTION_RANGE = (0.4, 1.0)
BAGGING_FRACTION_RANGE = (0.4, 1.0)
BAGGING_FREQ_RANGE = (1, 7)
REG_ALPHA_RANGE = (1e-8, 10.0)
REG_LAMBDA_RANGE = (1e-8, 10.0)

# Optuna optimizes inner-CV RMSE (not MAE) -- picked once and used
# consistently for every search and for the reported tuning objective.
TUNING_OBJECTIVE_METRIC = "rmse"

MLFLOW_EXPERIMENT = "delhi_phase1_lightgbm"


def build_feature_list(df):
    return [col for col in df.columns if col not in ID_COLS and col != TARGET_COL]


def cast_season_categorical(df):
    df = df.copy()
    df[SEASON_COL] = pd.Categorical(df[SEASON_COL], categories=SEASON_CATEGORIES)
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def compute_buffer_neighbors(station_df, buffer_km):
    # Same haversine buffer logic as 02_validate_lightgbm_cv.py and the LME's
    # 02_validate_lme_cv.py -- duplicated per the repo's no-shared-utils
    # convention. Used here only to keep 2km-neighbour station clusters inside
    # the same tuning block (see build_station_blocks below), not to exclude
    # anything from training.
    ids = station_df[GROUP_COL].values
    lats = station_df["latitude"].values
    lons = station_df["longitude"].values
    n = len(station_df)

    # Plain Python ints throughout: the block assignment ends up in a json
    # output, and numpy int64 is not JSON-serializable.
    neighbors = {}
    for i in range(n):
        distances = haversine_km(lats[i], lons[i], lats, lons)
        neighbor_mask = (distances <= buffer_km) & (np.arange(n) != i)
        neighbors[int(ids[i])] = set(int(x) for x in ids[neighbor_mask])
    return neighbors


def buffer_clusters(station_ids, neighbors):
    # Connected components of the "within buffer_km of each other" graph. Two
    # stations under 2km apart are near-duplicate sites, so they must land in
    # the same tuning block: otherwise, holding out station i in the outer
    # spatial LOSO would still have let i's near-twin sit in the tuning
    # universe that chose i's hyperparameters -- exactly the leak the block
    # design exists to close.
    seen = set()
    clusters = []
    for station_id in station_ids:
        if station_id in seen:
            continue
        stack = [station_id]
        cluster = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            cluster.append(current)
            stack.extend(neighbors.get(current, set()))
        clusters.append(sorted(cluster))
    return clusters


def build_station_blocks(station_ids, neighbors, n_blocks, seed):
    # Fixed, one-time partition of the stations into n_blocks tuning blocks --
    # entirely separate from the 42 individual outer LOSO folds. Clusters are
    # shuffled with a fixed seed, then assigned largest-first to whichever
    # block is currently smallest, which lands on evenly-sized blocks whenever
    # the cluster sizes allow it (42 stations / 6 blocks = exactly 7 each for
    # the current Delhi roster: two 3-station clusters, three 2-station
    # clusters and 30 singletons).
    clusters = buffer_clusters(station_ids, neighbors)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(clusters))
    clusters = [clusters[i] for i in order]
    clusters.sort(key=len, reverse=True)

    blocks = [[] for _ in range(n_blocks)]
    for cluster in clusters:
        target = min(range(n_blocks), key=lambda b: (len(blocks[b]), b))
        blocks[target].extend(cluster)
    return [sorted(block) for block in blocks]


def suggest_params(trial, learning_rate, max_boosting_rounds, seed):
    return {
        "objective": "regression",
        "metric": TUNING_OBJECTIVE_METRIC,
        "learning_rate": learning_rate,
        "n_estimators": max_boosting_rounds,
        "verbosity": -1,
        "random_state": seed,
        "num_leaves": trial.suggest_int("num_leaves", NUM_LEAVES_RANGE[0],
                                        NUM_LEAVES_RANGE[1], log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", MIN_CHILD_SAMPLES_RANGE[0],
                                               MIN_CHILD_SAMPLES_RANGE[1], log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", FEATURE_FRACTION_RANGE[0],
                                                FEATURE_FRACTION_RANGE[1]),
        "subsample": trial.suggest_float("subsample", BAGGING_FRACTION_RANGE[0],
                                         BAGGING_FRACTION_RANGE[1]),
        "subsample_freq": trial.suggest_int("subsample_freq", BAGGING_FREQ_RANGE[0],
                                            BAGGING_FREQ_RANGE[1]),
        "reg_alpha": trial.suggest_float("reg_alpha", REG_ALPHA_RANGE[0],
                                         REG_ALPHA_RANGE[1], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", REG_LAMBDA_RANGE[0],
                                          REG_LAMBDA_RANGE[1], log=True),
    }


def inner_cv_rmse(params, df, feature_cols, n_inner_folds, early_stopping_rounds):
    # Inner CV is GroupKFold by location_id, never a plain random KFold: with
    # 13 station-constant land-use features in the feature set, a random row
    # split would let a trial score well purely by memorizing each station's
    # own baseline, and the search would then reward exactly the hyperparameter
    # settings that overfit station identity -- the same spatial-leakage
    # inflation the LME's random-CV vs spatial-LOSO gap already demonstrated
    # empirically (docs/logs/tasks/9-LME_Model.md).
    #
    # The inner validation fold doubles as the early-stopping set here, which
    # is mildly optimistic in absolute terms but shared identically by every
    # trial, so it does not distort the ranking the search actually uses. The
    # outer evaluation fits in 02_validate_lightgbm_cv.py carve their own
    # separate early-stopping slice instead, so no reported outer number
    # depends on it.
    groups = df[GROUP_COL].values
    splitter = GroupKFold(n_splits=n_inner_folds)
    fold_rmses = []
    for train_idx, val_idx in splitter.split(df, groups=groups):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(train_df[feature_cols], train_df[TARGET_COL],
                  eval_X=val_df[feature_cols], eval_y=val_df[TARGET_COL],
                  eval_metric=TUNING_OBJECTIVE_METRIC,
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                             lgb.log_evaluation(0)])
        predictions = model.predict(val_df[feature_cols])
        residuals = val_df[TARGET_COL].values - predictions
        fold_rmses.append(np.sqrt(np.mean(residuals ** 2)))
    return float(np.mean(fold_rmses))


def run_search(df, feature_cols, n_trials, n_inner_folds, learning_rate,
               max_boosting_rounds, early_stopping_rounds, seed, label):
    def objective(trial):
        params = suggest_params(trial, learning_rate, max_boosting_rounds, seed)
        return inner_cv_rmse(params, df, feature_cols, n_inner_folds, early_stopping_rounds)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=label)
    study.optimize(objective, n_trials=n_trials)

    print(f"{label}: best inner-CV {TUNING_OBJECTIVE_METRIC}={study.best_value:.4f} "
          f"after {len(study.trials)} trials")
    print(f"{label}: best params {study.best_params}")
    return study


def trials_dataframe(study, label):
    rows = []
    for trial in study.trials:
        row = {"search": label, "trial": trial.number, "objective_rmse": trial.value}
        row.update(trial.params)
        rows.append(row)
    return pd.DataFrame(rows)


def log_search_to_mlflow(run_name, study, label, extra_params):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "tuning")
        mlflow.log_param("tuning_objective_metric", TUNING_OBJECTIVE_METRIC)
        mlflow.log_param("n_trials", len(study.trials))
        for key, value in extra_params.items():
            mlflow.log_param(key, value)
        for key, value in study.best_params.items():
            mlflow.log_param("best_" + key, value)
        for trial in study.trials:
            if trial.value is not None:
                mlflow.log_metric("trial_objective_rmse", trial.value, step=trial.number)
        mlflow.log_metric("best_objective_rmse", study.best_value)
        print(f"Logged MLflow run '{run_name}' ({label})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="lightgbm_ready_dataset.csv")
    parser.add_argument("--station_file", required=True, help="cpcb_stations_delhi_status.csv")
    parser.add_argument("--buffer_km", required=True, type=float)
    parser.add_argument("--n_blocks", required=True, type=int)
    parser.add_argument("--n_inner_folds", required=True, type=int)
    parser.add_argument("--n_trials", required=True, type=int)
    parser.add_argument("--learning_rate", required=True, type=float)
    parser.add_argument("--max_boosting_rounds", required=True, type=int)
    parser.add_argument("--early_stopping_rounds", required=True, type=int)
    parser.add_argument("--random_seed", required=True, type=int)
    parser.add_argument("--output", required=True, help="tuned hyperparameters json")
    parser.add_argument("--block_output", required=True, help="station block assignment csv")
    parser.add_argument("--trials_output", required=True, help="all Optuna trials csv")
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for outpath in [args.output, args.block_output, args.trials_output]:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    df = pd.read_csv(args.input)
    df = cast_season_categorical(df)
    feature_cols = build_feature_list(df)
    print(f"Loaded {args.input}: {len(df)} rows, {df[GROUP_COL].nunique()} stations, "
          f"{len(feature_cols)} features")
    print(f"Features: {feature_cols}")

    stations = pd.read_csv(args.station_file)
    keep_stations = stations[stations["status"] == "KEEP"].reset_index(drop=True)
    neighbors = compute_buffer_neighbors(keep_stations, args.buffer_km)

    station_ids = sorted(int(x) for x in df[GROUP_COL].unique())
    blocks = build_station_blocks(station_ids, neighbors, args.n_blocks, args.random_seed)

    print(f"=== Tuning blocks ({args.n_blocks} blocks, {args.buffer_km} km clusters kept intact) ===")
    block_rows = []
    for block_index, block in enumerate(blocks):
        print(f"Block {block_index + 1}: {len(block)} stations {block}")
        for station_id in block:
            block_rows.append({"location_id": station_id, "block": block_index + 1})
    block_df = pd.DataFrame(block_rows).sort_values("location_id")
    block_df.to_csv(args.block_output, index=False)
    print(f"Wrote {args.block_output}")

    tuned = {
        "tuning_objective_metric": TUNING_OBJECTIVE_METRIC,
        "learning_rate": args.learning_rate,
        "max_boosting_rounds": args.max_boosting_rounds,
        "early_stopping_rounds": args.early_stopping_rounds,
        "n_inner_folds": args.n_inner_folds,
        "n_trials": args.n_trials,
        "random_seed": args.random_seed,
        "buffer_km": args.buffer_km,
        "feature_cols": feature_cols,
        "blocks": {},
        "block_hyperparameters": {},
        "block_objective_rmse": {},
        "full_data_hyperparameters": {},
    }
    all_trials = []

    print("=== Block-nested tuning searches ===")
    for block_index, block in enumerate(blocks):
        label = f"block_{block_index + 1}"
        # The entire inner-CV universe for this search is the OTHER blocks'
        # stations. No station in this block appears anywhere in the search --
        # not as training data, not as inner-validation data -- so the
        # hyperparameters later used to evaluate it were never influenced by
        # its own rows.
        search_df = df[~df[GROUP_COL].isin(block)].reset_index(drop=True)
        print(f"{label}: tuning on {search_df[GROUP_COL].nunique()} stations "
              f"({len(search_df)} rows), holding out {len(block)} stations {block}")
        study = run_search(search_df, feature_cols, args.n_trials, args.n_inner_folds,
                           args.learning_rate, args.max_boosting_rounds,
                           args.early_stopping_rounds, args.random_seed + block_index, label)
        tuned["blocks"][label] = block
        tuned["block_hyperparameters"][label] = study.best_params
        tuned["block_objective_rmse"][label] = study.best_value
        all_trials.append(trials_dataframe(study, label))
        log_search_to_mlflow(
            run_name="tune_" + label,
            study=study,
            label=label,
            extra_params={
                "block": block_index + 1,
                "held_out_stations": block,
                "n_tuning_stations": search_df[GROUP_COL].nunique(),
                "n_tuning_rows": len(search_df),
                "n_inner_folds": args.n_inner_folds,
                "inner_cv_scheme": "GroupKFold by location_id",
                "learning_rate": args.learning_rate,
            },
        )

    print("=== Full-data tuning search (production model, no station held out) ===")
    # There is no held-out test set left to protect once the deliverable model
    # is being built, so this search sees all 42 stations. Its winner is used
    # only by 01_fit_lightgbm_model.py and by the random-CV arm of
    # 02_validate_lightgbm_cv.py (whose folds are row-level and therefore
    # already leaky by construction) -- never by the spatial-LOSO arm, which is
    # the number that actually measures generalization.
    full_study = run_search(df, feature_cols, args.n_trials, args.n_inner_folds,
                            args.learning_rate, args.max_boosting_rounds,
                            args.early_stopping_rounds, args.random_seed + args.n_blocks,
                            "full_data")
    tuned["full_data_hyperparameters"] = full_study.best_params
    tuned["full_data_objective_rmse"] = full_study.best_value
    all_trials.append(trials_dataframe(full_study, "full_data"))
    log_search_to_mlflow(
        run_name="tune_full_data",
        study=full_study,
        label="full_data",
        extra_params={
            "block": "none (all 42 stations)",
            "held_out_stations": "none",
            "n_tuning_stations": df[GROUP_COL].nunique(),
            "n_tuning_rows": len(df),
            "n_inner_folds": args.n_inner_folds,
            "inner_cv_scheme": "GroupKFold by location_id",
            "learning_rate": args.learning_rate,
        },
    )

    with open(args.output, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"Wrote {args.output}")

    trials_df = pd.concat(all_trials, ignore_index=True)
    trials_df.to_csv(args.trials_output, index=False)
    print(f"Wrote {args.trials_output}: {len(trials_df)} trials across "
          f"{args.n_blocks + 1} searches")

    print("=== Tuning summary ===")
    for label in tuned["block_hyperparameters"]:
        print(f"{label}: inner-CV {TUNING_OBJECTIVE_METRIC}="
              f"{tuned['block_objective_rmse'][label]:.4f} "
              f"params={tuned['block_hyperparameters'][label]}")
    print(f"full_data: inner-CV {TUNING_OBJECTIVE_METRIC}="
          f"{tuned['full_data_objective_rmse']:.4f} "
          f"params={tuned['full_data_hyperparameters']}")


if __name__ == "__main__":
    main()
