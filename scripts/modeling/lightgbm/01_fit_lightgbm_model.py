import argparse
import json
import os

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score

# Same feature set as scripts/modeling/lightgbm/00_tune_lightgbm_hyperparameters.py
# and 02_validate_lightgbm_cv.py -- kept as a separate copy here (repo
# convention: no shared utils module across scripts), must be kept in sync if
# the feature set changes. Every column in lightgbm_ready_dataset.csv except
# the 3 ID/meta columns and the target.
ID_COLS = ["location_id", "name", "date"]
TARGET_COL = "pm25_daily"
GROUP_COL = "location_id"

SEASON_COL = "season"
SEASON_CATEGORIES = ["summer", "monsoon", "post_monsoon", "winter"]

# Fixed LightGBM params, applied on top of whichever tuned hyperparameter set
# is loaded. learning_rate and the boosting-round cap come from the CLI so
# they stay identical to the values the tuning search ran under.
FIXED_PARAMS = {"objective": "regression", "metric": "rmse", "verbosity": -1}

MLFLOW_EXPERIMENT = "delhi_phase1_lightgbm"
MLFLOW_RUN_NAME_BASE = "full_data_fit"
# Registers the primary full-data fit as a numbered version of this model in
# the MLflow Model Registry, mirroring the LME module's delhi_phase1_lme
# entry. A new experiment name (not delhi_phase1_lme) because LightGBM and the
# LME are different model families and should not share tracking history.
MLFLOW_REGISTERED_MODEL_NAME = "delhi_phase1_lightgbm"


def build_feature_list(df):
    return [col for col in df.columns if col not in ID_COLS and col != TARGET_COL]


def cast_season_categorical(df):
    # Explicit fixed category order so the codes LightGBM sees are stable
    # across this script, the tuning script and the validation script.
    df = df.copy()
    df[SEASON_COL] = pd.Categorical(df[SEASON_COL], categories=SEASON_CATEGORIES)
    return df


def check_boosting_settings_match(tuned, args):
    # learning_rate / max_boosting_rounds / early_stopping_rounds live in
    # params.yaml once per stage, so the tuning stage and this one could drift
    # apart silently. A tuned hyperparameter set is only valid for the
    # learning rate it was searched under (num_leaves and the regularization
    # terms trade off directly against it), so a mismatch is a hard failure
    # rather than a warning. The tuning script records the values it ran under
    # in its output json, which makes the check possible at all.
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


def build_params(tuned_params, learning_rate, max_boosting_rounds, seed):
    params = dict(FIXED_PARAMS)
    params.update(tuned_params)
    params["learning_rate"] = learning_rate
    params["n_estimators"] = max_boosting_rounds
    params["random_state"] = seed
    return params


def split_validation_stations(df, validation_station_fraction, seed):
    # Early stopping needs a validation slice that the model does not train
    # on. Split it off by STATION, not by random rows: 13 of the 23 features
    # are constant within a station, so a random row split would put near-
    # copies of each training row in the validation slice and early stopping
    # would then never trigger until the model had memorized station
    # baselines. Only used to discover the boosting-round count -- the final
    # model is refit on 100% of the rows (see main()).
    station_ids = np.array(sorted(df[GROUP_COL].unique()))
    n_validation = max(1, int(round(len(station_ids) * validation_station_fraction)))
    rng = np.random.RandomState(seed)
    validation_ids = set(rng.choice(station_ids, size=n_validation, replace=False))
    train_df = df[~df[GROUP_COL].isin(validation_ids)]
    validation_df = df[df[GROUP_COL].isin(validation_ids)]
    return train_df, validation_df, sorted(int(x) for x in validation_ids)


def find_best_iteration(params, train_df, validation_df, feature_cols, early_stopping_rounds):
    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[feature_cols], train_df[TARGET_COL],
              eval_X=validation_df[feature_cols], eval_y=validation_df[TARGET_COL],
              eval_metric="rmse",
              callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                         lgb.log_evaluation(0)])
    return model.best_iteration_


def fit_final_model(params, df, feature_cols, n_estimators):
    params = dict(params)
    params["n_estimators"] = n_estimators
    model = lgb.LGBMRegressor(**params)
    model.fit(df[feature_cols], df[TARGET_COL])
    return model


def compute_point_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae}


def build_importance_table(model, feature_cols):
    # Gain-based importance is this model's analog of the LME's fixed-effect
    # coefficient table -- reported from the full-data fit only, not per CV
    # fold. Split counts are reported alongside because a feature can rank
    # high on gain from a handful of very profitable splits, or on splits from
    # being used constantly for small refinements, and the two tell different
    # stories.
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    table = pd.DataFrame({
        "feature": feature_cols,
        "gain": gain,
        "gain_pct": 100.0 * gain / gain.sum(),
        "split": split,
    })
    table = table.sort_values("gain", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="lightgbm_ready_dataset.csv")
    parser.add_argument("--hyperparameters", required=True,
                         help="tuned hyperparameters json from "
                              "00_tune_lightgbm_hyperparameters.py")
    parser.add_argument("--model_output", required=True,
                         help="fitted booster, LightGBM native text format")
    parser.add_argument("--importance_output", required=True, help="feature importance csv")
    parser.add_argument("--summary_output", required=True, help="model summary text file")
    parser.add_argument("--learning_rate", required=True, type=float)
    parser.add_argument("--max_boosting_rounds", required=True, type=int)
    parser.add_argument("--early_stopping_rounds", required=True, type=int)
    parser.add_argument("--validation_station_fraction", required=True, type=float,
                         help="fraction of stations held out purely to early-stop on, "
                              "for discovering the boosting-round count. The final model "
                              "is refit on all rows with that count.")
    parser.add_argument("--random_seed", required=True, type=int)
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

    for outpath in [args.model_output, args.importance_output, args.summary_output]:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(args.hyperparameters) as f:
        tuned = json.load(f)
    tuned_params = tuned["full_data_hyperparameters"]
    print(f"Loaded {args.hyperparameters}")
    print(f"Full-data tuned hyperparameters: {tuned_params}")
    print(f"Full-data tuning inner-CV rmse: {tuned['full_data_objective_rmse']}")

    df = pd.read_csv(args.input)
    df = cast_season_categorical(df)
    feature_cols = build_feature_list(df)
    print(f"Loaded {args.input}: {len(df)} rows, {df[GROUP_COL].nunique()} stations, "
          f"{len(feature_cols)} features")

    if feature_cols != tuned["feature_cols"]:
        raise SystemExit("Feature list does not match the one the hyperparameters were "
                         "tuned on -- re-run 00_tune_lightgbm_hyperparameters.py")

    check_boosting_settings_match(tuned, args)

    params = build_params(tuned_params, args.learning_rate, args.max_boosting_rounds,
                          args.random_seed)

    print("=== Boosting-round selection (early stopping on held-out stations) ===")
    train_df, validation_df, validation_ids = split_validation_stations(
        df, args.validation_station_fraction, args.random_seed)
    print(f"Early-stopping validation stations ({len(validation_ids)}): {validation_ids}")
    print(f"Train rows: {len(train_df)}, validation rows: {len(validation_df)}")
    best_iteration = find_best_iteration(params, train_df, validation_df, feature_cols,
                                         args.early_stopping_rounds)
    print(f"Early stopping selected {best_iteration} boosting rounds "
          f"(cap was {args.max_boosting_rounds})")

    print("=== Final fit on all rows ===")
    # Refit on 100% of the data with the round count discovered above, rather
    # than keeping the early-stopped model: the validation stations are not a
    # held-out test set that needs protecting here (this is the deliverable
    # model, not a generalization measurement), so there is no reason to throw
    # away their rows.
    model = fit_final_model(params, df, feature_cols, best_iteration)
    predictions = model.predict(df[feature_cols])
    in_sample = compute_point_metrics(df[TARGET_COL].values, predictions)
    print(f"In-sample (not a generalization estimate -- see "
          f"02_validate_lightgbm_cv.py): R2={in_sample['r2']:.3f} "
          f"RMSE={in_sample['rmse']:.3f} MAE={in_sample['mae']:.3f} ug/m3")

    importance_table = build_importance_table(model, feature_cols)
    print("=== Feature importance (gain-based, top 10) ===")
    for row in importance_table.head(10).itertuples(index=False):
        print(f"  {row.rank}. {row.feature}: gain={row.gain:.1f} "
              f"({row.gain_pct:.1f}%) splits={row.split}")

    model.booster_.save_model(args.model_output)
    print(f"Wrote {args.model_output}")

    importance_table.to_csv(args.importance_output, index=False)
    print(f"Wrote {args.importance_output}")

    with open(args.summary_output, "w") as f:
        f.write("LightGBM full-data fit (Delhi Phase 1)\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Stations: {df[GROUP_COL].nunique()}\n")
        f.write(f"Target: {TARGET_COL} (raw ug/m3, no transform)\n")
        f.write(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
        f.write(f"Categorical features: {SEASON_COL}\n\n")
        f.write("Hyperparameters (full-data Optuna search):\n")
        for key in sorted(tuned_params):
            f.write(f"  {key}: {tuned_params[key]}\n")
        f.write(f"  learning_rate: {args.learning_rate}\n")
        f.write(f"  n_estimators (early stopping): {best_iteration}\n")
        f.write(f"  max_boosting_rounds (cap): {args.max_boosting_rounds}\n")
        f.write(f"  early_stopping_rounds: {args.early_stopping_rounds}\n")
        f.write(f"  random_state: {args.random_seed}\n\n")
        f.write(f"Early-stopping validation stations: {validation_ids}\n")
        f.write(f"Full-data tuning inner-CV rmse: {tuned['full_data_objective_rmse']}\n\n")
        f.write("In-sample metrics (NOT a generalization estimate):\n")
        f.write(f"  r2: {in_sample['r2']}\n")
        f.write(f"  rmse_ugm3: {in_sample['rmse']}\n")
        f.write(f"  mae_ugm3: {in_sample['mae']}\n\n")
        f.write("Feature importance (gain-based):\n")
        for row in importance_table.itertuples(index=False):
            f.write(f"  {row.rank}. {row.feature}: gain={row.gain} "
                    f"({row.gain_pct:.2f}%) splits={row.split}\n")
    print(f"Wrote {args.summary_output}")

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME_BASE):
        mlflow.set_tag("stage", "fit")
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("target_transform", "none (raw ug/m3)")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("categorical_features", SEASON_COL)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_stations", df[GROUP_COL].nunique())
        mlflow.log_param("hyperparameter_source", "full_data optuna search")
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("n_estimators", best_iteration)
        mlflow.log_param("max_boosting_rounds", args.max_boosting_rounds)
        mlflow.log_param("early_stopping_rounds", args.early_stopping_rounds)
        mlflow.log_param("early_stopping_validation_stations", validation_ids)
        mlflow.log_param("random_seed", args.random_seed)
        for key in sorted(tuned_params):
            mlflow.log_param(key, tuned_params[key])
        mlflow.log_metric("in_sample_r2", in_sample["r2"])
        mlflow.log_metric("in_sample_rmse_ugm3", in_sample["rmse"])
        mlflow.log_metric("in_sample_mae_ugm3", in_sample["mae"])
        mlflow.log_metric("best_iteration", best_iteration)
        mlflow.log_metric("tuning_inner_cv_rmse", tuned["full_data_objective_rmse"])
        for row in importance_table.itertuples(index=False):
            mlflow.log_metric("gain_" + row.feature, row.gain)
        mlflow.log_artifact(args.importance_output)
        mlflow.log_artifact(args.summary_output)
        mlflow.log_artifact(args.model_output)
        mlflow.lightgbm.log_model(model.booster_, name="model")

        run_id = mlflow.active_run().info.run_id
        registered = mlflow.register_model(f"runs:/{run_id}/model", MLFLOW_REGISTERED_MODEL_NAME)
        print(f"Registered model '{MLFLOW_REGISTERED_MODEL_NAME}' version {registered.version}")
        print(f"Logged run to MLflow experiment: {MLFLOW_EXPERIMENT}")


if __name__ == "__main__":
    main()
