import argparse
import os
import pickle

import mlflow
import mlflow.statsmodels
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from dotenv import load_dotenv

# Fixed effects for the Delhi Phase 1 LME equation:
# PM2.5 = b0 + (b1 + b_int*Season)*AOD + b_met*Met + b_lu*LandUse + b_s*Season + u_i + eps
# The equation only names "Met" and "LandUse" as broad groups. Met = the 4 ERA5
# meteorology columns. LandUse is used broadly here to mean every other
# non-meteorology, non-AOD, non-season covariate in lme_ready_dataset.csv:
# WorldCover land-cover fractions, NDVI, SRTM terrain (elevation/slope), and the
# OSM covariates (road density, industrial fraction, powerplant distance) -- none
# of these are meteorological, and the project context doc's equation has no
# separate bucket for them.
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
GROUP_COL = "location_id"

# Same experiment split as scripts/modeling/lme/02_validate_lme_cv.py: the
# gapfill-robustness variant (confidence_bucket == low excluded) always logs
# to its own MLflow experiment, never mixed into the primary one.
MLFLOW_EXPERIMENT_FULL = "delhi_phase1_lme"
MLFLOW_EXPERIMENT_GAPFILL_ROBUSTNESS = "delhi_phase1_lme_gapfill_robustness"
MLFLOW_RUN_NAME_BASE = "full_data_reml_fit"
# Registers each PRIMARY run's model.statsmodels artifact as a new numbered
# version of this same named model in the MLflow Model Registry, so the
# fitted model can be referenced as "models:/delhi_phase1_lme/<version>" (or
# "/latest") instead of only by run_id. Diagnostic variants (gapfill
# robustness / AOD winsorized / excl outlier stations) are NOT registered --
# they exist for comparison, not as a candidate "the" model, and mixing them
# into the same version history as the primary fit would make the registry
# ambiguous about which version is production-representative. Registry works
# on this repo's plain file-store tracking URI in the installed MLflow
# version (verified 2026-09-09) -- older MLflow versions required a
# database-backed store for this.
MLFLOW_REGISTERED_MODEL_NAME = "delhi_phase1_lme"


def build_formula():
    return TARGET_COL + " ~ " + " + ".join(FIXED_EFFECTS)


def fit_lme(df, reml):
    # No re_formula given -- statsmodels defaults to a random intercept only,
    # grouped by whatever is passed to `groups` (location_id here).
    formula = build_formula()
    model = smf.mixedlm(formula, data=df, groups=df[GROUP_COL])
    result = model.fit(reml=reml)
    return result


def compute_aic_bic(result):
    # MixedLMResults.aic / .bic come back as NaN in this statsmodels version
    # (df_modelwc isn't populated for this model class), so compute by hand:
    # k = fixed-effect params + variance components. This model has one
    # variance component (the random-intercept variance) plus the residual
    # variance, so n_variance_components = 1 (cov_re, 1x1) + 1 (scale) = 2.
    n_re = result.cov_re.shape[0]
    n_variance_components = n_re * (n_re + 1) // 2 + 1
    k_total = result.k_fe + n_variance_components
    n = result.nobs
    aic = -2 * result.llf + 2 * k_total
    bic = -2 * result.llf + k_total * np.log(n)
    return aic, bic, k_total


def build_coef_table(result):
    # result.params/.bse/.tvalues/.pvalues/.conf_int() all include the "Group
    # Var" random-effect variance term alongside the fixed effects -- filter
    # down to fe_params.index to report fixed effects only here. The random-
    # intercept variance is reported separately (it isn't a fixed effect).
    fe_names = result.fe_params.index
    ci = result.conf_int()
    coef_table = pd.DataFrame({
        "term": fe_names,
        "coef": result.params.loc[fe_names].values,
        "std_err": result.bse.loc[fe_names].values,
        "z": result.tvalues.loc[fe_names].values,
        "p_value": result.pvalues.loc[fe_names].values,
        "ci_lower": ci.loc[fe_names, 0].values,
        "ci_upper": ci.loc[fe_names, 1].values,
    })
    return coef_table


def log_coefficients_to_mlflow(coef_table):
    for row in coef_table.itertuples(index=False):
        safe_term = row.term.replace(":", "_")
        mlflow.log_metric("coef_" + safe_term, row.coef)
        mlflow.log_metric("se_" + safe_term, row.std_err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="lme_ready_dataset.csv")
    parser.add_argument("--model_output", required=True, help="pickled fitted MixedLMResults")
    parser.add_argument("--coef_output", required=True, help="fixed-effect coefficient table csv")
    parser.add_argument("--summary_output", required=True, help="full model summary text file")
    parser.add_argument("--exclude_low_confidence", default="false", choices=["true", "false"],
                         help="true fits on the gap-fill robustness variant "
                              "(excludes confidence_bucket == low), logged to a separate "
                              "MLflow experiment, matching 02_validate_lme_cv.py. "
                              "Default: false (primary fit, all data).")
    parser.add_argument("--exclude_stations", default="",
                         help="Comma-separated location_ids to drop entirely before fitting "
                              "(e.g. for a with/without outlier-station comparison fit). "
                              "Default: none excluded.")
    parser.add_argument("--run_tag", default="",
                         help="Optional tag appended to the MLflow run name, so a diagnostic "
                              "variant (e.g. a fit against a winsorized --input dataset) "
                              "doesn't collide with the primary run in the same MLflow "
                              "experiment. Default: no tag.")
    args = parser.parse_args()

    load_dotenv()
    # MLflow 3.x refuses the plain filesystem tracking backend by default
    # (it's in "maintenance mode") unless this is set -- the repo's tracking
    # URI convention (file:./models/mlflow_tracking) needs this to work.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("No MLFLOW_TRACKING_URI found. Set it in .env")
    mlflow.set_tracking_uri(tracking_uri)

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

    if args.run_tag:
        run_name_suffix += "_" + args.run_tag

    # Only the primary fit (no exclusions, no run_tag) is registered to the
    # Model Registry -- see MLFLOW_REGISTERED_MODEL_NAME comment above.
    is_primary = not exclude_low_confidence and not exclude_station_ids and not args.run_tag

    print("Fitting MixedLM (REML=True, random intercept by station)...")
    result = fit_lme(df, reml=True)
    print(result.summary())

    aic, bic, k_total = compute_aic_bic(result)
    group_var = result.cov_re.iloc[0, 0]
    residual_var = result.scale

    print("=== Model fit summary ===")
    print(f"logLik: {result.llf}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    print(f"Random-intercept variance (station): {group_var}")
    print(f"Residual variance: {residual_var}")
    print(f"Total parameters (fixed effects + variance components): {k_total}")

    coef_table = build_coef_table(result)

    for outpath in [args.model_output, args.coef_output, args.summary_output]:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(args.model_output, "wb") as f:
        pickle.dump(result, f)
    print(f"Wrote {args.model_output}")

    coef_table.to_csv(args.coef_output, index=False)
    print(f"Wrote {args.coef_output}")

    with open(args.summary_output, "w") as f:
        f.write(str(result.summary()))
        f.write("\n\n")
        f.write(f"logLik: {result.llf}\n")
        f.write(f"AIC: {aic}\n")
        f.write(f"BIC: {bic}\n")
        f.write(f"Random-intercept variance (station): {group_var}\n")
        f.write(f"Residual variance: {residual_var}\n")
    print(f"Wrote {args.summary_output}")

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME_BASE + run_name_suffix):
        mlflow.log_param("formula", build_formula())
        mlflow.log_param("reml", True)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_stations", df[GROUP_COL].nunique())
        mlflow.log_param("exclude_low_confidence", exclude_low_confidence)
        mlflow.log_param("exclude_stations", sorted(exclude_station_ids) if exclude_station_ids else "none")
        mlflow.log_param("run_tag", args.run_tag or "none")
        mlflow.log_metric("log_likelihood", result.llf)
        mlflow.log_metric("aic", aic)
        mlflow.log_metric("bic", bic)
        mlflow.log_metric("random_intercept_variance", group_var)
        mlflow.log_metric("residual_variance", residual_var)
        log_coefficients_to_mlflow(coef_table)
        mlflow.log_artifact(args.coef_output)
        mlflow.log_artifact(args.summary_output)
        mlflow.log_artifact(args.model_output)
        mlflow.statsmodels.log_model(result, name="model")

        if is_primary:
            run_id = mlflow.active_run().info.run_id
            registered = mlflow.register_model(f"runs:/{run_id}/model", MLFLOW_REGISTERED_MODEL_NAME)
            print(f"Registered model '{MLFLOW_REGISTERED_MODEL_NAME}' version {registered.version}")
        else:
            print("Diagnostic variant run -- not registered to the Model Registry")

        print("Logged run to MLflow experiment: " +
              (MLFLOW_EXPERIMENT_GAPFILL_ROBUSTNESS if exclude_low_confidence else MLFLOW_EXPERIMENT_FULL))


if __name__ == "__main__":
    main()
