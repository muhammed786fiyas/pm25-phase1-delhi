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

# statsmodels re_formula for the random-slope-AOD variant: adds a per-station
# random slope for aod_055 alongside the random intercept (correlated 2x2
# covariance by default), matching the project blueprint's
# (b0+u0i) + (b1+u1i)*AOD equation variant -- see docs/logs/tasks/9-LME_Model.md.
# Only the base aod_055 term gets a random slope, not the season-interaction
# AOD columns (aod_x_monsoon etc.) -- those stay pooled fixed effects.
RE_FORMULA_RANDOM_SLOPE_AOD = "~ aod_055"

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
# robustness / AOD winsorized / excl outlier stations / random-slope-AOD) are
# NOT registered -- they exist for comparison, not as a candidate "the"
# model, and mixing a genuinely different equation spec (random-slope-AOD)
# into the same version history as the random-intercept-only primary fit
# would make the registry actively misleading, not just ambiguous. Registry
# works on this repo's plain file-store tracking URI in the installed MLflow
# version (verified 2026-09-09) -- older MLflow versions required a
# database-backed store for this.
MLFLOW_REGISTERED_MODEL_NAME = "delhi_phase1_lme"


def build_formula():
    return TARGET_COL + " ~ " + " + ".join(FIXED_EFFECTS)


def fit_lme(df, reml, re_formula=None):
    # re_formula=None -- statsmodels defaults to a random intercept only,
    # grouped by whatever is passed to `groups` (location_id here).
    # re_formula="~ aod_055" (RE_FORMULA_RANDOM_SLOPE_AOD) adds a random
    # slope for aod_055 alongside the intercept instead.
    formula = build_formula()
    model = smf.mixedlm(formula, data=df, groups=df[GROUP_COL], re_formula=re_formula)
    result = model.fit(reml=reml)
    return result


def compute_aic_bic(result):
    # MixedLMResults.aic / .bic come back as NaN in this statsmodels version
    # (df_modelwc isn't populated for this model class), so compute by hand:
    # k = fixed-effect params + variance components. n_variance_components
    # generalizes to any number of random effects (n_re): n_re*(n_re+1)//2
    # unique entries in the symmetric cov_re matrix, plus 1 for the residual
    # variance. Random-intercept-only: n_re=1 -> 2 total. Random-intercept +
    # random-slope-AOD: n_re=2 -> 4 total (2 variances + 1 covariance + 1
    # residual).
    n_re = result.cov_re.shape[0]
    n_variance_components = n_re * (n_re + 1) // 2 + 1
    k_total = result.k_fe + n_variance_components
    n = result.nobs
    aic = -2 * result.llf + 2 * k_total
    bic = -2 * result.llf + k_total * np.log(n)
    return aic, bic, k_total


def variance_component_rows(cov_re):
    # Flattens the random-effects covariance matrix (1x1 for random-intercept
    # -only, 2x2 for random-intercept + random-slope-AOD) into (label, value)
    # pairs: one row per variance (diagonal) and one row per covariance
    # (off-diagonal, only present with more than one random effect). Keeps
    # reporting/logging generic across both model variants instead of
    # hardcoding "the one random-intercept variance".
    rows = []
    labels = list(cov_re.index)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if j < i:
                continue
            if i == j:
                rows.append((f"{li}_variance", cov_re.iloc[i, j]))
            else:
                rows.append((f"{li}_{lj}_covariance", cov_re.iloc[i, j]))
    return rows


def residual_heteroscedasticity_table(result, n_quintiles=5):
    # Residual variance across fitted-value quintiles. The LME needs roughly
    # constant residual variance for its standard errors / p-values / CIs to
    # be valid, and docs/logs/tasks/8-Modeling_Data_Prep.md section 9 measured
    # a ~12.8x growth across this range on the RAW target (vs a flat profile
    # on the log target) -- which is exactly why target_transform=log was
    # chosen. That measurement was made on an exploratory pre-fit, so this
    # recomputes it from the actual fitted model, for whichever target the fit
    # used. A large ratio does NOT invalidate the point predictions or the
    # predictive CV metrics -- only the inference (SEs, p-values, CIs).
    fitted = result.fittedvalues
    residuals = result.resid
    quintile = pd.qcut(fitted, n_quintiles, labels=False, duplicates="drop")
    rows = []
    for q in sorted(pd.unique(quintile)):
        mask = quintile == q
        rows.append({
            "fitted_quintile": int(q) + 1,
            "n": int(mask.sum()),
            "fitted_min": fitted[mask].min(),
            "fitted_max": fitted[mask].max(),
            "residual_variance": residuals[mask].var(),
        })
    table = pd.DataFrame(rows)
    ratio = table["residual_variance"].max() / table["residual_variance"].min()
    return table, ratio


def build_coef_table(result):
    # result.params/.bse/.tvalues/.pvalues/.conf_int() all include the
    # variance-component terms ("Group Var", and with a random slope also
    # "aod_055 Var"/"Group x aod_055 Cov") alongside the fixed effects --
    # filter down to fe_params.index to report fixed effects only here. The
    # variance components are reported separately (they aren't fixed
    # effects, and don't get p-values the same way -- see the project's
    # M0/A/B/C summary doc on why BLUP-adjacent variance terms aren't
    # ordinary fixed-effect estimates).
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
    parser.add_argument("--random_slope_aod", default="false", choices=["true", "false"],
                         help="true fits the random-slope-AOD equation variant: adds a "
                              "per-station random slope for aod_055 alongside the random "
                              "intercept (re_formula='~ aod_055'), instead of the primary "
                              "random-intercept-only model. Default: false.")
    parser.add_argument("--target_transform", default="log", choices=["log", "raw"],
                         help="Which scale modeling_target is on in --input. 'log' is the "
                              "primary dataset (modeling_target = log(pm25_daily)); 'raw' is "
                              "the raw-target comparison variant built by the "
                              "prepare_lme_dataset_raw_target stage. Only affects reporting "
                              "and the AIC/BIC comparability warning -- the fit itself just "
                              "uses whatever modeling_target holds. Default: log.")
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

    random_slope_aod = args.random_slope_aod == "true"
    re_formula = RE_FORMULA_RANDOM_SLOPE_AOD if random_slope_aod else None
    if random_slope_aod:
        run_name_suffix += "_random_slope_aod"

    if args.run_tag:
        run_name_suffix += "_" + args.run_tag

    # Only the primary fit (no exclusions, no run_tag, random-intercept-only)
    # is registered to the Model Registry -- see MLFLOW_REGISTERED_MODEL_NAME
    # comment above.
    is_primary = (not exclude_low_confidence and not exclude_station_ids
                  and not args.run_tag and not random_slope_aod
                  and args.target_transform == "log")

    re_description = "random intercept + random AOD slope by station" if random_slope_aod \
        else "random intercept only by station"
    print(f"Fitting MixedLM (REML=True, {re_description})...")
    result = fit_lme(df, reml=True, re_formula=re_formula)
    print(result.summary())

    aic, bic, k_total = compute_aic_bic(result)
    variance_rows = variance_component_rows(result.cov_re)
    residual_var = result.scale

    print("=== Model fit summary ===")
    print(f"target_transform: {args.target_transform}")
    print(f"logLik: {result.llf}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    for label, value in variance_rows:
        print(f"{label}: {value}")
    print(f"Residual variance: {residual_var}")
    print(f"Total parameters (fixed effects + variance components): {k_total}")
    print("NOTE: logLik/AIC/BIC are only comparable between fits on the SAME "
          "target_transform -- a log-target and a raw-target fit have different "
          "response variables and so different likelihood scales. Never table "
          "this AIC against one from the other transform.")

    hetero_table, hetero_ratio = residual_heteroscedasticity_table(result)
    print("=== Residual variance by fitted-value quintile (heteroscedasticity check) ===")
    for row in hetero_table.itertuples(index=False):
        print(f"quintile {row.fitted_quintile} (n={row.n}, fitted "
              f"{row.fitted_min:.3f} to {row.fitted_max:.3f}): "
              f"residual variance {row.residual_variance:.4f}")
    print(f"Max/min residual variance ratio: {hetero_ratio:.2f} "
          f"(near 1 is homoscedastic; a large ratio invalidates the standard "
          f"errors/p-values in the coefficient table, but NOT the point "
          f"predictions or the CV metrics)")

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
        f.write(f"target_transform: {args.target_transform}\n")
        f.write(f"logLik: {result.llf}\n")
        f.write(f"AIC: {aic}\n")
        f.write(f"BIC: {bic}\n")
        for label, value in variance_rows:
            f.write(f"{label}: {value}\n")
        f.write(f"Residual variance: {residual_var}\n")
        f.write("\nNOTE: logLik/AIC/BIC are only comparable between fits on the same\n")
        f.write("target_transform -- a log-target and a raw-target fit have different\n")
        f.write("response variables and so different likelihood scales.\n")
        f.write("\nResidual variance by fitted-value quintile (heteroscedasticity check):\n")
        for row in hetero_table.itertuples(index=False):
            f.write(f"  quintile {row.fitted_quintile} (n={row.n}, fitted "
                    f"{row.fitted_min:.4f} to {row.fitted_max:.4f}): "
                    f"residual variance {row.residual_variance}\n")
        f.write(f"  max/min ratio: {hetero_ratio}\n")
        f.write("  A large ratio invalidates the standard errors/p-values in the\n")
        f.write("  coefficient table above, but not the point predictions or the CV metrics.\n")
    print(f"Wrote {args.summary_output}")

    with mlflow.start_run(run_name=MLFLOW_RUN_NAME_BASE + run_name_suffix):
        mlflow.log_param("formula", build_formula())
        mlflow.log_param("re_formula", re_formula or "none (random intercept only)")
        mlflow.log_param("reml", True)
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_stations", df[GROUP_COL].nunique())
        mlflow.log_param("exclude_low_confidence", exclude_low_confidence)
        mlflow.log_param("exclude_stations", sorted(exclude_station_ids) if exclude_station_ids else "none")
        mlflow.log_param("random_slope_aod", random_slope_aod)
        mlflow.log_param("target_transform", args.target_transform)
        mlflow.log_param("run_tag", args.run_tag or "none")
        mlflow.log_metric("log_likelihood", result.llf)
        mlflow.log_metric("aic", aic)
        mlflow.log_metric("bic", bic)
        for label, value in variance_rows:
            mlflow.log_metric(label, value)
        mlflow.log_metric("residual_variance", residual_var)
        mlflow.log_metric("residual_variance_quintile_ratio", hetero_ratio)
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
            print("Diagnostic/alternate-equation variant run -- not registered to the Model Registry")

        print("Logged run to MLflow experiment: " +
              (MLFLOW_EXPERIMENT_GAPFILL_ROBUSTNESS if exclude_low_confidence else MLFLOW_EXPERIMENT_FULL))


if __name__ == "__main__":
    main()
