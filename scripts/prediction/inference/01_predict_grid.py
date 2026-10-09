import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

# Runs the trained booster over every prediction grid cell-day and writes the
# estimate with its conformal interval and an extrapolation flag.
#
# Predictions are PRECOMPUTED rather than produced per request. 871,620 rows is
# small, the booster is deterministic, and precomputing makes the output a
# DVC-tracked artifact that can be diffed and served straight from disk.
#
# The feature table is prepared by the SAME script the training data goes
# through (03_prepare_lightgbm_dataset.py, with the target made optional), so
# the WorldCover drops, the negative-AOD clip, the column set and its order are
# identical to what the model was fitted on. Season is cast with the explicit
# fixed category order for the same reason -- a plain astype("category") would
# derive codes from whichever rows are present and silently disagree with the
# booster.

ID_COLS = ["location_id", "name", "date"]
SEASON_COL = "season"
SEASON_CATEGORIES = ["summer", "monsoon", "post_monsoon", "winter"]

# The level the app serves. 80 and 95 are also calibrated and sit in the json.
SERVED_COVERAGE_LEVEL = "90"


def cast_season_categorical(df):
    df = df.copy()
    df[SEASON_COL] = pd.Categorical(df[SEASON_COL], categories=SEASON_CATEGORIES)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_features", required=True,
                        help="lightgbm_ready_grid.csv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--conformal", required=True,
                        help="conformal_calibration.json")
    parser.add_argument("--extrapolation", required=True,
                        help="grid_extrapolation_severity.csv")
    parser.add_argument("--output", required=True, help="parquet")
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    for path in [args.output, args.summary_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    booster = lgb.Booster(model_file=args.model)
    print(f"Loaded model: {args.model} ({booster.num_trees()} trees)")

    with open(args.conformal) as f:
        conformal = json.load(f)
    level = conformal["levels"][SERVED_COVERAGE_LEVEL]
    q = level["q"]
    print(f"Conformal {SERVED_COVERAGE_LEVEL}%: q={q:.4f} "
          f"(+/- {level['width_pct_of_prediction']}% of the prediction), "
          f"held-out coverage {100 * level['coverage_held_out_mean']:.1f}%")

    df = pd.read_csv(args.grid_features)
    print(f"Loaded grid features: {len(df)} rows, "
          f"{df['location_id'].nunique()} cells, {df['date'].nunique()} dates")

    df = cast_season_categorical(df)
    feature_cols = [c for c in df.columns if c not in ID_COLS]

    expected = booster.feature_name()
    if list(feature_cols) != list(expected):
        raise SystemExit(
            "ERROR: feature columns do not match the booster's.\n"
            f"  grid     : {feature_cols}\n"
            f"  booster  : {expected}\n"
            "Serving a differently-ordered or differently-named feature matrix "
            "would silently produce wrong predictions.")
    print(f"Feature matrix matches the booster: {len(feature_cols)} features")

    predictions = booster.predict(df[feature_cols])
    out = df[ID_COLS].copy()
    out["predicted_pm25_ugm3"] = np.round(predictions, 2)

    # Width scales with the prediction because the model's error is
    # multiplicative; the lower bound is floored because negative PM2.5 is
    # meaningless.
    out["pm25_lower_90_ugm3"] = np.round(np.maximum(0.0, predictions * (1.0 - q)), 2)
    out["pm25_upper_90_ugm3"] = np.round(predictions * (1.0 + q), 2)

    # Carry the extrapolation flag per cell. The interval is NOT widened for
    # these cells: whether unusualness predicts worse coverage was tested
    # against the 42 stations' measured held-out coverage and found
    # unsupported (Pearson r = -0.415 looks significant but is driven entirely
    # by one station at 102 range-widths out; Spearman r = -0.060 p = 0.71, the
    # sign reverses when that station is dropped, and a median split shows no
    # difference). Any widening factor would therefore be invented. The flag
    # states the limit instead -- see reports/prediction/.
    sev = pd.read_csv(args.extrapolation)[
        ["location_id", "worst_overshoot_frac", "worst_feature",
         "outside_training_range", "materially_outside"]]
    before = len(out)
    out = out.merge(sev, on="location_id", how="left")
    if len(out) != before:
        raise SystemExit("ERROR: extrapolation merge changed the row count")
    if out["outside_training_range"].isna().any():
        raise SystemExit("ERROR: some cells have no extrapolation flag")

    out.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows")

    n_cells = int(out["location_id"].nunique())
    summary = {
        "n_rows": len(out),
        "n_cells": n_cells,
        "n_dates": int(out["date"].nunique()),
        "conformal_level_pct": int(SERVED_COVERAGE_LEVEL),
        "conformal_q": q,
        "predicted_min": float(out["predicted_pm25_ugm3"].min()),
        "predicted_max": float(out["predicted_pm25_ugm3"].max()),
        "predicted_mean": float(out["predicted_pm25_ugm3"].mean()),
        "predicted_median": float(out["predicted_pm25_ugm3"].median()),
        "n_cells_outside_training_range": int(
            out.groupby("location_id")["outside_training_range"].first().sum()),
        "n_cells_materially_outside": int(
            out.groupby("location_id")["materially_outside"].first().sum()),
        "n_negative_predictions": int((out["predicted_pm25_ugm3"] < 0).sum()),
    }
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.summary_output}")

    print("=== prediction distribution (ug/m3) ===")
    print(out["predicted_pm25_ugm3"].describe().to_string())
    print("=== by season ===")
    seasonal = df[[SEASON_COL]].copy()
    seasonal["predicted"] = predictions
    print(seasonal.groupby(SEASON_COL, observed=True)["predicted"]
          .agg(["count", "mean", "median", "max"]).to_string())

    if summary["n_negative_predictions"] > 0:
        print(f"WARNING: {summary['n_negative_predictions']} negative predictions "
              f"-- PM2.5 cannot be negative; investigate before serving")


if __name__ == "__main__":
    main()
