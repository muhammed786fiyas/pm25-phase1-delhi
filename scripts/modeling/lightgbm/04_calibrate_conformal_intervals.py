import argparse
import json
import os

import numpy as np
import pandas as pd

# Calibrates prediction intervals for the LightGBM model from its spatial-LOSO
# out-of-fold residuals (02_validate_lightgbm_cv.py writes them per row).
#
# Method: NORMALIZED conformal prediction, scaled by the prediction.
#   interval = prediction * (1 +/- q)
# where q is a quantile of the relative absolute residual |y - yhat| / yhat
# measured on held-out stations. One calibrated ratio produces a different
# width for every cell.
#
# Why normalized rather than a fixed +/- width: error here is multiplicative.
# The 90% width divided by the prediction is near-constant across prediction
# quintiles while the absolute width spans roughly +/-21 to +/-115 ug/m3.
#
# Why conformal rather than a Gaussian +/- 1.645*RMSE: the residuals are far
# from normal (heavy-tailed; median absolute error around 13 against a maximum
# past 400). A Gaussian interval is much too wide in the middle of the
# distribution and too narrow in the tail -- it overstates ordinary uncertainty
# and understates the extremes, which for air quality is the worse failure.
# Conformal just counts percentiles, so the shape does not matter.
#
# Why it is not circular: q is calibrated on OUT-OF-FOLD residuals -- errors at
# stations the model never saw. The prediction acts as an index into a table of
# measured past performance, not as the model assessing its own confidence.
# Conditioning on yhat = f(x) is conditioning on a summary of the features,
# which is ordinary conditional coverage; conditioning on the observed value
# would be the cheat.
#
# Kept as a separate copy of the column names (repo convention: no shared utils
# module across scripts).
GROUP_COL = "location_id"
DATE_COL = "date"
OBSERVED_COL = "observed_pm25"
PREDICTED_COL = "predicted_pm25"

COVERAGE_LEVELS = [0.80, 0.90, 0.95]

# Predictions near zero would make the relative residual explode, so the
# denominator is floored. Delhi's observed PM2.5 almost never goes this low --
# the floor only guards against a degenerate division.
PREDICTION_FLOOR_UGM3 = 5.0


def relative_residual(df):
    denom = df[PREDICTED_COL].clip(lower=PREDICTION_FLOOR_UGM3)
    return (df[OBSERVED_COL] - df[PREDICTED_COL]).abs() / denom


def absolute_residual(df):
    return (df[OBSERVED_COL] - df[PREDICTED_COL]).abs()


def coverage_of(df, q):
    # Fraction of rows whose observed value falls inside prediction*(1 +/- q).
    lower = (df[PREDICTED_COL] * (1.0 - q)).clip(lower=0.0)
    upper = df[PREDICTED_COL] * (1.0 + q)
    inside = (df[OBSERVED_COL] >= lower) & (df[OBSERVED_COL] <= upper)
    return float(inside.mean())


def leave_one_station_out_coverage(df, level):
    # The honest test of a calibration: for each station, calibrate q on the
    # OTHER 41 stations and measure coverage on the held-out one. Calibrating
    # and evaluating on the same rows would report the target level by
    # construction and tell us nothing.
    rows = []
    for station in sorted(df[GROUP_COL].unique()):
        held = df[df[GROUP_COL] == station]
        rest = df[df[GROUP_COL] != station]
        q = float(np.quantile(relative_residual(rest), level))
        rows.append({
            GROUP_COL: station,
            "n_rows": len(held),
            "q_from_other_stations": q,
            "coverage": coverage_of(held, q),
        })
    return pd.DataFrame(rows)


def bias_by_prediction_band(df, n_bands=5):
    # Conformal guarantees the interval's WIDTH, not that it is centred. If the
    # model systematically under-predicted high values, a symmetric interval
    # would sit on a biased point and miss more often on one side. This reports
    # signed error by prediction band so that assumption is checked rather than
    # assumed.
    out = df.copy()
    out["band"] = pd.qcut(out[PREDICTED_COL], n_bands, duplicates="drop")
    rows = []
    for band, g in out.groupby("band", observed=True):
        signed = g[OBSERVED_COL] - g[PREDICTED_COL]
        rows.append({
            "prediction_band": f"{band.left:.0f}-{band.right:.0f}",
            "n_rows": len(g),
            "mean_signed_error": float(signed.mean()),
            "median_signed_error": float(signed.median()),
            "bias_pct_of_level": float(100.0 * signed.mean() / g[PREDICTED_COL].mean()),
            "abs_resid_p90": float(np.quantile(g[OBSERVED_COL].sub(g[PREDICTED_COL]).abs(), 0.90)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_input", required=True,
                         help="cv_spatial_loso_oof_predictions.csv -- spatial LOSO "
                              "residuals, i.e. errors at stations the model never saw, "
                              "which is the situation at an unmonitored grid cell")
    parser.add_argument("--calibration_output", required=True, help="conformal json")
    parser.add_argument("--diagnostics_output", required=True,
                         help="per-station held-out coverage csv")
    args = parser.parse_args()

    for outpath in [args.calibration_output, args.diagnostics_output]:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    df = pd.read_csv(args.oof_input)
    print(f"Loaded {args.oof_input}: {len(df)} out-of-fold rows, "
          f"{df[GROUP_COL].nunique()} stations")

    rel = relative_residual(df)
    absr = absolute_residual(df)
    print("=== residual distribution (out-of-fold, ug/m3) ===")
    print(f"mean |error|: {absr.mean():.2f}")
    for pct in [50, 80, 90, 95, 99]:
        print(f"p{pct} |error|: {np.percentile(absr, pct):.2f}")
    print(f"max |error|: {absr.max():.2f}")
    # Normality is judged on the SIGNED residual: |residual| is right-skewed by
    # construction even for perfectly normal errors, so its skew says nothing.
    signed = df[OBSERVED_COL] - df[PREDICTED_COL]
    print(f"signed residual skew: {signed.skew():.2f}  kurtosis: {signed.kurtosis():.2f} "
          f"(0 and 0 would be normal -- high kurtosis is why a Gaussian "
          f"+/- 1.645*RMSE interval misfits these tails)")

    calibration = {
        "method": "normalized_conformal_scaled_by_prediction",
        "formula": "interval = prediction * (1 +/- q)",
        "calibrated_on": args.oof_input,
        "calibration_scheme": ("spatial leave-one-station-out out-of-fold residuals -- "
                               "errors at stations the model never trained on"),
        "n_calibration_rows": len(df),
        "n_calibration_stations": int(df[GROUP_COL].nunique()),
        "signed_residual_skew": float((df[OBSERVED_COL] - df[PREDICTED_COL]).skew()),
        "signed_residual_kurtosis": float((df[OBSERVED_COL] - df[PREDICTED_COL]).kurtosis()),
        "prediction_floor_ugm3": PREDICTION_FLOOR_UGM3,
        "levels": {},
    }

    print("=== calibrated ratios ===")
    for level in COVERAGE_LEVELS:
        q = float(np.quantile(rel, level))
        in_sample = coverage_of(df, q)
        loso = leave_one_station_out_coverage(df, level)
        calibration["levels"][f"{int(level * 100)}"] = {
            "q": q,
            "width_pct_of_prediction": round(100.0 * q, 1),
            "coverage_in_sample": in_sample,
            "coverage_held_out_mean": float(loso["coverage"].mean()),
            "coverage_held_out_min": float(loso["coverage"].min()),
            "coverage_held_out_max": float(loso["coverage"].max()),
            "n_stations_below_target": int((loso["coverage"] < level).sum()),
            # Conformal guarantees MARGINAL coverage (right on average across all
            # rows) and not CONDITIONAL coverage (right at every station). The
            # per-station spread below is the honest measure of that gap.
            "coverage_held_out_std": float(loso["coverage"].std()),
            "worst_covered_station": int(loso.loc[loso["coverage"].idxmin(), GROUP_COL]),
            "absolute_width_equivalent_ugm3": float(np.quantile(absr, level)),
        }
        print(f"{int(level * 100)}% coverage: q={q:.4f} "
              f"(+/- {100 * q:.1f}% of the prediction), "
              f"held-out coverage mean {100 * loso['coverage'].mean():.1f}% "
              f"(min {100 * loso['coverage'].min():.1f}%, "
              f"max {100 * loso['coverage'].max():.1f}%)")
        if level == 0.90:
            loso.to_csv(args.diagnostics_output, index=False)
            print(f"Wrote {args.diagnostics_output}")

    bias = bias_by_prediction_band(df)
    calibration["centring_check"] = {
        "note": ("conformal fixes the interval WIDTH but not its centre. A symmetric "
                 "interval is only honest if the model is roughly unbiased across "
                 "prediction levels -- checked here rather than assumed."),
        "bands": bias.to_dict(orient="records"),
        "max_abs_bias_pct_of_level": float(bias["bias_pct_of_level"].abs().max()),
    }
    print("=== centring check: signed error by prediction band ===")
    for row in bias.itertuples(index=False):
        print(f"{row.prediction_band}: n={row.n_rows} "
              f"mean signed error {row.mean_signed_error:+.2f} "
              f"({row.bias_pct_of_level:+.0f}% of level), "
              f"p90 |error| {row.abs_resid_p90:.1f}")

    q90 = calibration["levels"]["90"]["q"]
    lower = (df[PREDICTED_COL] * (1.0 - q90)).clip(lower=0.0)
    upper = df[PREDICTED_COL] * (1.0 + q90)
    calibration["centring_check"]["missed_below_pct_at_90"] = float(
        100.0 * (df[OBSERVED_COL] < lower).mean())
    calibration["centring_check"]["missed_above_pct_at_90"] = float(
        100.0 * (df[OBSERVED_COL] > upper).mean())
    print(f"At 90%: missed below {calibration['centring_check']['missed_below_pct_at_90']:.1f}%, "
          f"missed above {calibration['centring_check']['missed_above_pct_at_90']:.1f}% "
          f"(balanced would be 5% and 5%)")

    with open(args.calibration_output, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Wrote {args.calibration_output}")


if __name__ == "__main__":
    main()
