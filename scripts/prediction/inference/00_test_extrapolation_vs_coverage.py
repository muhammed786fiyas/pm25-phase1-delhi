import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

# Does being UNUSUAL, relative to the rest of the training stations, predict
# WORSE conformal coverage?
#
# Why this matters. The served interval is prediction * (1 +/- q) with q
# calibrated on out-of-fold residuals at the 42 CPCB stations, all of which are
# urban. 585 of the 2388 prediction grid cells (25%) sit materially outside the
# range those stations span -- some beyond any value the model has seen. Serving
# them a flat "90% interval" asserts coverage that was never measured there.
#
# There is no ground truth at a grid cell, so the question cannot be tested
# directly. Stations are the only place it can be: each one has already been
# held out in turn, so each has a MEASURED coverage, and each can be scored for
# how unusual it is relative to the other 41 using exactly the metric applied to
# the grid cells. If unusual stations are covered worse, that is a measured
# relationship which can be carried (cautiously) to unusual cells. If they are
# not, then widening intervals for extrapolating cells would be inventing a
# correction, and the honest move is one interval plus a clear flag.
#
# This repeats the discipline that retired the distance-based confidence badge:
# distance to the nearest station was intuitive, was tested against actual
# per-fold error, scored r = 0.149 (p = 0.33), and was removed rather than kept
# because it sounded reasonable.

GROUP_COL = "location_id"
ID_COLS = ["location_id", "name", "date", "pm25_daily", "season"]


def overshoot_vs_others(values, others):
    # How far outside the OTHER stations' range this value sits, as a fraction
    # of that range's width. 0 means inside. 1.0 means a full range-width
    # beyond, where the model has no information at all. Scale-free, so it is
    # comparable across features with very different units, and it is the same
    # measure used for the prediction grid.
    low = others.min()
    high = others.max()
    width = high - low
    if width <= 0:
        return 0.0
    if values < low:
        return float((low - values) / width)
    if values > high:
        return float((values - high) / width)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="lightgbm_ready_dataset.csv")
    parser.add_argument("--coverage", required=True,
                        help="conformal_heldout_coverage.csv -- per-station "
                             "coverage from leave-one-station-out calibration")
    parser.add_argument("--output", required=True, help="per-station table")
    parser.add_argument("--summary_output", required=True, help="test result json")
    args = parser.parse_args()

    for path in [args.output, args.summary_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.read_csv(args.dataset)
    coverage = pd.read_csv(args.coverage)
    print(f"Loaded {len(df)} rows, {df[GROUP_COL].nunique()} stations")
    print(f"Loaded coverage for {len(coverage)} stations")

    features = [c for c in df.columns if c not in ID_COLS]
    features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    print(f"Scoring unusualness over {len(features)} numeric features")

    # One row per station: its feature values are constant for the static ones
    # and averaged for the time-varying ones. Averaging is the right summary
    # here because the question is whether the STATION as a whole sits outside
    # the others' envelope, not whether any single day did.
    per_station = df.groupby(GROUP_COL)[features].mean().reset_index()

    rows = []
    for i in range(len(per_station)):
        station_id = per_station.iloc[i][GROUP_COL]
        others = per_station[per_station[GROUP_COL] != station_id]
        worst = 0.0
        worst_feature = ""
        n_outside = 0
        for feature in features:
            score = overshoot_vs_others(per_station.iloc[i][feature], others[feature])
            if score > 0:
                n_outside = n_outside + 1
            if score > worst:
                worst = score
                worst_feature = feature
        rows.append({
            GROUP_COL: int(station_id),
            "worst_overshoot_frac": round(worst, 4),
            "worst_feature": worst_feature,
            "n_features_outside": n_outside,
        })

    unusual = pd.DataFrame(rows)
    merged = unusual.merge(coverage[[GROUP_COL, "coverage", "n_rows"]], on=GROUP_COL)
    merged = merged.sort_values("worst_overshoot_frac", ascending=False)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    print("=== most unusual stations ===")
    print(merged.head(8).to_string(index=False))
    print("=== least unusual stations ===")
    print(merged.tail(5).to_string(index=False))

    result = {
        "question": ("does being unusual relative to the other training stations "
                     "predict worse conformal coverage?"),
        "n_stations": len(merged),
        "coverage_min": float(merged["coverage"].min()),
        "coverage_max": float(merged["coverage"].max()),
        "coverage_mean": float(merged["coverage"].mean()),
    }

    print("=== correlation tests ===")
    for label, column in [("worst_overshoot_frac", "worst_overshoot_frac"),
                          ("n_features_outside", "n_features_outside")]:
        pearson_r, pearson_p = stats.pearsonr(merged[column], merged["coverage"])
        spearman_r, spearman_p = stats.spearmanr(merged[column], merged["coverage"])
        print(f"{label} vs coverage:")
        print(f"  Pearson  r = {pearson_r:+.3f}  p = {pearson_p:.4f}")
        print(f"  Spearman r = {spearman_r:+.3f}  p = {spearman_p:.4f}")
        result[label] = {
            "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        }

    # A correlation can be dragged around by one station, so also split at the
    # median and compare the halves directly -- the same guard that exposed the
    # buffer-exclusion gradient as a 2-station artifact earlier in this project.
    median_split = merged["worst_overshoot_frac"].median()
    more = merged[merged["worst_overshoot_frac"] > median_split]["coverage"]
    less = merged[merged["worst_overshoot_frac"] <= median_split]["coverage"]
    t_stat, t_p = stats.ttest_ind(more, less, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(more, less, alternative="two-sided")
    print("=== median split ===")
    print(f"more unusual (n={len(more)}): mean coverage {more.mean():.4f}")
    print(f"less unusual (n={len(less)}): mean coverage {less.mean():.4f}")
    print(f"difference {more.mean() - less.mean():+.4f}  "
          f"Welch p = {t_p:.4f}  Mann-Whitney p = {u_p:.4f}")
    result["median_split"] = {
        "threshold": float(median_split),
        "mean_coverage_more_unusual": float(more.mean()),
        "mean_coverage_less_unusual": float(less.mean()),
        "difference": float(more.mean() - less.mean()),
        "welch_p": float(t_p),
        "mannwhitney_p": float(u_p),
    }

    # Leave the worst station out and re-test. If the relationship survives only
    # with it, it is one point and not a pattern.
    trimmed = merged.iloc[1:]
    r_trim, p_trim = stats.pearsonr(trimmed["worst_overshoot_frac"], trimmed["coverage"])
    print("=== robustness: drop the single most unusual station ===")
    print(f"Pearson r = {r_trim:+.3f}  p = {p_trim:.4f}  (n = {len(trimmed)})")
    result["without_most_unusual_station"] = {
        "pearson_r": float(r_trim), "pearson_p": float(p_trim), "n": len(trimmed),
    }

    # A bare p-value is not enough to call this. The overshoot metric is wildly
    # skewed -- one station scores 102 against a next-highest of 2.45 -- so a
    # Pearson correlation can be produced single-handedly by that leverage
    # point. The verdict therefore requires the relationship to survive every
    # check, not just the one most likely to be fooled:
    #   - Pearson significant, AND
    #   - rank correlation agrees (immune to the leverage point), AND
    #   - it survives dropping the most extreme station, AND
    #   - the median split points the same way.
    # This project has already been bitten once by exactly this: an apparent
    # buffer-exclusion gradient turned out to be an artifact of 2 stations and
    # reversed when they were removed.
    checks = {
        "pearson_significant":
            result["worst_overshoot_frac"]["pearson_p"] < 0.05,
        "spearman_agrees":
            (result["worst_overshoot_frac"]["spearman_p"] < 0.05
             and result["worst_overshoot_frac"]["spearman_r"] < 0),
        "survives_dropping_most_extreme":
            (result["without_most_unusual_station"]["pearson_p"] < 0.05
             and result["without_most_unusual_station"]["pearson_r"] < 0),
        "median_split_agrees":
            (result["median_split"]["welch_p"] < 0.05
             and result["median_split"]["difference"] < 0),
    }
    result["checks"] = checks
    significant = all(checks.values())
    result["relationship_found"] = bool(significant)
    failed = [k for k, v in checks.items() if not v]
    result["conclusion"] = (
        "Unusualness predicts coverage on every check: an extrapolating grid "
        "cell can be given an expected coverage fitted to this relationship."
        if significant else
        "NO measured relationship. Checks failed: " + ", ".join(failed) + ". "
        "Any apparent correlation is driven by leverage from a single extreme "
        "station rather than a pattern across stations. Widening intervals for "
        "extrapolating cells would therefore be inventing a correction. Serve "
        "one interval, flag the cells, and report the measured per-station "
        "coverage spread as the honest bound.")
    print("=== robustness checks ===")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("=== conclusion ===")
    print(result["conclusion"])

    with open(args.summary_output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
