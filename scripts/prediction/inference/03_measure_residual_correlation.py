import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

# Measures rho: the same-day correlation between two locations' prediction
# errors, as a function of how far apart they are.
#
# Why it decides the aggregated interval. Averaging N cells into one coarse
# value reduces the error of that average -- but only as far as the errors are
# independent. The standard error of a mean of N equally-variable, equally
# correlated errors is
#     sigma * sqrt((1 + (N-1)*rho) / N)
# so rho = 0 gives the full 1/sqrt(N) benefit and rho = 1 gives none at all.
# Underestimating rho therefore makes coarse intervals look TIGHTER than they
# are, which is the dangerous direction: it would understate uncertainty in a
# published product.
#
# An earlier estimate of 0.032 came from only 16 stations with 8 pairs under
# 2 km -- thin exactly where 1 km -> 2 km aggregation depends on it. This uses
# all 42 stations' spatial-LOSO out-of-fold residuals, which is the right error
# population: each station's residual comes from a model that never saw it, the
# same situation as an unmonitored grid cell.
#
# Correlation is measured on RELATIVE residuals, (observed - predicted) /
# predicted, because the served interval is multiplicative: prediction *
# (1 +/- q). The error of an areal mean, when the cell predictions are of
# similar magnitude, is approximately the mean of the relative errors, so the
# relative residual is the quantity whose correlation governs the shrinkage.
# Absolute residuals are reported alongside as a cross-check.

EARTH_RADIUS_KM = 6371.0
GROUP_COL = "location_id"
PREDICTION_FLOOR_UGM3 = 5.0

# Minimum shared days before a pair's correlation is trusted. A pair with a
# handful of overlapping dates produces a correlation that is mostly noise.
MIN_SHARED_DAYS = 60


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def shrinkage(n_cells, rho):
    # Standard error of a mean of n equally correlated errors, relative to a
    # single one. Clipped at 0 because a negative rho large enough to drive the
    # bracket below zero is a small-sample artifact, not a real variance.
    inner = (1.0 + (n_cells - 1) * rho) / n_cells
    return float(math.sqrt(max(inner, 0.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True,
                        help="cv_spatial_loso_oof_predictions.csv")
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--pairs_output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    for path in [args.pairs_output, args.summary_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    oof = pd.read_csv(args.oof)
    stations = pd.read_csv(args.station_file)
    stations = stations[stations["status"] == "KEEP"][
        [GROUP_COL, "name", "latitude", "longitude"]]
    print(f"Loaded {len(oof)} out-of-fold rows, {oof[GROUP_COL].nunique()} stations")

    oof["relative_residual"] = ((oof["observed_pm25"] - oof["predicted_pm25"])
                                / oof["predicted_pm25"].clip(lower=PREDICTION_FLOOR_UGM3))
    oof["absolute_residual"] = oof["observed_pm25"] - oof["predicted_pm25"]

    rel = oof.pivot(index="date", columns=GROUP_COL, values="relative_residual")
    absolute = oof.pivot(index="date", columns=GROUP_COL, values="absolute_residual")
    coords = stations.set_index(GROUP_COL)

    ids = [i for i in rel.columns if i in coords.index]
    rows = []
    for a_index in range(len(ids)):
        for b_index in range(a_index + 1, len(ids)):
            a = ids[a_index]
            b = ids[b_index]
            pair = rel[[a, b]].dropna()
            if len(pair) < MIN_SHARED_DAYS:
                continue
            r_rel, p_rel = stats.pearsonr(pair[a], pair[b])
            pair_abs = absolute[[a, b]].dropna()
            r_abs, _ = stats.pearsonr(pair_abs[a], pair_abs[b])
            distance = haversine_km(coords.loc[a, "latitude"], coords.loc[a, "longitude"],
                                    coords.loc[b, "latitude"], coords.loc[b, "longitude"])
            rows.append({
                "station_a": int(a), "station_b": int(b),
                "distance_km": round(distance, 3),
                "n_shared_days": len(pair),
                "rho_relative": round(float(r_rel), 4),
                "rho_absolute": round(float(r_abs), 4),
                "p_relative": float(p_rel),
            })

    pairs = pd.DataFrame(rows).sort_values("distance_km")
    pairs.to_csv(args.pairs_output, index=False)
    print(f"Wrote {args.pairs_output}: {len(pairs)} station pairs "
          f"(>= {MIN_SHARED_DAYS} shared days)")

    print("=== rho by separation distance ===")
    edges = [0, 2, 5, 10, 20, 50]
    band_rows = []
    print("%-14s %7s %10s %10s %10s" % ("distance", "pairs", "mean rho", "median", "sd"))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        band = pairs[(pairs["distance_km"] >= lo) & (pairs["distance_km"] < hi)]
        if len(band) == 0:
            continue
        print("%-14s %7d %10.4f %10.4f %10.4f" % (
            f"{lo}-{hi} km", len(band), band["rho_relative"].mean(),
            band["rho_relative"].median(), band["rho_relative"].std()))
        band_rows.append({
            "band_km": f"{lo}-{hi}", "n_pairs": len(band),
            "mean_rho_relative": float(band["rho_relative"].mean()),
            "median_rho_relative": float(band["rho_relative"].median()),
            "sd_rho_relative": float(band["rho_relative"].std()),
            "mean_rho_absolute": float(band["rho_absolute"].mean()),
        })

    overall = float(pairs["rho_relative"].mean())
    # The pairs that matter most for 1 km -> 2 km aggregation are the closest
    # ones, and they are also the scarcest. Report them explicitly rather than
    # letting a global average stand in for them.
    close = pairs[pairs["distance_km"] < 2.0]
    print(f"=== overall ===")
    print(f"all pairs           : n={len(pairs)}  mean rho={overall:.4f}")
    if len(close) > 0:
        print(f"pairs under 2 km    : n={len(close)}  "
              f"mean rho={close['rho_relative'].mean():.4f}  "
              f"range {close['rho_relative'].min():.3f} to "
              f"{close['rho_relative'].max():.3f}")
    else:
        print("pairs under 2 km    : none")

    # Does rho actually decay with distance, or is it flat? If flat, a single
    # value is defensible for every aggregation level; if it decays, close cells
    # share more error than distant ones and each level needs its own.
    slope, intercept, r_value, p_value, stderr = stats.linregress(
        pairs["distance_km"], pairs["rho_relative"])
    print("=== does rho decay with distance? ===")
    print(f"slope {slope:+.6f} per km, p = {p_value:.4f}, r = {r_value:+.3f}")
    decays = p_value < 0.05 and slope < 0

    # Aggregation levels, as cells-per-coarse-cell at 1 km base resolution.
    print("=== interval shrinkage when aggregating ===")
    print("%-10s %6s %14s %14s %14s" % (
        "grid", "cells", "independent", f"rho={overall:.3f}", "conservative"))
    # A conservative value guards against rho being underestimated by a thin
    # close-pair sample: use the mean of the closest band, which is the highest
    # correlation the data supports at aggregation distances.
    conservative = (float(close["rho_relative"].mean()) if len(close) > 0
                    else float(pairs.nsmallest(10, "distance_km")["rho_relative"].mean()))
    levels = []
    for km, n_cells in [(1, 1), (2, 4), (5, 25), (10, 100)]:
        s_ind = shrinkage(n_cells, 0.0)
        s_obs = shrinkage(n_cells, overall)
        s_con = shrinkage(n_cells, conservative)
        print("%-10s %6d %14.3f %14.3f %14.3f" % (
            f"{km} km", n_cells, s_ind, s_obs, s_con))
        levels.append({"grid_km": km, "n_cells": n_cells,
                       "shrinkage_if_independent": round(s_ind, 4),
                       "shrinkage_at_measured_rho": round(s_obs, 4),
                       "shrinkage_at_conservative_rho": round(s_con, 4)})

    # Bootstrap the CI over STATIONS, not pairs. The 861 pairs are not
    # independent -- each station appears in 41 of them -- so resampling pairs
    # would give a CI several times too narrow (about [0.039, 0.061] instead of
    # the honest [0.022, 0.085]). Resampling stations respects the dependence.
    rng = np.random.default_rng(42)
    station_ids = sorted(set(pairs["station_a"]) | set(pairs["station_b"]))
    boot = []
    for _ in range(4000):
        drawn = set(rng.choice(station_ids, size=len(station_ids), replace=True))
        sub = pairs[pairs["station_a"].isin(drawn) & pairs["station_b"].isin(drawn)]
        if len(sub) > 20:
            boot.append(sub["rho_relative"].mean())
    rho_low, rho_high = np.percentile(boot, [2.5, 97.5])
    print("=== bootstrap CI (resampled over stations) ===")
    print(f"rho = {overall:.4f}, 95% CI [{rho_low:.4f}, {rho_high:.4f}] "
          f"from {len(boot)} resamples")
    if len(close) > 0 and not (rho_low <= close["rho_relative"].mean() <= rho_high):
        print(f"note: the close-pair mean {close['rho_relative'].mean():.3f} falls "
              f"OUTSIDE this interval, so it is a small-sample fluctuation rather "
              f"than a better-targeted estimate -- which follows from rho not "
              f"decaying with distance: close pairs measure the same quantity as "
              f"all the others, just with n=8.")

    summary = {
        "n_pairs": len(pairs),
        "rho_ci_low": float(rho_low),
        "rho_ci_high": float(rho_high),
        "rho_ci_method": ("bootstrap over stations, 4000 resamples -- pairs sharing "
                          "a station are dependent, so a pair-level CI is far too "
                          "narrow"),
        "n_stations": int(oof[GROUP_COL].nunique()),
        "min_shared_days": MIN_SHARED_DAYS,
        "mean_rho_relative": overall,
        "median_rho_relative": float(pairs["rho_relative"].median()),
        "mean_rho_absolute": float(pairs["rho_absolute"].mean()),
        "n_pairs_under_2km": len(close),
        "mean_rho_under_2km": (float(close["rho_relative"].mean())
                               if len(close) > 0 else None),
        "conservative_rho": conservative,
        "decay_slope_per_km": float(slope),
        "decay_p_value": float(p_value),
        "rho_decays_with_distance": bool(decays),
        "bands": band_rows,
        "aggregation_levels": levels,
        "note": ("rho is measured on spatial-LOSO out-of-fold relative residuals -- "
                 "errors at stations the model never trained on, the same situation "
                 "as an unmonitored grid cell. The served interval is multiplicative, "
                 "so the relative residual is the quantity whose correlation governs "
                 "how much an areal mean's interval may shrink."),
    }
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.summary_output}")

    if len(close) < 10:
        print(f"WARNING: only {len(close)} pairs under 2 km. The 1 km -> 2 km "
              f"shrinkage rests on a thin sample; prefer the conservative rho.")


if __name__ == "__main__":
    main()
