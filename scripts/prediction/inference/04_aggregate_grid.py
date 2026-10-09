import argparse
import json
import math
import os

import numpy as np
import pandas as pd

# Aggregates the 1 km grid to coarser resolutions for the download products.
#
# AGGREGATION IS OF PREDICTIONS, NEVER OF FEATURES. LightGBM is nonlinear, so
# mean(f(x)) != f(mean(x)) -- the same Jensen's-inequality problem that drove
# the LME's Duan smearing work. Averaging the features and predicting once
# would answer "PM2.5 at a hypothetical location with average conditions",
# which nobody is asking; averaging the predictions answers "average PM2.5 over
# this area", which is the question. It is also why the coarse products ship
# without features: a researcher re-running the model on supplied coarse
# features could not reproduce the supplied coarse prediction, and not shipping
# them makes that impossible to get wrong.
#
# THE INTERVAL SHRINKS, AND BY HOW MUCH IS MEASURED, NOT ASSUMED. Averaging N
# cells reduces the error of the average only as far as those errors are
# independent:
#     shrinkage = sqrt((1 + (N-1)*rho) / N)
# rho is the same-day correlation between two locations' prediction errors,
# measured in 03_measure_residual_correlation.py on spatial-LOSO out-of-fold
# residuals across all 861 station pairs: rho = 0.050, bootstrap 95% CI
# [0.022, 0.085].
#
# The point estimate is used for the served interval and the CI is carried
# through alongside, so a user can see how much the shrinkage itself is
# uncertain rather than being handed a single number as though it were exact.
#
# A FLOOR EXISTS. rho does not decay with distance (slope p = 0.62) -- it is a
# whole-day effect, days when the model errs the same way across all of Delhi
# at once. Such an error cannot be averaged away, so shrinkage tends to
# sqrt(rho) rather than zero as N grows. At rho = 0.050 that floor is 0.224,
# which 10 km (0.244) has nearly reached: aggregating beyond about 5 km buys
# very little precision and is worth offering for file size, not accuracy.

GROUP_COL = "location_id"


def shrinkage(n_cells, rho):
    inner = (1.0 + (n_cells - 1) * rho) / n_cells
    return math.sqrt(max(inner, 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_predictions", required=True)
    parser.add_argument("--grid_cells", required=True)
    parser.add_argument("--correlation", required=True,
                        help="residual_correlation.json")
    parser.add_argument("--conformal", required=True)
    parser.add_argument("--levels", default="2,5,10",
                        help="coarse resolutions in km, comma separated")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)

    with open(args.correlation) as f:
        corr = json.load(f)
    rho = corr["mean_rho_relative"]
    # Bootstrapped over stations, because pairs sharing a station are not
    # independent and a pair-level interval would be far too narrow.
    rho_low = corr.get("rho_ci_low")
    rho_high = corr.get("rho_ci_high")
    print(f"rho = {rho:.4f}" + (f", 95% CI [{rho_low:.4f}, {rho_high:.4f}]"
                                if rho_low is not None else ""))

    with open(args.conformal) as f:
        conformal = json.load(f)
    q = conformal["levels"]["90"]["q"]
    print(f"base 90% interval: prediction * (1 +/- {q:.4f})")

    preds = pd.read_parquet(args.grid_predictions)
    cells = pd.read_csv(args.grid_cells)
    cells = cells[cells["status"] == "KEEP"][
        [GROUP_COL, "latitude", "longitude", "grid_row", "grid_col",
         "dist_to_nearest_station_km"]]
    df = preds.merge(cells, on=GROUP_COL, how="inner")
    print(f"Loaded {len(df)} cell-days over {df[GROUP_COL].nunique()} cells")

    summary_levels = []
    for km_text in args.levels.split(","):
        km = int(km_text.strip())
        # Coarse cells are exact blocks of the 1 km lattice, so each 1 km cell
        # belongs to exactly one coarse cell and nothing is double counted.
        df["coarse_row"] = df["grid_row"] // km
        df["coarse_col"] = df["grid_col"] // km

        grouped = df.groupby(["coarse_row", "coarse_col", "date"], sort=False)
        agg = grouped.agg(
            pm25_ugm3=("predicted_pm25_ugm3", "mean"),
            n_cells_averaged=("predicted_pm25_ugm3", "size"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            dist_to_nearest_station_km=("dist_to_nearest_station_km", "mean"),
            n_cells_outside_training_range=("materially_outside", "sum"),
        ).reset_index()

        # Shrinkage depends on how many cells were actually averaged, which is
        # not always km*km: coarse cells at the edge of the coverage area are
        # partial, and giving them the full-block shrinkage would overstate
        # their precision.
        n = agg["n_cells_averaged"].to_numpy()
        factor = np.array([shrinkage(int(x), rho) for x in n])
        half_width = q * factor
        agg["pm25_lower_90_ugm3"] = np.maximum(
            0.0, agg["pm25_ugm3"] * (1.0 - half_width))
        agg["pm25_upper_90_ugm3"] = agg["pm25_ugm3"] * (1.0 + half_width)
        agg["interval_shrinkage"] = np.round(factor, 4)

        # The CI on rho carried through, so the uncertainty in the shrinkage
        # itself is visible rather than hidden behind a point estimate.
        if rho_low is not None:
            agg["interval_shrinkage_rho_low"] = np.round(
                [shrinkage(int(x), rho_low) for x in n], 4)
            agg["interval_shrinkage_rho_high"] = np.round(
                [shrinkage(int(x), rho_high) for x in n], 4)

        agg["fraction_outside_training_range"] = np.round(
            agg["n_cells_outside_training_range"] / agg["n_cells_averaged"], 3)
        for col in ["pm25_ugm3", "pm25_lower_90_ugm3", "pm25_upper_90_ugm3",
                    "latitude", "longitude", "dist_to_nearest_station_km"]:
            agg[col] = agg[col].round(4 if col in ("latitude", "longitude") else 2)

        out_path = os.path.join(args.outdir, f"grid_predictions_{km}km.parquet")
        agg.to_parquet(out_path, index=False)
        n_coarse = agg.groupby(["coarse_row", "coarse_col"]).ngroups
        typical = shrinkage(km * km, rho)
        print(f"{km} km: {n_coarse} cells, {len(agg)} cell-days -> {out_path}")
        print(f"      full block = {km * km} cells, shrinkage {typical:.3f} "
              f"(interval +/- {100 * q * typical:.1f}% instead of "
              f"{100 * q:.1f}%)")
        summary_levels.append({
            "grid_km": km,
            "n_coarse_cells": int(n_coarse),
            "n_cell_days": len(agg),
            "cells_per_full_block": km * km,
            "shrinkage_full_block": round(typical, 4),
            "interval_pct_full_block": round(100 * q * typical, 2),
            "shrinkage_rho_low": (round(shrinkage(km * km, rho_low), 4)
                                  if rho_low is not None else None),
            "shrinkage_rho_high": (round(shrinkage(km * km, rho_high), 4)
                                   if rho_high is not None else None),
            "min_cells_averaged": int(agg["n_cells_averaged"].min()),
            "mean_pm25": float(agg["pm25_ugm3"].mean()),
        })

    summary = {
        "rho": rho,
        "rho_ci_low": rho_low,
        "rho_ci_high": rho_high,
        "base_q_90": q,
        "shrinkage_floor": round(math.sqrt(max(rho, 0.0)), 4),
        "levels": summary_levels,
        "what_the_interval_means": (
            "The shrunk interval applies to the AREAL MEAN over the coarse cell, "
            "NOT to any particular point inside it. A 5 km value with a tighter "
            "interval is a more precise estimate of the average across 25 square "
            "kilometres; it says nothing more precise about any one street."),
        "aggregation_rule": (
            "predictions are averaged, never features -- LightGBM is nonlinear so "
            "mean(f(x)) != f(mean(x)), and coarse products therefore ship without "
            "features so a supplied prediction can never fail to reproduce"),
        "floor_note": (
            "rho does not decay with distance, so it is a whole-day effect rather "
            "than a local one, and cannot be averaged away. Shrinkage tends to "
            "sqrt(rho) as N grows, so aggregating beyond about 5 km buys very "
            "little precision."),
    }
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.summary_output}")
    print(f"=== shrinkage floor: {summary['shrinkage_floor']:.3f} "
          f"(no amount of averaging goes below this) ===")


if __name__ == "__main__":
    main()
