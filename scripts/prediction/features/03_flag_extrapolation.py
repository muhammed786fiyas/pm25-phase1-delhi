import argparse
import os

import numpy as np
import pandas as pd

# Flags grid cells whose features fall outside the range the model was TRAINED
# on, and records how far outside.
#
# Why this is needed. LightGBM cannot extrapolate: a tree fitted on cropland up
# to 78% contains no split above 78%, so a 97%-cropland cell silently receives
# the boundary prediction. The conformal interval has the same problem from the
# other side -- its 90% coverage was measured on out-of-fold residuals at 42
# CPCB stations, all urban, so it is not evidence about a farmland cell.
# Neither the estimate nor its interval is validated there, and the product
# should say so rather than present a confident number.
#
# Severity is recorded, not just a yes/no, because a binary flag badly
# overstates the problem: a cell 2% beyond the training range is a rounding
# detail, one a full range-width beyond is a different thing entirely.
# Measured as a fraction of the training range's WIDTH, which makes it
# comparable across features with very different units.
#
# The intervals are deliberately NOT widened for flagged cells. Whether
# unusualness predicts worse coverage was tested against the 42 stations'
# measured held-out coverage (00_test_extrapolation_vs_coverage.py) and found
# unsupported: the Pearson correlation looks significant but is produced
# entirely by one station sitting 102 range-widths out, the rank correlation is
# null, the sign reverses when that station is dropped, and a median split
# shows no difference. Any widening factor would therefore be invented, and an
# invented correction that looks rigorous is worse than an honest flag.

ID_COLS = ["location_id", "name", "date", "pm25_daily", "season"]

# Below this, "outside the range" is a rounding detail rather than a real
# extrapolation -- the cell sits a few percent past an edge the model has
# effectively seen.
MATERIAL_OVERSHOOT_FRAC = 0.25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True,
                        help="lightgbm_ready_dataset.csv -- defines the range "
                             "the model actually saw")
    parser.add_argument("--grid", required=True, help="grid master_feature_table.csv")
    parser.add_argument("--severity_output", required=True)
    parser.add_argument("--by_feature_output", required=True)
    args = parser.parse_args()

    for path in [args.severity_output, args.by_feature_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    train = pd.read_csv(args.training)
    grid = pd.read_csv(args.grid, low_memory=False)
    print(f"Training: {len(train)} rows, {train['location_id'].nunique()} stations")
    print(f"Grid: {len(grid)} rows, {grid['location_id'].nunique()} cells")

    features = [c for c in train.columns
                if c not in ID_COLS and pd.api.types.is_numeric_dtype(train[c])]
    print(f"Comparing {len(features)} numeric features")

    # One row per cell. Static features are constant; time-varying ones are
    # averaged, because the question is whether the CELL sits outside the
    # envelope, not whether any single day did.
    cell = grid.groupby("location_id").first().reset_index()

    per_feature = []
    worst = np.zeros(len(cell))
    worst_feature = np.array([""] * len(cell), dtype=object)
    n_outside = np.zeros(len(cell), dtype=int)

    for feature in features:
        if feature not in cell.columns:
            continue
        low = train[feature].min()
        high = train[feature].max()
        width = high - low
        if width <= 0:
            continue
        below = (low - cell[feature]).clip(lower=0) / width
        above = (cell[feature] - high).clip(lower=0) / width
        overshoot = np.maximum(below, above).values
        n_outside = n_outside + (overshoot > 0).astype(int)
        worst_feature = np.where(overshoot > worst, feature, worst_feature)
        worst = np.maximum(worst, overshoot)
        if (overshoot > 0).sum() > 0:
            per_feature.append({
                "feature": feature,
                "train_min": low, "train_max": high,
                "grid_min": float(cell[feature].min()),
                "grid_max": float(cell[feature].max()),
                "n_cells_outside": int((overshoot > 0).sum()),
                "max_overshoot_frac": round(float(overshoot.max()), 4),
            })

    by_feature = pd.DataFrame(per_feature).sort_values(
        "n_cells_outside", ascending=False)
    by_feature.to_csv(args.by_feature_output, index=False)
    print(f"Wrote {args.by_feature_output}")
    print(by_feature.to_string(index=False))

    severity = pd.DataFrame({
        "location_id": cell["location_id"],
        "worst_overshoot_frac": np.round(worst, 4),
        "worst_feature": worst_feature,
        "n_features_outside": n_outside,
    })
    severity["outside_training_range"] = severity["worst_overshoot_frac"] > 0
    severity["materially_outside"] = (
        severity["worst_overshoot_frac"] > MATERIAL_OVERSHOOT_FRAC)
    severity.to_csv(args.severity_output, index=False)
    print(f"Wrote {args.severity_output}")

    bands = pd.cut(severity["worst_overshoot_frac"],
                   [-0.001, 0, 0.05, 0.25, 0.5, 1.0, 1e9],
                   labels=["inside range", "<5% beyond", "5-25%",
                           "25-50%", "50-100%", ">100%"])
    print("=== how far beyond the training range each cell sits ===")
    print(bands.value_counts().reindex(
        ["inside range", "<5% beyond", "5-25%", "25-50%", "50-100%", ">100%"]
    ).to_string())
    n_any = int(severity["outside_training_range"].sum())
    n_material = int(severity["materially_outside"].sum())
    print(f"outside on >=1 feature : {n_any} of {len(severity)} "
          f"({100 * n_any / len(severity):.0f}%)")
    print(f"materially outside     : {n_material} of {len(severity)} "
          f"({100 * n_material / len(severity):.0f}%)")
    print("=== which feature drives the worst overshoot ===")
    print(severity[severity["outside_training_range"]]["worst_feature"]
          .value_counts().to_string())


if __name__ == "__main__":
    main()
