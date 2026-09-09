import argparse

import numpy as np
import pandas as pd

# aod_x_monsoon/aod_x_post_monsoon/aod_x_winter are exactly
# aod_055 * season_<name> (verified against lme_ready_dataset.csv -- max abs
# diff 0.0), so after capping aod_055 these three MUST be recomputed from the
# capped value, not left as-is, or the capped-away extremes would still leak
# back in through the interaction columns.
AOD_COL = "aod_055"
SEASON_INTERACTION_COLS = {
    "aod_x_monsoon": "season_monsoon",
    "aod_x_post_monsoon": "season_post_monsoon",
    "aod_x_winter": "season_winter",
}


def winsorize_aod(df, zscore_cap):
    # aod_055 is already standardized in lme_ready_dataset.csv (mean ~0,
    # std ~1), so a z-score cap directly bounds it at [-zscore_cap, zscore_cap]
    # -- no need to recompute mean/std here.
    original = df[AOD_COL].values
    capped = np.clip(original, -zscore_cap, zscore_cap)
    n_capped = int((capped != original).sum())
    return capped, n_capped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="lme_ready_dataset.csv")
    parser.add_argument("--output", required=True, help="winsorized dataset csv")
    parser.add_argument("--zscore_cap", required=True, type=float,
                         help="cap aod_055 at +/- this many standard deviations")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {args.input}: {len(df)} rows")
    print(f"aod_055 before: mean={df[AOD_COL].mean():.4f} std={df[AOD_COL].std():.4f} "
          f"min={df[AOD_COL].min():.4f} max={df[AOD_COL].max():.4f}")

    capped, n_capped = winsorize_aod(df, args.zscore_cap)
    df[AOD_COL] = capped
    print(f"Capped aod_055 at +/-{args.zscore_cap}: {n_capped} of {len(df)} rows "
          f"({100 * n_capped / len(df):.2f}%) changed")
    print(f"aod_055 after: mean={df[AOD_COL].mean():.4f} std={df[AOD_COL].std():.4f} "
          f"min={df[AOD_COL].min():.4f} max={df[AOD_COL].max():.4f}")

    for interaction_col, season_col in SEASON_INTERACTION_COLS.items():
        df[interaction_col] = df[AOD_COL] * df[season_col]
    print("Recomputed aod_x_monsoon/aod_x_post_monsoon/aod_x_winter from the "
          "capped aod_055 (they are aod_055 * season dummy, so leaving them "
          "as-is would let the capped-away extremes leak back in)")

    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
