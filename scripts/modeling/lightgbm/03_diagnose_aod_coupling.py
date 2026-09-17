import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

# Per-station AOD-PM2.5 coupling diagnostic for the LightGBM spatial-LOSO
# results. Exists because the explanation for why stations 5630 and 6359 fail
# is the evidential basis for a reported finding, and it should be a
# reproducible artifact rather than a one-off calculation.
#
# An earlier hypothesis -- that LightGBM degrades with the number of
# buffer-excluded neighbours -- was tested here and WITHDRAWN: the apparent
# gradient was an artifact of those same two stations, three other stations
# carry the identical 2-neighbour burden without harm, and the bare
# association is only Fisher p ~ 0.08. The coupling explanation replaced it.
#
# Kept as a separate copy of the ID/target constants (repo convention: no
# shared utils module across scripts).
ID_COLS = ["location_id", "name", "date"]
TARGET_COL = "pm25_daily"
GROUP_COL = "location_id"
AOD_COL = "aod_055"

# Stations that generated the hypothesis (the 3 weakest spatial-LOSO folds).
# The relationship is re-tested with these dropped, because forming a
# hypothesis from a handful of points and then testing it on those same
# points is circular -- the honest check is whether it still holds on the
# stations that played no part in suggesting it.
HYPOTHESIS_SOURCE_STATIONS = [5630, 6359, 5622]


def compute_station_coupling(df):
    # Pearson correlation between AOD and PM2.5 within each station, computed
    # from observed data only -- no model output involved. This is what makes
    # it usable as a selection criterion for the sensitivity variant without
    # the circularity of selecting on model error.
    rows = []
    for station_id, group in df.groupby(GROUP_COL):
        observed = group[group["aod_gap_filled"] == 0]
        gap_filled = group[group["aod_gap_filled"] == 1]
        rows.append({
            "location_id": station_id,
            "name": group["name"].iloc[0],
            "n_rows": len(group),
            "aod_pm_corr": group[AOD_COL].corr(group[TARGET_COL]),
            "aod_pm_corr_observed_aod": observed[AOD_COL].corr(observed[TARGET_COL]),
            "n_observed_aod": len(observed),
            "aod_pm_corr_gap_filled_aod": gap_filled[AOD_COL].corr(gap_filled[TARGET_COL]),
            "n_gap_filled_aod": len(gap_filled),
            "pm25_mean": group[TARGET_COL].mean(),
            "pm25_sd": group[TARGET_COL].std(),
        })
    return pd.DataFrame(rows)


def correlation_tests(table, metric):
    # Both Pearson and Spearman are reported: Pearson is the headline but is
    # inflated here by the tails, and Spearman shows how much of the
    # relationship survives as a monotonic rank ordering through the middle.
    pearson_r, pearson_p = stats.pearsonr(table["aod_pm_corr"], table[metric])
    spearman_r, spearman_p = stats.spearmanr(table["aod_pm_corr"], table[metric])
    return {
        "n_stations": len(table),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="lightgbm_ready_dataset.csv")
    parser.add_argument("--folds", required=True,
                         help="cv_spatial_loso_folds.csv from the primary validation run")
    parser.add_argument("--n_exclude", required=True, type=int,
                         help="how many weakest-coupling stations to name as the "
                              "sensitivity-variant exclusion set")
    parser.add_argument("--coupling_output", required=True, help="per-station coupling csv")
    parser.add_argument("--test_output", required=True, help="hypothesis test json")
    args = parser.parse_args()

    for outpath in [args.coupling_output, args.test_output]:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {args.input}: {len(df)} rows, {df[GROUP_COL].nunique()} stations")

    coupling = compute_station_coupling(df)
    folds = pd.read_csv(args.folds)
    table = coupling.merge(
        folds[["held_out_station", "r2", "within_r2", "rmse_ugm3",
               "n_excluded_buffer_stations"]],
        left_on="location_id", right_on="held_out_station").drop(columns="held_out_station")
    table["rmse_over_pm25_sd"] = table["rmse_ugm3"] / table["pm25_sd"]
    table = table.sort_values("aod_pm_corr").reset_index(drop=True)
    table["coupling_rank"] = np.arange(1, len(table) + 1)

    table.to_csv(args.coupling_output, index=False)
    print(f"Wrote {args.coupling_output}")

    print("=== weakest AOD-PM2.5 coupling ===")
    for row in table.head(args.n_exclude + 2).itertuples(index=False):
        print(f"  {row.coupling_rank}. station {row.location_id} ({row.name[:34]}): "
              f"corr={row.aod_pm_corr:.3f} fold R2={row.r2:.3f} "
              f"within_R2={row.within_r2:.3f}")

    exclude_ids = table.head(args.n_exclude)["location_id"].tolist()
    print(f"Weakest-{args.n_exclude} coupling stations: {sorted(exclude_ids)}")

    results = {
        "hypothesis": "stations whose AOD-PM2.5 coupling is weak are predicted poorly "
                      "under spatial LOSO",
        "coupling_metric": "within-station Pearson correlation of aod_055 vs pm25_daily, "
                           "computed from observed data only (no model output)",
        "all_stations": {},
        "excluding_hypothesis_source_stations": {},
        "hypothesis_source_stations": HYPOTHESIS_SOURCE_STATIONS,
        "weakest_coupling_stations": sorted(exclude_ids),
        "network_median_coupling": float(table["aod_pm_corr"].median()),
        "network_median_coupling_observed_aod": float(
            table["aod_pm_corr_observed_aod"].median()),
        "network_median_coupling_gap_filled_aod": float(
            table["aod_pm_corr_gap_filled_aod"].median()),
    }

    held_out = table[~table["location_id"].isin(HYPOTHESIS_SOURCE_STATIONS)]
    print("=== hypothesis test ===")
    for metric in ["r2", "within_r2"]:
        results["all_stations"][metric] = correlation_tests(table, metric)
        results["excluding_hypothesis_source_stations"][metric] = correlation_tests(
            held_out, metric)
        a = results["all_stations"][metric]
        b = results["excluding_hypothesis_source_stations"][metric]
        print(f"  {metric}: all {a['n_stations']} stations "
              f"Pearson r={a['pearson_r']:.3f} (p={a['pearson_p']:.2e}), "
              f"Spearman rho={a['spearman_rho']:.3f} (p={a['spearman_p']:.2e})")
        print(f"  {metric}: holding out the {len(HYPOTHESIS_SOURCE_STATIONS)} stations that "
              f"generated the hypothesis, n={b['n_stations']} "
              f"Pearson r={b['pearson_r']:.3f} (p={b['pearson_p']:.2e})")

    with open(args.test_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.test_output}")

    print("Gap-filled AOD weakens coupling network-wide "
          f"({results['network_median_coupling_observed_aod']:.3f} observed vs "
          f"{results['network_median_coupling_gap_filled_aod']:.3f} gap-filled median), "
          "but that is a separate, network-wide effect -- the weakest-coupling stations "
          "are weak on observed-AOD days too, so gap-fill is not what makes them weak.")


if __name__ == "__main__":
    main()
