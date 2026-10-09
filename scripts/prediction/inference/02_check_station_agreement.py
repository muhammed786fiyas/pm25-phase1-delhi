import argparse
import json
import math
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

# End-to-end validation of the grid pipeline.
#
# For each of the 42 stations, find the grid cell containing it and compare
# that cell's prediction against the model's prediction AT THE STATION, for the
# same date and the same booster.
#
# Why this is the check worth running: it is the only test that exercises the
# whole grid pipeline at once -- all 23 features, every join, the gap-fill, the
# NDVI period mapping and the season assignment -- against a reference that was
# built independently. If a feature were joined wrongly, scaled differently or
# silently shifted, the two predictions would diverge. Nothing else available
# tests the grid end to end, because grid cells have no ground truth.
#
# What a difference MEANS here matters. Both sides use the same booster, so a
# gap is not model disagreement; it is a feature difference between the station
# point and its cell centre. The two are up to ~700 m apart and each covariate
# is a 1 km-buffer statistic, so some difference is EXPECTED and correct --
# genuinely different ground. A near-zero median with a modest spread is the
# pass condition; a large systematic offset would indicate a pipeline fault.

EARTH_RADIUS_KM = 6371.0
ID_COLS = ["location_id", "name", "date"]
SEASON_COL = "season"
SEASON_CATEGORIES = ["summer", "monsoon", "post_monsoon", "winter"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (math.sin(delta_lat / 2) ** 2
         + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station_features", required=True)
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--grid_predictions", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    for path in [args.output, args.summary_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    booster = lgb.Booster(model_file=args.model)

    stations = pd.read_csv(args.station_file)
    stations = stations[stations["status"] == "KEEP"]
    grid = pd.read_csv(args.grid)

    # Nearest cell centre to each station. At 1 km spacing the containing cell
    # is at most ~700 m away.
    pairs = []
    for station in stations.itertuples(index=False):
        best_id = None
        best_km = None
        for cell in grid.itertuples(index=False):
            d = haversine_km(station.latitude, station.longitude,
                             cell.latitude, cell.longitude)
            if best_km is None or d < best_km:
                best_km = d
                best_id = cell.location_id
        pairs.append({"station_id": station.location_id,
                      "station_name": station.name,
                      "cell_id": int(best_id),
                      "station_to_cell_km": round(best_km, 4)})
    pairs = pd.DataFrame(pairs)
    print(f"Matched {len(pairs)} stations to cells "
          f"(max distance {pairs['station_to_cell_km'].max():.3f} km)")

    # Model prediction AT the station, using the same booster. This is an
    # in-sample prediction and is not a performance estimate -- it is the
    # reference the grid cell should reproduce.
    sdf = pd.read_csv(args.station_features)
    sdf[SEASON_COL] = pd.Categorical(sdf[SEASON_COL], categories=SEASON_CATEGORIES)
    feature_cols = booster.feature_name()
    station_pred = sdf[["location_id", "date"]].copy()
    station_pred["station_prediction"] = booster.predict(sdf[feature_cols])

    gp = pd.read_parquet(args.grid_predictions)[
        ["location_id", "date", "predicted_pm25_ugm3"]]
    gp = gp.rename(columns={"location_id": "cell_id",
                            "predicted_pm25_ugm3": "cell_prediction"})

    merged = station_pred.rename(columns={"location_id": "station_id"})
    merged = merged.merge(pairs, on="station_id", how="inner")
    merged = merged.merge(gp, on=["cell_id", "date"], how="inner")
    merged["difference"] = merged["cell_prediction"] - merged["station_prediction"]
    merged["abs_difference"] = merged["difference"].abs()
    merged["pct_difference"] = (100.0 * merged["difference"]
                                / merged["station_prediction"].clip(lower=1.0))
    print(f"Compared {len(merged)} station-days")

    merged.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    corr = float(np.corrcoef(merged["station_prediction"],
                             merged["cell_prediction"])[0, 1])
    summary = {
        "n_station_days": len(merged),
        "n_stations": int(merged["station_id"].nunique()),
        "max_station_to_cell_km": float(pairs["station_to_cell_km"].max()),
        "correlation": corr,
        "mean_difference_ugm3": float(merged["difference"].mean()),
        "median_difference_ugm3": float(merged["difference"].median()),
        "median_abs_difference_ugm3": float(merged["abs_difference"].median()),
        "p90_abs_difference_ugm3": float(np.percentile(merged["abs_difference"], 90)),
        "max_abs_difference_ugm3": float(merged["abs_difference"].max()),
        "median_abs_pct_difference": float(merged["pct_difference"].abs().median()),
        "station_mean_prediction": float(merged["station_prediction"].mean()),
        "cell_mean_prediction": float(merged["cell_prediction"].mean()),
    }

    print("=== grid cell vs station, same booster ===")
    print(f"correlation                 : {corr:.4f}")
    print(f"mean difference             : {summary['mean_difference_ugm3']:+.3f} ug/m3")
    print(f"median difference           : {summary['median_difference_ugm3']:+.3f} ug/m3")
    print(f"median |difference|         : {summary['median_abs_difference_ugm3']:.3f} ug/m3")
    print(f"median |difference| as pct  : {summary['median_abs_pct_difference']:.2f}%")
    print(f"p90 |difference|            : {summary['p90_abs_difference_ugm3']:.3f} ug/m3")
    print(f"max |difference|            : {summary['max_abs_difference_ugm3']:.3f} ug/m3")
    print(f"mean level, station vs cell : {summary['station_mean_prediction']:.2f} "
          f"vs {summary['cell_mean_prediction']:.2f} ug/m3")

    worst = (merged.groupby(["station_id", "station_name"])["abs_difference"]
             .median().sort_values(ascending=False).head(5))
    print("=== stations whose cell agrees worst (median |difference|) ===")
    print(worst.to_string())

    # A systematic offset would mean the grid is biased against the stations --
    # a pipeline fault. Scatter around zero is expected: the cell centre is up
    # to ~700 m from the station and every covariate is a 1 km-buffer statistic,
    # so the two describe genuinely different ground.
    checks = {
        "correlation_above_0.95": corr > 0.95,
        "median_bias_under_2_ugm3": abs(summary["median_difference_ugm3"]) < 2.0,
        "median_abs_diff_under_10_pct": summary["median_abs_pct_difference"] < 10.0,
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    print("=== verdict ===")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("GRID PIPELINE AGREES WITH THE STATIONS" if summary["passed"]
          else "DISAGREEMENT -- investigate before serving the grid")

    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
