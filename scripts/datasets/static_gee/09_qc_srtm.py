import argparse
import os
import sys
import yaml
import pandas as pd

PARAMS_FILE = "params.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    # Optional override of the roster named in params. The prediction grid is
    # written in the station file's exact schema (see scripts/grid/01_build_grid.py),
    # so pointing this at a grid CSV runs the same extractor over grid cells
    # instead of stations. Default None keeps the station behaviour unchanged.
    parser.add_argument("--stations", default=None,
                        help="station/grid roster CSV; defaults to params station_file")
    parser.add_argument("--elevation_min_m", type=float, default=None,
                        help="override the params bound; for the prediction grid")
    parser.add_argument("--elevation_max_m", type=float, default=None,
                        help="override the params bound; for the prediction grid")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["static_gee_layers"]["srtm"]["qc"]

    # The params bounds are calibrated for the 42 CPCB stations, which are all
    # urban and span 199.7-271.7 m. The prediction grid reaches terrain the
    # stations never sample -- its southernmost row sits on the Aravalli ridge
    # in south Delhi, where 301 m is genuine. Overriding the ceiling for that
    # run keeps the check meaningful (a 1000 m reading still fails) instead of
    # suppressing it or loosening it for the station stages too.
    elevation_min = params["elevation_min_m"]
    elevation_max = params["elevation_max_m"]
    if args.elevation_max_m is not None:
        elevation_max = args.elevation_max_m
    if args.elevation_min_m is not None:
        elevation_min = args.elevation_min_m
    slope_min = params["slope_min_deg"]
    slope_max = params["slope_max_deg"]

    print("=== Loading raw extraction ===")
    raw_df = pd.read_csv(args.input)
    print("Stations in raw file:", len(raw_df))

    print("=== Loading station list (source of truth) ===")
    station_file = args.stations if args.stations else params["station_file"]
    stations = pd.read_csv(station_file)
    stations = stations[stations["status"] == "KEEP"]
    expected_count = len(stations)
    print("Stations expected:", expected_count)

    qc_rows = []
    hard_fail_count = 0

    # --- Check 1: missing station entirely ---
    print("=== Check 1: missing stations ===")
    raw_ids = set(raw_df["location_id"])
    expected_ids = set(stations["location_id"])
    missing_ids = expected_ids - raw_ids

    if len(missing_ids) > 0:
        for missing_id in missing_ids:
            qc_rows.append({
                "location_id": missing_id,
                "check": "missing_station",
                "severity": "HARD_FAIL",
                "detail": "Station present in station file but absent from raw extraction"
            })
            hard_fail_count = hard_fail_count + 1
        print("MISSING STATIONS:", missing_ids)
    else:
        print("All expected stations present.")

    # --- Check 2 & 3: implausible elevation / slope ---
    print("=== Check 2 & 3: implausible elevation / slope ===")
    for index, row in raw_df.iterrows():
        location_id = row["location_id"]
        name = row["name"]
        elevation = row["elevation_m"]
        slope = row["slope_deg"]

        if pd.isna(elevation) or elevation < elevation_min or elevation > elevation_max:
            qc_rows.append({
                "location_id": location_id,
                "check": "implausible_elevation",
                "severity": "HARD_FAIL",
                "detail": name + ": elevation=" + str(elevation) + " outside [" + str(elevation_min) + ", " + str(elevation_max) + "]"
            })
            hard_fail_count = hard_fail_count + 1

        if pd.isna(slope) or slope < slope_min or slope > slope_max:
            qc_rows.append({
                "location_id": location_id,
                "check": "implausible_slope",
                "severity": "HARD_FAIL",
                "detail": name + ": slope=" + str(slope) + " outside [" + str(slope_min) + ", " + str(slope_max) + "]"
            })
            hard_fail_count = hard_fail_count + 1

    print("=== Saving QC summary ===")
    if len(qc_rows) == 0:
        qc_rows.append({
            "location_id": None,
            "check": "all_checks",
            "severity": "PASS",
            "detail": "No issues found across " + str(len(raw_df)) + " stations"
        })

    qc_df = pd.DataFrame(qc_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    qc_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("=== QC Result ===")
    print("Hard fails:", hard_fail_count)

    if hard_fail_count > 0:
        print("QC FAILED — stopping pipeline. Review", args.output, "before proceeding.")
        sys.exit(1)
    else:
        print("QC PASSED.")


if __name__ == "__main__":
    main()