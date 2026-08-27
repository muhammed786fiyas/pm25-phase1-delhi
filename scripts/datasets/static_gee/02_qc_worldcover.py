import argparse
import os
import sys
import yaml
import pandas as pd

PARAMS_FILE = "params.yaml"

WORLDCOVER_CLASSES = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "95", "100"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["static_gee_layers"]["worldcover"]["qc"]

    tolerance_pct = params["expected_pixels_tolerance_pct"]
    dominant_warn_pct = params["dominant_class_warn_pct"]
    implausible_classes = params["implausible_classes"]

    print("=== Loading raw extraction ===")
    raw_df = pd.read_csv(args.input)
    print("Stations in raw file:", len(raw_df))

    print("=== Loading station list (source of truth) ===")
    stations = pd.read_csv(params["station_file"])
    stations = stations[stations["status"] == "KEEP"]
    expected_count = len(stations)
    print("Stations expected:", expected_count)

    qc_rows = []
    hard_fail_count = 0
    warning_count = 0

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

    median_pixels = raw_df["total_pixels"].median()
    print("Median total_pixels across all stations:", median_pixels)

    for index, row in raw_df.iterrows():
        location_id = row["location_id"]
        total_pixels = row["total_pixels"]

        source_row = stations[stations["location_id"] == location_id]
        if len(source_row) > 0:
            source_lat = source_row.iloc[0]["latitude"]
            source_lon = source_row.iloc[0]["longitude"]
            if row["latitude"] != source_lat or row["longitude"] != source_lon:
                qc_rows.append({
                    "location_id": location_id,
                    "check": "coordinate_mismatch",
                    "severity": "HARD_FAIL",
                    "detail": "Extracted lat/lon does not match station file"
                })
                hard_fail_count = hard_fail_count + 1

        if total_pixels == 0:
            qc_rows.append({
                "location_id": location_id,
                "check": "zero_pixels",
                "severity": "HARD_FAIL",
                "detail": "Station returned zero pixels"
            })
            hard_fail_count = hard_fail_count + 1
        else:
            deviation_pct = abs(total_pixels - median_pixels) / median_pixels * 100
            if deviation_pct > tolerance_pct:
                qc_rows.append({
                    "location_id": location_id,
                    "check": "expected_pixels",
                    "severity": "HARD_FAIL",
                    "detail": "total_pixels deviates " + str(round(deviation_pct, 2)) + "% from median (tolerance " + str(tolerance_pct) + "%)"
                })
                hard_fail_count = hard_fail_count + 1

        class_max_count = 0
        class_max_code = None
        for class_code in WORLDCOVER_CLASSES:
            column_name = "class_" + class_code + "_count"
            count = row[column_name]
            if count > class_max_count:
                class_max_count = count
                class_max_code = class_code

        if total_pixels > 0:
            dominant_pct = class_max_count / total_pixels * 100
            if dominant_pct >= dominant_warn_pct:
                qc_rows.append({
                    "location_id": location_id,
                    "check": "all_one_class",
                    "severity": "WARNING",
                    "detail": "Class " + str(class_max_code) + " makes up " + str(round(dominant_pct, 2)) + "% of pixels"
                })
                warning_count = warning_count + 1

        for implausible_code in implausible_classes:
            column_name = "class_" + str(implausible_code) + "_count"
            if column_name in row and row[column_name] > 0:
                qc_rows.append({
                    "location_id": location_id,
                    "check": "implausible_class",
                    "severity": "WARNING",
                    "detail": "Class " + str(implausible_code) + " present (" + str(row[column_name]) + " pixels) — geographically unlikely for Delhi"
                })
                warning_count = warning_count + 1

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
    print("Warnings:", warning_count)

    if hard_fail_count > 0:
        print("QC FAILED — stopping pipeline. Review", args.output, "before proceeding.")
        sys.exit(1)
    else:
        print("QC PASSED.")


if __name__ == "__main__":
    main()