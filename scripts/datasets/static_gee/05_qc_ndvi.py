import argparse
import os
import sys
import yaml
import pandas as pd

PARAMS_FILE = "params.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["static_gee_layers"]["ndvi"]["qc"]

    tolerance_pct = params["null_rate_warn_pct"]
    ndvi_min_bound = params["ndvi_plausible_min"]
    ndvi_max_bound = params["ndvi_plausible_max"]

    print("=== Loading raw NDVI periods ===")
    raw_df = pd.read_csv(args.input)
    print("Total rows:", len(raw_df))

    print("=== Loading station list (source of truth) ===")
    stations = pd.read_csv(params["station_file"])
    stations = stations[stations["status"] == "KEEP"]
    expected_count = len(stations)
    print("Stations expected:", expected_count)

    qc_rows = []
    hard_fail_count = 0
    warning_count = 0

    # --- Check 1: missing stations entirely (never appear in any period) ---
    print("=== Check 1: missing stations ===")
    raw_ids = set(raw_df["location_id"])
    expected_ids = set(stations["location_id"])
    missing_ids = expected_ids - raw_ids

    if len(missing_ids) > 0:
        for missing_id in missing_ids:
            qc_rows.append({
                "scope": "station",
                "key": missing_id,
                "check": "missing_station",
                "severity": "HARD_FAIL",
                "detail": "Station never appears in any period's extraction"
            })
            hard_fail_count = hard_fail_count + 1
    else:
        print("All expected stations present in at least one period.")

    # --- Check 2: row count sanity (should be periods x stations, minus zero-image periods) ---
    total_periods = raw_df["period_index"].nunique()
    expected_rows = total_periods * expected_count
    print("Periods found:", total_periods, "| Expected rows:", expected_rows, "| Actual rows:", len(raw_df))

    if len(raw_df) != expected_rows:
        qc_rows.append({
            "scope": "overall",
            "key": None,
            "check": "row_count_mismatch",
            "severity": "WARNING",
            "detail": "Expected " + str(expected_rows) + " rows (periods x stations), got " + str(len(raw_df))
        })
        warning_count = warning_count + 1

    # --- Check 3: overall null rate ---
    print("=== Check 3: overall null rate ===")
    total_rows = len(raw_df)
    null_rows = raw_df["ndvi_mean"].isna().sum()
    overall_null_pct = (null_rows / total_rows) * 100
    print("Overall null rate:", round(overall_null_pct, 2), "%")

    qc_rows.append({
        "scope": "overall",
        "key": None,
        "check": "overall_null_rate",
        "severity": "INFO",
        "detail": str(round(overall_null_pct, 2)) + "% of rows have null NDVI (" + str(null_rows) + " of " + str(total_rows) + ")"
    })

    # --- Check 4: null rate by month ---
    print("=== Check 4: null rate by month ===")
    raw_df["period_start_date"] = pd.to_datetime(raw_df["period_start"])
    raw_df["month"] = raw_df["period_start_date"].dt.to_period("M").astype(str)

    monthly_summary = raw_df.groupby("month")["ndvi_mean"].apply(
        lambda x: (x.isna().sum() / len(x)) * 100
    )

    for month, null_pct in monthly_summary.items():
        severity = "INFO"
        if null_pct > tolerance_pct:
            severity = "WARNING"
            warning_count = warning_count + 1
        qc_rows.append({
            "scope": "month",
            "key": month,
            "check": "null_rate_by_month",
            "severity": severity,
            "detail": str(round(null_pct, 2)) + "% null this month"
        })
        print("Month", month, ":", round(null_pct, 2), "% null")

    # --- Check 5: null rate by station ---
    print("=== Check 5: null rate by station ===")
    station_summary = raw_df.groupby(["location_id", "name"])["ndvi_mean"].apply(
        lambda x: (x.isna().sum() / len(x)) * 100
    )

    for (location_id, name), null_pct in station_summary.items():
        severity = "INFO"
        if null_pct > tolerance_pct:
            severity = "WARNING"
            warning_count = warning_count + 1
        qc_rows.append({
            "scope": "station",
            "key": location_id,
            "check": "null_rate_by_station",
            "severity": severity,
            "detail": name + ": " + str(round(null_pct, 2)) + "% null"
        })

    # --- Check 6: implausible NDVI values (outside physical range) ---
    print("=== Check 6: implausible NDVI values ===")
    implausible = raw_df[
        (raw_df["ndvi_mean"].notna()) &
        ((raw_df["ndvi_mean"] < ndvi_min_bound) | (raw_df["ndvi_mean"] > ndvi_max_bound))
    ]

    if len(implausible) > 0:
        for index, row in implausible.iterrows():
            qc_rows.append({
                "scope": "station_period",
                "key": str(row["location_id"]) + "_" + str(row["period_index"]),
                "check": "implausible_ndvi",
                "severity": "HARD_FAIL",
                "detail": "NDVI=" + str(row["ndvi_mean"]) + " outside plausible range [" + str(ndvi_min_bound) + ", " + str(ndvi_max_bound) + "]"
            })
            hard_fail_count = hard_fail_count + 1
    else:
        print("No implausible NDVI values found.")

    print("=== Saving QC summary ===")
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
        print("QC PASSED (warnings are informational, do not block).")


if __name__ == "__main__":
    main()