import argparse
import os
import yaml
import pandas as pd

with open("params.yaml") as f:
    params = yaml.safe_load(f)

HIGH_VALUE_LIMIT = params["cpcb_qc"]["high_value_limit"]
STUCK_RUN_LIMIT = params["cpcb_qc"]["stuck_run_limit"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Merged/trimmed PM2.5 CSV")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    df["dt_local"] = pd.to_datetime(df["datetime_from_local"])

    print(f"Loaded {len(df)} rows")

    # 1. duplicate check
    dup_count = df.duplicated(subset=["location_id", "dt_local"]).sum()
    print(f"\n=== Duplicates ===")
    print(f"Duplicate rows: {dup_count}")
    if dup_count > 0:
        dups = df[df.duplicated(subset=["location_id", "dt_local"], keep=False)]
        dup_path = os.path.join(args.outdir, "duplicates.csv")
        dups.to_csv(dup_path, index=False)
        print(f"Wrote duplicate rows to {dup_path}")

    # 2. range check
    print(f"\n=== Range check ===")
    negatives = df[df["value"] < 0]
    print(f"Negative values: {len(negatives)}")
    if len(negatives) > 0:
        neg_path = os.path.join(args.outdir, "negative_values_hourly.csv")
        negatives.to_csv(neg_path, index=False)
        print(f"Wrote negative rows to {neg_path}")

    too_high = df[df["value"] > HIGH_VALUE_LIMIT]
    print(f"Values above {HIGH_VALUE_LIMIT}: {len(too_high)}")
    if len(too_high) > 0:
        high_path = os.path.join(args.outdir, "high_values_hourly.csv")
        too_high.to_csv(high_path, index=False)
        print(f"Wrote high value rows to {high_path}")

    # 3. stuck-sensor check (per station, run-length of repeated values)
    print(f"\n=== Stuck sensor check ===")
    stuck_rows = []
    station_ids = df["location_id"].unique()

    for station_id in station_ids:
        sub = df[df["location_id"] == station_id].sort_values("dt_local").copy()

        sub["is_repeat"] = sub["value"] == sub["value"].shift(1)
        sub["run_id"] = (sub["is_repeat"] == False).cumsum()

        run_lengths = sub.groupby("run_id").size()
        stuck_runs = run_lengths[run_lengths >= STUCK_RUN_LIMIT]

        if len(stuck_runs) > 0:
            total_stuck_hours = stuck_runs.sum()
            print(f"station {station_id}: {len(stuck_runs)} stuck runs, {total_stuck_hours} hours total")
            stuck_rows.append({
                "location_id": station_id,
                "stuck_run_count": len(stuck_runs),
                "stuck_hours_total": total_stuck_hours,
                "longest_run": stuck_runs.max(),
            })

    if len(stuck_rows) > 0:
        stuck_df = pd.DataFrame(stuck_rows)
        stuck_path = os.path.join(args.outdir, "stuck_sensor_summary_hourly.csv")
        stuck_df.to_csv(stuck_path, index=False)
        print(f"Wrote stuck sensor summary to {stuck_path}")
    else:
        print(f"No runs of {STUCK_RUN_LIMIT}+ repeated hours found")

    # 4. zero value check
    print(f"\n=== Zero value check ===")
    zero_counts = df[df["value"] == 0].groupby("location_id").size()
    total_counts = df.groupby("location_id").size()
    zero_pct = (zero_counts / total_counts * 100).round(1)

    zero_summary = pd.DataFrame({
        "zero_count": zero_counts,
        "total_count": total_counts,
        "zero_pct": zero_pct,
    }).fillna(0)
    zero_summary = zero_summary.sort_values("zero_pct", ascending=False)

    zero_path = os.path.join(args.outdir, "zero_value_summary_hourly.csv")
    zero_summary.to_csv(zero_path)
    print(f"Wrote zero value summary to {zero_path}")
    print(zero_summary.head(10))

if __name__ == "__main__":
    main()