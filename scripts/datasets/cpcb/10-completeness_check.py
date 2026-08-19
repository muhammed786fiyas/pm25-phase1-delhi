import argparse
import os
import pandas as pd

WINDOW_START = "2025-03-01"
WINDOW_END = "2026-02-28"
COMPLETENESS_THRESHOLD = 60

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Daily aggregated PM2.5 CSV")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} station-days, {df['location_id'].nunique()} stations")

    possible_days = (pd.Timestamp(WINDOW_END) - pd.Timestamp(WINDOW_START)).days + 1
    print(f"Possible days in window: {possible_days}")

    actual_days = df.groupby("location_id").size()
    actual_days = actual_days.reset_index(name="actual_days")

    actual_days["possible_days"] = possible_days
    actual_days["completeness_pct"] = (actual_days["actual_days"] / possible_days * 100).round(1)
    actual_days["passes_60pct"] = actual_days["completeness_pct"] >= COMPLETENESS_THRESHOLD

    actual_days = actual_days.sort_values("completeness_pct")

    out_path = os.path.join(args.outdir, "station_completeness.csv")
    actual_days.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    print(f"\n=== Stations below {COMPLETENESS_THRESHOLD}% ===")
    failed = actual_days[actual_days["passes_60pct"] == False]
    if len(failed) == 0:
        print("None")
    else:
        print(failed)

    print(f"\nAll stations, sorted by completeness:")
    print(actual_days)

if __name__ == "__main__":
    main()