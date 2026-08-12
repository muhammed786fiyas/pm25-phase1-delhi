"""
Merge two per-station PM2.5 download folders into one (e.g. an original 2025 pull
plus a later Jan-Mar 2026 extension), station by station, deduplicating any
overlapping timestamps and re-sorting chronologically.

Usage:
    python merge_downloads.py --dir1 data\\pm25_delhi_2025 --dir2 data\\pm25_delhi_2026_q1 --outdir data\\pm25_delhi_merged
"""
import argparse
import glob
import os
import re
import pandas as pd


def station_id_from_filename(path):
    m = re.search(r"pm25_loc_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir1", required=True, help="First folder (e.g. original 2025 download)")
    p.add_argument("--dir2", required=True, help="Second folder (e.g. Jan-Mar 2026 extension)")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files1 = {station_id_from_filename(f): f for f in glob.glob(os.path.join(args.dir1, "pm25_loc_*.csv"))}
    files2 = {station_id_from_filename(f): f for f in glob.glob(os.path.join(args.dir2, "pm25_loc_*.csv"))}

    all_stations = sorted(set(files1) | set(files2))
    print(f"dir1: {len(files1)} station files | dir2: {len(files2)} station files | union: {len(all_stations)}\n")

    summary = []
    for loc in all_stations:
        parts = []
        if loc in files1:
            parts.append(pd.read_csv(files1[loc]))
        if loc in files2:
            parts.append(pd.read_csv(files2[loc]))

        df = pd.concat(parts, ignore_index=True)
        before = len(df)
        df = df.drop_duplicates(subset=["datetime_from_utc"], keep="first")
        df = df.sort_values("datetime_from_utc")

        out_path = os.path.join(args.outdir, f"pm25_loc_{loc}.csv")
        df.to_csv(out_path, index=False)

        dmin = df["datetime_from_local"].min() if len(df) else None
        dmax = df["datetime_from_local"].max() if len(df) else None
        print(f"location {loc}: {before} -> {len(df)} rows after dedup | {dmin} to {dmax}")
        summary.append({"location_id": loc, "rows": len(df), "min_date": dmin, "max_date": dmax})

    combined = pd.concat((pd.read_csv(os.path.join(args.outdir, f"pm25_loc_{loc}.csv")) for loc in all_stations),
                          ignore_index=True)
    combined_path = os.path.join(args.outdir, "ALL_COMBINED.csv")
    combined.to_csv(combined_path, index=False)

    pd.DataFrame(summary).to_csv(os.path.join(args.outdir, "merge_summary.csv"), index=False)

    print(f"\nCombined: {len(combined)} rows from {len(all_stations)} stations -> {combined_path}")
    print(f"Per-station summary -> {os.path.join(args.outdir, 'merge_summary.csv')}")


if __name__ == "__main__":
    main()