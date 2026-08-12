"""
Download hourly PM2.5 from OpenAQ (CPCB stations) for a date range.

Why this exists instead of main.py:
  main.py's get_all_measurements() never passes datetime_from/datetime_to to the API.
  It pages through a sensor's ENTIRE history from the start and filters client-side,
  capped at max_pages=100 x limit=100 = 10,000 records. For a station with data since
  2016 (~92,000 hourly records), pages 1-100 only reach ~early 2017, so a 2025 request
  returns nothing. This script filters server-side instead.

Also fixed here:
  - Only the pm25 sensor is queried, not all 18 sensors at a station
  - End date is inclusive
  - One CSV per station, written as it completes -> a crash loses one station, not everything
  - Re-running skips stations already downloaded (resume)

Usage:
    python download_pm25.py --stations cpcb_stations_delhi_2025_FINAL.csv \
        --from-date 2025-01-01 --to-date 2025-12-31 --outdir pm25_delhi_2025

    # or an explicit ID list instead of a CSV:
    python download_pm25.py --location-id 17 50 235 \
        --from-date 2025-01-01 --to-date 2025-12-31 --outdir pm25_delhi_2025

Get your own free API key at https://openaq.org (Account -> API keys) and either
pass --api-key or set the OPENAQ_API_KEY environment variable.
"""

import argparse
import os
import sys
import time
import glob
import requests
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://api.openaq.org/v3"
PAGE_LIMIT = 1000          # API max per page; 8760 hourly records -> ~9 pages/station
MAX_RETRIES = 4
SLEEP_BETWEEN = 0.5        # polite delay between requests


def make_session(api_key):
    s = requests.Session()
    s.headers.update({"X-API-Key": api_key})
    return s


def get_with_retry(session, url, params=None):
    """GET with backoff on 429 / 5xx. Returns response or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            wait = 5 * attempt
            print(f"      network error ({e.__class__.__name__}); retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue

        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 10 * attempt
            print(f"      HTTP {r.status_code}; retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 401:
            sys.exit("HTTP 401 - API key rejected. Check --api-key / OPENAQ_API_KEY.")
        print(f"      HTTP {r.status_code} - giving up on this request")
        return None
    return None


def get_pm25_sensors(session, location_id, win_from, win_to):
    """
    Return every pm25 sensor at this location whose coverage overlaps the requested
    window, as a list of (sensor_id, first_local, last_local).

    A station can have MULTIPLE pm25 sensor objects over its lifetime: when a monitor
    is replaced or re-registered, OpenAQ creates a new sensor id and the old one is
    frozen with its historical data. Picking only the first pm25 sensor returned can
    land on a sensor retired years ago, or miss part of the window that is held by a
    different sensor object. Both happen in the CPCB Delhi network.
    """
    r = get_with_retry(session, f"{BASE_URL}/locations/{location_id}/sensors")
    time.sleep(SLEEP_BETWEEN)
    if r is None:
        return []

    wf = pd.Timestamp(win_from)
    wt = pd.Timestamp(win_to)
    out = []
    for s in r.json().get("results", []):
        if (s.get("parameter") or {}).get("name", "").lower() != "pm25":
            continue
        first = ((s.get("datetimeFirst") or {}).get("local") or "")[:10]
        last = ((s.get("datetimeLast") or {}).get("local") or "")[:10]
        if not first or not last:
            continue
        f, l = pd.Timestamp(first), pd.Timestamp(last)
        if l >= wf and f <= wt:          # overlaps the requested window
            out.append((s.get("id"), first, last))
    return sorted(out, key=lambda t: t[1])


def fetch_hourly(session, sensor_id, date_from, date_to):
    """
    Fetch hourly records for one sensor between date_from and date_to (inclusive).
    Uses the /hours resource, which OpenAQ documents as preferred over
    /measurements/hourly. Date filtering happens server-side.
    """
    rows = []
    page = 1
    while True:
        params = {
            "datetime_from": date_from,
            "datetime_to": date_to,
            "limit": PAGE_LIMIT,
            "page": page,
        }
        r = get_with_retry(session, f"{BASE_URL}/sensors/{sensor_id}/hours", params=params)
        time.sleep(SLEEP_BETWEEN)
        if r is None:
            print(f"      aborting sensor {sensor_id} at page {page}")
            break

        results = r.json().get("results", [])
        if not results:
            break

        for e in results:
            period = e.get("period") or {}
            param = e.get("parameter") or {}
            summ = e.get("summary") or {}
            cov = e.get("coverage") or {}
            rows.append({
                "sensor_id": sensor_id,
                "parameter_name": param.get("name"),
                "units": param.get("units"),
                "value": e.get("value"),
                "datetime_from_utc": (period.get("datetimeFrom") or {}).get("utc"),
                "datetime_from_local": (period.get("datetimeFrom") or {}).get("local"),
                "datetime_to_local": (period.get("datetimeTo") or {}).get("local"),
                "summary_min": summ.get("min"),
                "summary_max": summ.get("max"),
                "summary_sd": summ.get("sd"),
                "coverage_expected": cov.get("expectedCount"),
                "coverage_observed": cov.get("observedCount"),
                "coverage_percent": cov.get("percentComplete"),
            })

        if len(results) < PAGE_LIMIT:
            break
        page += 1

    return rows


def load_location_ids(args):
    if args.location_id:
        return [int(x) for x in args.location_id]

    df = pd.read_csv(args.stations)
    if "location_id" not in df.columns:
        sys.exit(f"{args.stations} has no 'location_id' column.")
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "KEEP"]
        print(f"Station list: kept {len(df)} of {before} rows (status == KEEP)")
    return df["location_id"].astype(int).tolist()


def main():
    p = argparse.ArgumentParser(description="Download hourly PM2.5 from OpenAQ CPCB stations")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stations", help="CSV with a location_id column (and optional status column)")
    src.add_argument("--location-id", nargs="+", help="Explicit location IDs, space separated")
    p.add_argument("--from-date", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--to-date", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--outdir", required=True, help="Directory for per-station CSVs")
    p.add_argument("--api-key", default=os.environ.get("OPENAQ_API_KEY"),
                   help="OpenAQ API key (or set OPENAQ_API_KEY)")
    p.add_argument("--combined", default=None,
                   help="Optional path for the merged CSV (default: <outdir>/ALL_COMBINED.csv)")
    args = p.parse_args()

    if not args.api_key:
        sys.exit("No API key. Pass --api-key or set OPENAQ_API_KEY.\n"
                 "Get a free key at https://openaq.org (Account -> API keys).")

    # make end date inclusive: OpenAQ treats datetime_to as an upper bound on the
    # hour start, so pushing to the next midnight captures the final day in full
    date_to_api = (pd.Timestamp(args.to_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    os.makedirs(args.outdir, exist_ok=True)
    session = make_session(args.api_key)
    location_ids = load_location_ids(args)

    print(f"\n{len(location_ids)} stations | {args.from_date} to {args.to_date} | -> {args.outdir}\n")

    done, empty, failed = [], [], []

    for n, loc in enumerate(location_ids, 1):
        out_csv = os.path.join(args.outdir, f"pm25_loc_{loc}.csv")
        if os.path.exists(out_csv):
            print(f"[{n}/{len(location_ids)}] location {loc}: already downloaded, skipping")
            done.append(loc)
            continue

        print(f"[{n}/{len(location_ids)}] location {loc}: finding pm25 sensors...", end=" ", flush=True)
        sensors = get_pm25_sensors(session, loc, args.from_date, args.to_date)
        if not sensors:
            print("no pm25 sensor covering this window")
            failed.append(loc)
            continue
        print(f"{len(sensors)} overlapping: " + ", ".join(f"{sid}({f}..{l})" for sid, f, l in sensors))

        rows = []
        for sid, _f, _l in sensors:
            got = fetch_hourly(session, sid, args.from_date, date_to_api)
            print(f"        sensor {sid}: {len(got)} raw rows")
            rows.extend(got)

        if not rows:
            print(f"      no records returned in range")
            empty.append(loc)
            continue

        df = pd.DataFrame(rows)
        df.insert(0, "location_id", loc)

        # hard client-side guard: keep only rows inside the requested local-date window
        d = df["datetime_from_local"].astype(str).str[:10]
        before = len(df)
        df = df[(d >= args.from_date) & (d <= args.to_date)]
        if len(df) != before:
            print(f"        dropped {before - len(df)} rows outside {args.from_date}..{args.to_date}")

        # a replaced sensor can double-report a boundary hour
        before = len(df)
        df = df.drop_duplicates(subset=["datetime_from_utc"], keep="first")
        if len(df) != before:
            print(f"        dropped {before - len(df)} duplicate timestamps")

        if df.empty:
            print(f"      nothing left after filtering")
            empty.append(loc)
            continue

        df = df.sort_values("datetime_from_utc")
        df.to_csv(out_csv, index=False)
        print(f"      {len(df):>5} rows -> {os.path.basename(out_csv)}")
        done.append(loc)

    # merge
    parts = sorted(glob.glob(os.path.join(args.outdir, "pm25_loc_*.csv")))
    if parts:
        combined = args.combined or os.path.join(args.outdir, "ALL_COMBINED.csv")
        big = pd.concat((pd.read_csv(f) for f in parts), ignore_index=True)
        big.to_csv(combined, index=False)
        print(f"\nCombined: {len(big)} rows from {len(parts)} stations -> {combined}")

    print(f"\nDone: {len(done)}   Empty: {len(empty)}   No pm25 sensor / failed: {len(failed)}")
    if empty:
        print(f"  empty  : {empty}")
    if failed:
        print(f"  failed : {failed}")


if __name__ == "__main__":
    main()