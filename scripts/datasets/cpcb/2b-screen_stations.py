"""
Decide which stations are worth downloading, using only the station metadata
from 2-list_stations.py -- no measurement data needed yet.

A station is dropped here if:
  - it has no records at all (no first/last timestamp)
  - its last record is before the study window starts
  - its first record is after the study window ends
  - it sits at the same coordinates as another eligible station
    (same physical site, reported twice under different agencies)

Everything else is marked ELIGIBLE. 3-download_pm25.py downloads only the
ELIGIBLE stations, and 10b-finalize_roster.py later turns ELIGIBLE into
KEEP or a completeness drop once the data is in.

The output deliberately leaves out datetime_last. That column moves forward
every day while a sensor keeps reporting, so writing it here would change
this file on every run and force every downstream stage to re-run.
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd

with open("params.yaml") as f:
    params = yaml.safe_load(f)

STUDY_START = params["cpcb_screen_stations"]["study_start"]
STUDY_END = params["cpcb_screen_stations"]["study_end"]
DUPLICATE_TOLERANCE_KM = params["cpcb_screen_stations"]["duplicate_tolerance_km"]

EARTH_RADIUS_KM = 6371.0
OUTPUT_COLUMNS = ["location_id", "name", "latitude", "longitude", "status"]


def distance_km(lat1, lon1, lat2, lon2):
    # haversine distance between two points
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def mark_duplicate_sites(df):
    # Only compare stations that are still eligible: dead predecessor ids can
    # sit a few hundred metres from their live replacements, and they have
    # already been dropped above, so they must not count as duplicates.
    # Within a duplicate pair, keep the station with the longest record
    # (earliest first_date); location_id breaks an exact tie.
    eligible = df[df["status"] == "ELIGIBLE"].sort_values(["first_date", "location_id"])

    kept = []
    for row in eligible.itertuples(index=False):
        duplicate_of = None
        for other in kept:
            if distance_km(row.latitude, row.longitude, other.latitude, other.longitude) < DUPLICATE_TOLERANCE_KM:
                duplicate_of = other.location_id
                break

        if duplicate_of is None:
            kept.append(row)
        else:
            df.loc[df["location_id"] == row.location_id, "status"] = f"DROP_duplicate_site_of_{duplicate_of}"

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw station list from 2-list_stations.py")
    parser.add_argument("--output", required=True, help="Screened station list")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} stations from {args.input}")

    # names occasionally come back from OpenAQ with stray spaces
    df["name"] = df["name"].str.strip()

    # the date part of the ISO timestamps is enough to compare against the window
    df["first_date"] = df["datetime_first"].str[:10]
    df["last_date"] = df["datetime_last"].str[:10]

    df["status"] = "ELIGIBLE"

    no_data = df["first_date"].isna() | df["last_date"].isna()
    df.loc[no_data, "status"] = "DROP_no_data"

    ended_early = (df["status"] == "ELIGIBLE") & (df["last_date"] < STUDY_START)
    df.loc[ended_early, "status"] = "DROP_ended_before_window"

    started_late = (df["status"] == "ELIGIBLE") & (df["first_date"] > STUDY_END)
    df.loc[started_late, "status"] = "DROP_started_after_window"

    df = mark_duplicate_sites(df)

    # fixed row order, so the file only changes when a decision changes
    df = df.sort_values("location_id")
    df[OUTPUT_COLUMNS].to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    print(f"=== Screening result (window {STUDY_START} .. {STUDY_END}) ===")
    print(df["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
