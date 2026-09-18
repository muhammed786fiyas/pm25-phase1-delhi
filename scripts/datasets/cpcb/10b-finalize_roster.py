"""
Build the final station roster that every later stage reads.

Starts from the screened list (2b-screen_stations.py). A station that was
ELIGIBLE becomes KEEP if it passed the completeness check
(10-completeness_check.py), otherwise DROP_low_completeness. Stations dropped
during screening keep their screening reason.

Last, any row in the overrides file replaces the rule-based status. That file
is written by hand and is the only place a human decision enters the roster,
so a re-run never overwrites manual work. For Delhi it is empty: the rules
alone reproduce the hand-curated 42 KEEP stations.
"""

import argparse
import os
import pandas as pd

OUTPUT_COLUMNS = ["location_id", "name", "latitude", "longitude",
                  "status", "completeness_pct", "status_source"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screened", required=True, help="Output of 2b-screen_stations.py")
    parser.add_argument("--completeness", required=True, help="station_completeness.csv from 10-completeness_check.py")
    parser.add_argument("--overrides", required=True, help="Hand-written overrides: location_id,status,reason")
    parser.add_argument("--output", required=True, help="Final station roster")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    roster = pd.read_csv(args.screened)
    completeness = pd.read_csv(args.completeness)
    overrides = pd.read_csv(args.overrides)
    print(f"Loaded {len(roster)} screened stations, {len(completeness)} completeness rows, "
          f"{len(overrides)} overrides")

    roster = roster.merge(completeness[["location_id", "completeness_pct", "passes_threshold"]],
                          on="location_id", how="left")

    eligible = roster["status"] == "ELIGIBLE"

    # An eligible station with no completeness row downloaded no usable days
    # at all, so it counts as failing the check.
    missing = eligible & roster["completeness_pct"].isna()
    if missing.sum() > 0:
        print(f"Eligible stations with no completeness row (no data): {roster.loc[missing, 'location_id'].tolist()}")
    roster.loc[missing, "completeness_pct"] = 0.0
    roster.loc[missing, "passes_threshold"] = False

    passed = eligible & (roster["passes_threshold"] == True)
    failed = eligible & (roster["passes_threshold"] == False)
    roster.loc[passed, "status"] = "KEEP"
    roster.loc[failed, "status"] = "DROP_low_completeness"
    roster["status_source"] = "rule"

    for row in overrides.itertuples(index=False):
        match = roster["location_id"] == row.location_id
        if match.sum() == 0:
            raise SystemExit(f"Override for unknown station {row.location_id} -- check {args.overrides}")
        print(f"Override: station {row.location_id} {roster.loc[match, 'status'].iloc[0]} -> {row.status} ({row.reason})")
        roster.loc[match, "status"] = row.status
        roster.loc[match, "status_source"] = "override"

    roster = roster.sort_values("location_id")
    roster[OUTPUT_COLUMNS].to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    print("=== Final roster ===")
    print(roster["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
