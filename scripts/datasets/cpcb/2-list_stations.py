"""
List all CPCB stations (via OpenAQ) within a bounding box and save their metadata to CSV.

This is intentionally separate from main.py's full measurement download: before pulling
years of hourly/daily PM2.5 data for ~150-200 stations, you want to see the actual station
inventory first -- names, coordinates, which parameters each station reports, and its
first/last record dates -- so you can:
  1. QC coordinates (blueprint step 5.2) before they get used for 1km AOD extraction
  2. Decide which stations to keep (data coverage window, whether pm25 is even measured)
  3. Map each station to one of the 7 airsheds (Zaid et al. boundaries)

Usage:
    python list_stations.py --region india --output cpcb_stations_india.csv
    python list_stations.py --region chennai --output cpcb_stations_chennai.csv
"""

import argparse
import os
import pandas as pd
from cpcb_fetcher import get_cpcb_locations_full, CITY_BBOX


def main():
    parser = argparse.ArgumentParser(description="List CPCB station metadata (via OpenAQ) for a region")
    parser.add_argument("--region", required=True, choices=list(CITY_BBOX.keys()),
                         help="Region to search (e.g. india, chennai, delhi, gurugram)")
    parser.add_argument("--output", required=True, help="Output CSV filename")
    parser.add_argument("--output-dir", default=os.path.join(os.getcwd(), "CPCB_Measurements"),
                         help="Directory to save the output CSV")
    args = parser.parse_args()

    bbox = CITY_BBOX[args.region]
    stations = get_cpcb_locations_full(*bbox)

    if not stations:
        print("No stations found. Nothing saved.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)

    df = pd.DataFrame(stations)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} station records to {output_path}")

    pm25_stations = df[df["parameters"].str.contains("pm25", case=False, na=False)]
    print(f"   Of which {len(pm25_stations)} report pm25.")


if __name__ == "__main__":
    main()