"""
Diagnostic: find out why /sensors/{id}/hours returns empty results for a date range.

Runs several variants and prints the RAW response so we can see what's actually
happening instead of guessing:
  1. No date filter at all, limit=5   -> confirms the sensor has ANY data, and shows
                                          what the most recent real timestamps look like
  2. datetime_from/datetime_to as plain "YYYY-MM-DD" strings (what download_pm25.py sends)
  3. date_from/date_to (the OTHER param name OpenAQ's own SDK uses for plain dates)
  4. datetime_from/datetime_to as full ISO datetimes with a "Z" suffix (UTC)

Usage:
    python debug_probe.py --location-id 17 --api-key YOUR_KEY
    (or set OPENAQ_API_KEY and drop --api-key)
"""
import argparse
import os
import sys
import json
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://api.openaq.org/v3"


def show(label, resp):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"URL: {resp.url}")
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
    except Exception:
        print("Non-JSON response:", resp.text[:500])
        return
    meta = data.get("meta", {})
    results = data.get("results", [])
    print(f"meta: {json.dumps(meta, indent=2)}")
    print(f"results count: {len(results)}")
    if results:
        print("first result:")
        print(json.dumps(results[0], indent=2)[:1200])
    if "detail" in data:
        print("detail/error field:", data["detail"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--location-id", required=True, type=int)
    p.add_argument("--api-key", default=os.environ.get("OPENAQ_API_KEY"))
    args = p.parse_args()

    if not args.api_key:
        sys.exit("No API key. Pass --api-key or set OPENAQ_API_KEY in .env")

    headers = {"X-API-Key": args.api_key}

    # find the pm25 sensor for this location
    r = requests.get(f"{BASE_URL}/locations/{args.location_id}/sensors", headers=headers)
    show(f"STEP 0: sensors at location {args.location_id}", r)
    sensor_id = None
    for s in r.json().get("results", []):
        if (s.get("parameter") or {}).get("name", "").lower() == "pm25":
            sensor_id = s.get("id")
            break
    if not sensor_id:
        sys.exit("No pm25 sensor found at this location -- stopping here.")
    print(f"\n>>> Using pm25 sensor_id = {sensor_id}\n")

    url = f"{BASE_URL}/sensors/{sensor_id}/hours"

    # 1. no filter at all -- what does the sensor actually have?
    r = requests.get(url, headers=headers, params={"limit": 5, "page": 1})
    show("VARIANT 1: no date filter (limit=5) -- shows real recent timestamps", r)

    # 2. plain date strings under datetime_from/datetime_to (what download_pm25.py sends)
    r = requests.get(url, headers=headers, params={
        "datetime_from": "2025-01-01", "datetime_to": "2025-01-08", "limit": 5, "page": 1
    })
    show("VARIANT 2: datetime_from/datetime_to as plain YYYY-MM-DD", r)

    # 3. date_from/date_to (the OTHER param name pair per OpenAQ's own SDK)
    r = requests.get(url, headers=headers, params={
        "date_from": "2025-01-01", "date_to": "2025-01-08", "limit": 5, "page": 1
    })
    show("VARIANT 3: date_from/date_to as plain YYYY-MM-DD", r)

    # 4. full ISO datetime with Z (UTC) suffix
    r = requests.get(url, headers=headers, params={
        "datetime_from": "2025-01-01T00:00:00Z", "datetime_to": "2025-01-08T00:00:00Z",
        "limit": 5, "page": 1
    })
    show("VARIANT 4: datetime_from/datetime_to as full ISO + Z", r)


if __name__ == "__main__":
    main()