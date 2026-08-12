import os
import requests
import time
from datetime import datetime
import pytz

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get("OPENAQ_API_KEY")
if not API_KEY:
    raise SystemExit("No OPENAQ_API_KEY found. Create a .env file with OPENAQ_API_KEY=your_key")

BASE_URL = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": API_KEY}

CITY_BBOX = {
    "chennai": (80.080170, 12.779249, 80.332855, 13.232850),
    "delhi": (76.8381, 28.4126, 77.3477, 28.8814),
    "gurugram": (76.9895, 28.3194, 77.1731, 28.5135),
    # Whole-country bounding box (mainland + islands), used for the national-scope pull.
    # Filtering to provider CPCB happens downstream, so the extra margin (ocean, neighboring
    # border strips) is harmless -- it just won't match any CPCB-provider stations there.
    "india": (68.0, 6.0, 97.5, 37.5),
}

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


def _get_with_retry(url, params=None):
    """GET with simple retry/backoff on rate limits (429) and server errors (5xx)."""
    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 429:
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"   ⏳ Rate limited (429). Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(wait)
            continue
        if response.status_code >= 500:
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"   ⏳ Server error {response.status_code}. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(wait)
            continue
        return response
    return response  # return last (failed) response after exhausting retries


def get_cpcb_locations_full(min_lon, min_lat, max_lon, max_lat, page_limit=1000):
    """
    Fetch full CPCB-provider location metadata (not just IDs) within a bounding box.
    Returns a list of dicts with id, name, lat, lon, locality, timezone, and the
    list of parameters measured at each station -- enough to QC and map stations
    to airsheds before pulling full measurement time series.
    """
    print("Fetching CPCB station metadata...")
    stations = []
    page = 1

    while True:
        url = f"{BASE_URL}/locations"
        params = {
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "limit": page_limit,
            "page": page,
        }
        try:
            response = _get_with_retry(url, params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as e:
            print(f"Failed to fetch locations (page {page}): {e}")
            break

        if not results:
            break

        for loc in results:
            provider = loc.get("provider", {})
            if provider.get("id") == 168 and provider.get("name") == "CPCB":
                coords = loc.get("coordinates", {}) or {}
                country = loc.get("country", {}) or {}
                sensors = loc.get("sensors", []) or []
                params_at_station = sorted({
                    s.get("parameter", {}).get("name")
                    for s in sensors
                    if s.get("parameter", {}).get("name")
                })
                stations.append({
                    "location_id": loc.get("id"),
                    "name": loc.get("name"),
                    "locality": loc.get("locality"),
                    "latitude": coords.get("latitude"),
                    "longitude": coords.get("longitude"),
                    "country_iso": country.get("code"),
                    "timezone": loc.get("timezone"),
                    "is_mobile": loc.get("isMobile"),
                    "parameters": ",".join(params_at_station),
                    "sensor_count": len(sensors),
                    "datetime_first": (loc.get("datetimeFirst") or {}).get("local"),
                    "datetime_last": (loc.get("datetimeLast") or {}).get("local"),
                })

        if len(results) < page_limit:
            break
        page += 1
        time.sleep(1)  # be polite between pages

    if not stations:
        print("No CPCB locations found for the given bounding box.")
    else:
        print(f"Found {len(stations)} CPCB stations.")

    return stations


def get_cpcb_location_ids(min_lon, min_lat, max_lon, max_lat):
    """Backwards-compatible wrapper: returns just the list of CPCB location IDs."""
    stations = get_cpcb_locations_full(min_lon, min_lat, max_lon, max_lat)
    return [s["location_id"] for s in stations]

def get_latest_sensors(location_id):
    url = f"{BASE_URL}/locations/{location_id}/latest"
    try:
        response = requests.get(url, headers=HEADERS)
        time.sleep(1)
        if response.status_code == 200:
            data = response.json()
            return list(set(item.get("sensorsId") for item in data.get("results", []) if item.get("sensorsId")))
    except Exception as e:
        print(f"Failed to fetch sensors for location {location_id}: {e}")
    return []

def parse_datetime(dt_str):
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None

def get_all_measurements(sensor_id, location_id, frequency="hourly", from_date=None, to_date=None, parameter=None):
    all_data = []
    page = 1
    max_pages = 100
    print(f"   🔎 Filtering data from {from_date} to {to_date}")
    if parameter:
        print(f"   🔍 Filtering only for parameter: {parameter}")

    tz = pytz.timezone("Asia/Kolkata")
    from_dt = tz.localize(datetime.strptime(from_date, "%Y-%m-%d"))
    to_dt = tz.localize(datetime.strptime(to_date, "%Y-%m-%d"))

    while page <= max_pages:
        url = f"{BASE_URL}/sensors/{sensor_id}/measurements/{frequency}"
        params = {"page": page, "limit": 100}
        try:
            response = _get_with_retry(url, params=params)
            time.sleep(1)
            if response.status_code != 200:
                print(f"Sensor {sensor_id} page {page} failed: {response.status_code}")
                break
            results = response.json().get("results", [])
        except Exception as e:
            print(f"Error fetching data for sensor {sensor_id}: {e}")
            break

        if not results:
            break

        for entry in results:
            period = entry.get("period", {})
            period_from_str = period.get("datetimeFrom", {}).get("local")
            period_from = parse_datetime(period_from_str)
            if not period_from or not (from_dt <= period_from < to_dt):
                continue

            param = entry.get("parameter", {})
            param_name = param.get("name", "").lower()

            if parameter and param_name != parameter:
                continue

            summary = entry.get("summary", {})
            coverage = entry.get("coverage", {})

            record = {
                "location_id": location_id,
                "sensor_id": sensor_id,
                "value": entry.get("value"),
                "parameter_id": param.get("id"),
                "parameter_name": param.get("name"),
                "units": param.get("units"),
                "period_label": period.get("label"),
                "period_from_utc": period.get("datetimeFrom", {}).get("utc"),
                "period_to_utc": period.get("datetimeTo", {}).get("utc"),
                "period_from_local": period_from_str,
                "period_to_local": period.get("datetimeTo", {}).get("local"),
                "summary_avg": summary.get("avg"),
                "summary_min": summary.get("min"),
                "summary_max": summary.get("max"),
                "summary_median": summary.get("median"),
                "summary_sd": summary.get("sd"),
                "coverage_percent": coverage.get("percentCoverage"),
                "coverage_complete": coverage.get("percentComplete"),
                "coverage_expected_count": coverage.get("expectedCount"),
                "coverage_observed_count": coverage.get("observedCount"),
                "coverage_from_utc": coverage.get("datetimeFrom", {}).get("utc"),
                "coverage_to_utc": coverage.get("datetimeTo", {}).get("utc"),
                "coverage_from_local": coverage.get("datetimeFrom", {}).get("local"),
                "coverage_to_local": coverage.get("datetimeTo", {}).get("local"),
            }

            all_data.append(record)

        page += 1

    if not all_data:
        print(f"No data found for sensor {sensor_id} in the given range.")

    return all_data