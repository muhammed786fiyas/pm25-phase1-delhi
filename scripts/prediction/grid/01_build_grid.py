import argparse
import math
import os

import numpy as np
import pandas as pd
import yaml

# Builds the prediction grid for Delhi and writes it in the EXACT schema of
# data/stations/cpcb_stations_delhi_status.csv -- location_id, name, latitude,
# longitude, status -- so every existing feature extractor consumes it with no
# code change at all. They all read that file and filter status == "KEEP";
# a grid cell is just a "station" that has no PM2.5 measurement.
#
# That is the whole trick behind step 6: the static and reanalysis covariates
# need no new extraction code, only a different input file.
#
# Cells are filtered to those within COVERAGE radius of a real station, because
# that is where the API is willing to answer at all (see app/api.py's
# DEFAULT_MAX_STATION_DISTANCE_KM). Extracting features for cells we would
# refuse to serve is wasted GEE time.

EARTH_RADIUS_KM = 6371.0

# Synthetic ids start well above the real CPCB ids (max 11607) so a grid row
# can never be confused with a station row if the two are ever concatenated.
GRID_ID_BASE = 900000


def haversine_km(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = (math.sin(delta_lat / 2) ** 2
         + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def distance_to_nearest_station(lat, lon, stations):
    nearest_km = None
    nearest_id = None
    for station in stations.itertuples(index=False):
        d = haversine_km(lat, lon, station.latitude, station.longitude)
        if nearest_km is None or d < nearest_km:
            nearest_km = d
            nearest_id = station.location_id
    return nearest_km, nearest_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--stations", required=True,
                        help="station roster, used for the coverage-radius filter")
    parser.add_argument("--output", required=True, help="grid CSV in station schema")
    parser.add_argument("--summary_output", required=True, help="one-row build summary")
    args = parser.parse_args()

    for path in [args.output, args.summary_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)["prediction"]["grid"]

    grid_km = params["grid_km"]
    coverage_km = params["coverage_radius_km"]
    min_lon = params["bbox_min_lon"]
    min_lat = params["bbox_min_lat"]
    max_lon = params["bbox_max_lon"]
    max_lat = params["bbox_max_lat"]

    stations = pd.read_csv(args.stations)
    stations = stations[stations["status"] == "KEEP"]
    print(f"Loaded {len(stations)} KEEP stations for the coverage filter")

    # Degrees per km vary with latitude for longitude but not for latitude, so
    # the longitude step is computed at the bbox's mid-latitude. Over Delhi's
    # 0.47 degree span the error in cell size is well under 1%.
    mid_lat = (min_lat + max_lat) / 2
    lat_step = grid_km / 110.57
    lon_step = grid_km / (111.32 * math.cos(math.radians(mid_lat)))
    print(f"Grid spacing {grid_km} km -> {lat_step:.5f} deg lat, {lon_step:.5f} deg lon "
          f"at {mid_lat:.3f} N")

    lats = np.arange(min_lat, max_lat, lat_step)
    lons = np.arange(min_lon, max_lon, lon_step)
    print(f"Bounding box grid: {len(lats)} rows x {len(lons)} cols = {len(lats) * len(lons)} cells")

    rows = []
    n_dropped = 0
    cell_index = 0
    for row_index in range(len(lats)):
        for col_index in range(len(lons)):
            lat = float(lats[row_index])
            lon = float(lons[col_index])
            nearest_km, nearest_id = distance_to_nearest_station(lat, lon, stations)
            if nearest_km > coverage_km:
                n_dropped = n_dropped + 1
                continue
            rows.append({
                "location_id": GRID_ID_BASE + cell_index,
                "name": f"grid_{grid_km:g}km_r{row_index:04d}c{col_index:04d}",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "status": "KEEP",
                "grid_km": grid_km,
                "grid_row": row_index,
                "grid_col": col_index,
                "dist_to_nearest_station_km": round(nearest_km, 3),
                "nearest_station_id": nearest_id,
            })
            cell_index = cell_index + 1

    grid_df = pd.DataFrame(rows)
    print(f"Kept {len(grid_df)} cells within {coverage_km} km of a station, "
          f"dropped {n_dropped} outside it")

    if len(grid_df) == 0:
        raise SystemExit("ERROR: no grid cells survived the coverage filter")

    grid_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    summary = pd.DataFrame([{
        "grid_km": grid_km,
        "coverage_radius_km": coverage_km,
        "n_cells_in_bbox": len(lats) * len(lons),
        "n_cells_kept": len(grid_df),
        "n_cells_dropped": n_dropped,
        "lat_step_deg": round(lat_step, 6),
        "lon_step_deg": round(lon_step, 6),
        "min_dist_to_station_km": round(grid_df["dist_to_nearest_station_km"].min(), 3),
        "max_dist_to_station_km": round(grid_df["dist_to_nearest_station_km"].max(), 3),
        "median_dist_to_station_km": round(grid_df["dist_to_nearest_station_km"].median(), 3),
    }])
    summary.to_csv(args.summary_output, index=False)
    print(f"Wrote {args.summary_output}")

    print("=== distance to nearest station, kept cells ===")
    print(grid_df["dist_to_nearest_station_km"].describe().to_string())


if __name__ == "__main__":
    main()
