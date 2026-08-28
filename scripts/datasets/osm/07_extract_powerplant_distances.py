"""
Compute distance from each CPCB station to the nearest OSM power plant
(power=plant). Search extent is much wider than the 1km station buffer --
see docs/logs/tasks/6-OSM_Features.md for the download bbox used and why.
"""

import argparse
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

UTM_CRS = "EPSG:32643"  # UTM zone 43N -- Delhi/NCR


def load_stations(station_file):
    df = pd.read_csv(station_file)
    df = df[df["status"] == "KEEP"].copy()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station_file", required=True)
    parser.add_argument("--powerplants_geojson", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    print("=== Loading stations ===")
    stations_df = load_stations(args.station_file)
    print("KEEP stations:", len(stations_df))

    print("=== Loading power plants ===")
    pp_gdf = gpd.read_file(args.powerplants_geojson)
    print("Raw power plant features loaded:", len(pp_gdf))
    pp_gdf = pp_gdf.set_crs("EPSG:4326", allow_override=True)

    # collapse polygons/multipolygons to centroids so every plant is one point
    pp_gdf_utm = pp_gdf.to_crs(UTM_CRS)
    pp_gdf_utm["geometry"] = pp_gdf_utm.geometry.centroid
    print("Power plant point locations:", len(pp_gdf_utm))

    if "name" not in pp_gdf_utm.columns:
        pp_gdf_utm["name"] = None

    rows = []
    failed_stations = []

    print("=== Computing nearest power plant distance per station ===")
    for index, row in stations_df.iterrows():
        location_id = row["location_id"]
        name = row["name"]
        lat = row["latitude"]
        lon = row["longitude"]

        try:
            point_utm = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(UTM_CRS).iloc[0]
            distances_m = pp_gdf_utm.geometry.distance(point_utm)

            nearest_idx = distances_m.idxmin()
            nearest_dist_m = distances_m.loc[nearest_idx]
            nearest_name = pp_gdf_utm.loc[nearest_idx, "name"]

            rows.append({
                "location_id": location_id,
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "dist_to_nearest_powerplant_m": nearest_dist_m,
                "nearest_powerplant_name": nearest_name,
                "n_powerplants_in_extract": len(pp_gdf_utm),
            })
            print("Done:", location_id, name, "-- nearest_dist_m:", round(nearest_dist_m, 1), "--", nearest_name)
        except Exception as e:
            print("FAILED:", location_id, name, "--", e)
            failed_stations.append({"location_id": location_id, "name": name, "error": str(e)})

    print("=== Saving raw output ===")
    output_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    summary_df = pd.DataFrame({
        "n_stations_total": [len(stations_df)],
        "n_stations_ok": [len(rows)],
        "n_stations_failed": [len(failed_stations)],
        "n_powerplants_in_extract": [len(pp_gdf_utm)],
        "min_dist_m": [output_df["dist_to_nearest_powerplant_m"].min() if len(output_df) else None],
        "max_dist_m": [output_df["dist_to_nearest_powerplant_m"].max() if len(output_df) else None],
    })
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    print("Summary saved to:", args.summary_output)

    if failed_stations:
        print("=== FAILED STATIONS ===")
        for f in failed_stations:
            print(f)


if __name__ == "__main__":
    main()
