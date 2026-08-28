"""
Extract total OSM road length within a 1km buffer of each CPCB station.
Input: a pre-downloaded OSM roads export (GeoJSON, highway=* ways) covering
the Delhi/NCR area -- see docs/logs/tasks/6-OSM_Features.md for how this
file was obtained (manual Overpass Turbo export, live API is blocked from
both the cloud sandbox and this machine's shell).
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
    parser.add_argument("--roads_geojson", required=True)
    parser.add_argument("--buffer_radius_m", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    print("=== Loading stations ===")
    stations_df = load_stations(args.station_file)
    print("KEEP stations:", len(stations_df))

    print("=== Loading roads (this can take a minute for a large file) ===")
    roads_gdf = gpd.read_file(args.roads_geojson)
    print("Raw road features loaded:", len(roads_gdf))
    roads_gdf = roads_gdf[roads_gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    roads_gdf = roads_gdf.set_crs("EPSG:4326", allow_override=True)
    roads_gdf_utm = roads_gdf.to_crs(UTM_CRS)
    print("Line features after filtering:", len(roads_gdf_utm))

    rows = []
    failed_stations = []

    print("=== Computing road length per station buffer ===")
    for index, row in stations_df.iterrows():
        location_id = row["location_id"]
        name = row["name"]
        lat = row["latitude"]
        lon = row["longitude"]

        try:
            point_utm = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(UTM_CRS).iloc[0]
            buffer_poly = point_utm.buffer(args.buffer_radius_m)

            clipped = roads_gdf_utm[roads_gdf_utm.intersects(buffer_poly)]
            clipped_geom = clipped.geometry.intersection(buffer_poly)
            total_length_m = clipped_geom.length.sum()

            rows.append({
                "location_id": location_id,
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "buffer_radius_m": args.buffer_radius_m,
                "n_road_segments": len(clipped),
                "total_road_length_m": total_length_m,
            })
            print("Done:", location_id, name, "-- length_m:", round(total_length_m, 1))
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
        "min_length_m": [output_df["total_road_length_m"].min() if len(output_df) else None],
        "max_length_m": [output_df["total_road_length_m"].max() if len(output_df) else None],
        "median_length_m": [output_df["total_road_length_m"].median() if len(output_df) else None],
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
