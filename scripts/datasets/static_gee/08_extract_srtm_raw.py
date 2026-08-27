import argparse
import os
import datetime
import yaml
import pandas as pd
import ee

PARAMS_FILE = "params.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["static_gee_layers"]["srtm"]["extract"]

    print("=== Initializing Earth Engine ===")
    ee.Initialize(project=params["gee_project"])

    print("=== Loading station list ===")
    stations = pd.read_csv(params["station_file"])
    stations = stations[stations["status"] == "KEEP"]
    print("Stations to process:", len(stations))

    srtm = ee.Image(params["collection"])
    elevation = srtm.select("elevation")
    slope = ee.Terrain.slope(elevation)
    terrain = elevation.addBands(slope)

    buffer_radius = params["buffer_radius_m"]
    scale = params["scale_m"]

    rows = []
    failed_stations = []

    print("=== Extracting elevation + slope per station ===")
    for index, station in stations.iterrows():
        location_id = station["location_id"]
        name = station["name"]
        lat = station["latitude"]
        lon = station["longitude"]

        try:
            point = ee.Geometry.Point([lon, lat])
            buffer_zone = point.buffer(buffer_radius)

            result = terrain.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer_zone,
                scale=scale,
                maxPixels=1e9,
            )

            values = result.getInfo()

            row = {}
            row["location_id"] = location_id
            row["name"] = name
            row["latitude"] = lat
            row["longitude"] = lon
            row["elevation_m"] = values.get("elevation")
            row["slope_deg"] = values.get("slope")

            rows.append(row)
            print("Done:", location_id, name, "elevation:", row["elevation_m"], "slope:", row["slope_deg"])

        except Exception as error:
            print("FAILED:", location_id, name, "-", error)
            failed_stations.append(location_id)

    print("=== Saving raw output ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df = pd.DataFrame(rows)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("=== Building extraction summary ===")
    summary_rows = []
    summary_rows.append({"metric": "run_timestamp", "value": datetime.datetime.now().isoformat()})
    summary_rows.append({"metric": "srtm_collection", "value": params["collection"]})
    summary_rows.append({"metric": "buffer_radius_m", "value": buffer_radius})
    summary_rows.append({"metric": "scale_m", "value": scale})
    summary_rows.append({"metric": "stations_expected", "value": len(stations)})
    summary_rows.append({"metric": "stations_extracted", "value": len(output_df)})
    summary_rows.append({"metric": "stations_failed", "value": len(failed_stations)})
    summary_rows.append({"metric": "failed_location_ids", "value": str(failed_stations)})

    if len(output_df) > 0:
        summary_rows.append({"metric": "elevation_min", "value": output_df["elevation_m"].min()})
        summary_rows.append({"metric": "elevation_max", "value": output_df["elevation_m"].max()})
        summary_rows.append({"metric": "elevation_mean", "value": output_df["elevation_m"].mean()})
        summary_rows.append({"metric": "slope_min", "value": output_df["slope_deg"].min()})
        summary_rows.append({"metric": "slope_max", "value": output_df["slope_deg"].max()})
        summary_rows.append({"metric": "slope_mean", "value": output_df["slope_deg"].mean()})

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    print("Saved summary to:", args.summary_output)


if __name__ == "__main__":
    main()