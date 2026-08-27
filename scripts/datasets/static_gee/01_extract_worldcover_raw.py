import argparse
import os
import yaml
import datetime
import pandas as pd
import ee

PARAMS_FILE = "params.yaml"
PARAMS_SECTION = "static_gee_layers"

WORLDCOVER_CLASSES = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "95", "100"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params[PARAMS_SECTION]

    print("=== Initializing Earth Engine ===")
    ee.Initialize(project=params["GEE_PROJECT"])

    print("=== Loading station list ===")
    stations = pd.read_csv(params["STATION_FILE"])
    stations = stations[stations["status"] == "KEEP"]
    print("Stations to process:", len(stations))

    worldcover = ee.Image(params["WORLDCOVER_COLLECTION"]).select(params["WORLDCOVER_BAND"])
    buffer_radius = params["BUFFER_RADIUS_M"]
    scale = params["WORLDCOVER_SCALE_M"]

    rows = []
    failed_stations = []

    print("=== Extracting raw pixel counts per station ===")
    for index, station in stations.iterrows():
        location_id = station["location_id"]
        name = station["name"]
        lat = station["latitude"]
        lon = station["longitude"]

        try:
            point = ee.Geometry.Point([lon, lat])
            buffer_zone = point.buffer(buffer_radius)

            hist_result = worldcover.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=buffer_zone,
                scale=scale,
                maxPixels=1e9,
            )

            class_counts = hist_result.get(params["WORLDCOVER_BAND"]).getInfo()

            row = {}
            row["location_id"] = location_id
            row["name"] = name
            row["latitude"] = lat
            row["longitude"] = lon

            total_pixels = 0
            for class_code in WORLDCOVER_CLASSES:
                count = class_counts.get(class_code, 0)
                row["class_" + class_code + "_count"] = count
                total_pixels = total_pixels + count

            row["total_pixels"] = total_pixels

            rows.append(row)
            print("Done:", location_id, name, "total_pixels:", total_pixels)

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
    summary_rows.append({"metric": "worldcover_collection", "value": params["WORLDCOVER_COLLECTION"]})
    summary_rows.append({"metric": "buffer_radius_m", "value": buffer_radius})
    summary_rows.append({"metric": "scale_m", "value": scale})
    summary_rows.append({"metric": "stations_expected", "value": len(stations)})
    summary_rows.append({"metric": "stations_extracted", "value": len(output_df)})
    summary_rows.append({"metric": "stations_failed", "value": len(failed_stations)})
    summary_rows.append({"metric": "failed_location_ids", "value": str(failed_stations)})

    if len(output_df) > 0:
        summary_rows.append({"metric": "total_pixels_min", "value": output_df["total_pixels"].min()})
        summary_rows.append({"metric": "total_pixels_max", "value": output_df["total_pixels"].max()})
        summary_rows.append({"metric": "total_pixels_median", "value": output_df["total_pixels"].median()})
        zero_pixel_count = (output_df["total_pixels"] == 0).sum()
        summary_rows.append({"metric": "stations_with_zero_pixels", "value": zero_pixel_count})

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    print("Saved summary to:", args.summary_output)


if __name__ == "__main__":
    main()