import argparse
import os
import yaml
import datetime
import pandas as pd
import ee

PARAMS_FILE = "params.yaml"

WORLDCOVER_CLASSES = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "95", "100"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    # Optional override of the roster named in params. The prediction grid is
    # written in the station file's exact schema (see scripts/grid/01_build_grid.py),
    # so pointing this at a grid CSV runs the same extractor over grid cells
    # instead of stations. Default None keeps the station behaviour unchanged.
    parser.add_argument("--stations", default=None,
                        help="station/grid roster CSV; defaults to params station_file")
    # One frequencyHistogram request per point is fine for 42 stations and about
    # 100 minutes for a 2388-cell prediction grid. Batching sends many buffers in
    # a single reduceRegions call instead. The geometry, reducer and scale come
    # from the same params either way, so this changes HOW the request is sent,
    # not WHAT is computed -- verified by running the batched path over the 42
    # stations and diffing against the committed output. Default 0 keeps the
    # per-point path, so the station stages are untouched.
    parser.add_argument("--batch_size", type=int, default=0,
                        help="points per reduceRegions call; 0 = one call per point")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["static_gee_layers"]["worldcover"]["extract"]

    print("=== Initializing Earth Engine ===")
    ee.Initialize(project=params["gee_project"])

    print("=== Loading station list ===")
    station_file = args.stations if args.stations else params["station_file"]
    stations = pd.read_csv(station_file)
    stations = stations[stations["status"] == "KEEP"]
    print("Stations to process:", len(stations))

    worldcover = ee.Image(params["collection"]).select(params["band"])
    buffer_radius = params["buffer_radius_m"]
    scale = params["scale_m"]

    rows = []
    failed_stations = []

    if args.batch_size > 0:
        print(f"=== Extracting raw pixel counts, batched "
              f"({args.batch_size} per call) ===")
        station_list = stations.reset_index(drop=True)
        for start in range(0, len(station_list), args.batch_size):
            block = station_list.iloc[start:start + args.batch_size]
            features = []
            for station in block.itertuples(index=False):
                features.append(ee.Feature(
                    ee.Geometry.Point([station.longitude, station.latitude])
                    .buffer(buffer_radius),
                    {"location_id": int(station.location_id)}))
            try:
                result = worldcover.reduceRegions(
                    collection=ee.FeatureCollection(features),
                    reducer=ee.Reducer.frequencyHistogram(),
                    scale=scale,
                ).getInfo()
            except Exception as error:
                print("FAILED batch starting at", start, "-", error)
                failed_stations.extend(block["location_id"].tolist())
                continue
            by_id = {}
            for feature in result["features"]:
                properties = feature["properties"]
                by_id[properties["location_id"]] = properties.get("histogram", {})
            for station in block.itertuples(index=False):
                class_counts = by_id.get(int(station.location_id))
                if class_counts is None:
                    failed_stations.append(station.location_id)
                    continue
                row = {}
                row["location_id"] = station.location_id
                row["name"] = station.name
                row["latitude"] = station.latitude
                row["longitude"] = station.longitude
                total_pixels = 0
                for class_code in WORLDCOVER_CLASSES:
                    count = class_counts.get(class_code, 0)
                    row["class_" + class_code + "_count"] = count
                    total_pixels = total_pixels + count
                row["total_pixels"] = total_pixels
                rows.append(row)
            print(f"  {min(start + args.batch_size, len(station_list))}"
                  f"/{len(station_list)} points")
        stations = stations.iloc[0:0]

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

            class_counts = hist_result.get(params["band"]).getInfo()

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
    summary_rows.append({"metric": "worldcover_collection", "value": params["collection"]})
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