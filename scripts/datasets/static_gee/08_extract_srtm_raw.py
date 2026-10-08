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
    # Optional override of the roster named in params. The prediction grid is
    # written in the station file's exact schema (see scripts/grid/01_build_grid.py),
    # so pointing this at a grid CSV runs the same extractor over grid cells
    # instead of stations. Default None keeps the station behaviour unchanged.
    parser.add_argument("--stations", default=None,
                        help="station/grid roster CSV; defaults to params station_file")
    # One request per point costs ~2.5 s, which is fine for 42 stations and
    # ~100 minutes for a 2388-cell prediction grid. Batching sends many buffers
    # in a single reduceRegions call instead: measured 2.0 s for all 42 stations,
    # and -- verified against the committed station output -- it reproduces
    # elevation and slope to 1e-9 on all 42. The geometry, reducer and scale are
    # read from the same params either way, so this changes HOW the request is
    # sent and not WHAT is computed. Default 0 keeps the per-point path, so
    # existing station stages are untouched.
    parser.add_argument("--batch_size", type=int, default=0,
                        help="points per reduceRegions call; 0 = one call per point")
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
    station_file = args.stations if args.stations else params["station_file"]
    stations = pd.read_csv(station_file)
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

    if args.batch_size > 0:
        print(f"=== Extracting elevation + slope, batched ({args.batch_size} per call) ===")
        station_list = stations.reset_index(drop=True)
        for start in range(0, len(station_list), args.batch_size):
            block = station_list.iloc[start:start + args.batch_size]
            features = []
            for station in block.itertuples(index=False):
                features.append(ee.Feature(
                    ee.Geometry.Point([station.longitude, station.latitude])
                    .buffer(buffer_radius),
                    {"location_id": int(station.location_id)}))
            collection = ee.FeatureCollection(features)
            try:
                result = terrain.reduceRegions(
                    collection=collection,
                    reducer=ee.Reducer.mean(),
                    scale=scale,
                ).getInfo()
            except Exception as error:
                print("FAILED batch starting at", start, "-", error)
                failed_stations.extend(block["location_id"].tolist())
                continue
            by_id = {}
            for feature in result["features"]:
                properties = feature["properties"]
                by_id[properties["location_id"]] = properties
            for station in block.itertuples(index=False):
                properties = by_id.get(int(station.location_id))
                if properties is None:
                    failed_stations.append(station.location_id)
                    continue
                rows.append({
                    "location_id": station.location_id,
                    "name": station.name,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "elevation_m": properties.get("elevation"),
                    "slope_deg": properties.get("slope"),
                })
            print(f"  {min(start + args.batch_size, len(station_list))}"
                  f"/{len(station_list)} points")
        stations = stations.iloc[0:0]

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