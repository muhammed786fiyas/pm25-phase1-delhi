import argparse
import os
import datetime
import yaml
import pandas as pd
import ee

PARAMS_FILE = "params.yaml"

NDVI_NIR_BAND = "B8"
NDVI_RED_BAND = "B4"
CLOUD_BIT = 10
CIRRUS_BIT = 11


def mask_s2_clouds(image):
    qa = image.select("QA60")
    cloud_bit_mask = 1 << CLOUD_BIT
    cirrus_bit_mask = 1 << CIRRUS_BIT
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)


def build_period_list(study_start, study_end, period_days):
    periods = []
    current_start = datetime.datetime.strptime(study_start, "%Y-%m-%d").date()
    final_end = datetime.datetime.strptime(study_end, "%Y-%m-%d").date()
    period_index = 0

    while current_start <= final_end:
        current_end = current_start + datetime.timedelta(days=period_days - 1)
        if current_end > final_end:
            current_end = final_end

        periods.append({
            "period_index": period_index,
            "period_start": current_start.isoformat(),
            "period_end": current_end.isoformat(),
        })

        current_start = current_end + datetime.timedelta(days=1)
        period_index = period_index + 1

    return periods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    args = parser.parse_args()

    print("=== Loading params ===")
    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    params = all_params["static_gee_layers"]["ndvi"]["extract"]

    print("=== Initializing Earth Engine ===")
    ee.Initialize(project=params["gee_project"])

    print("=== Loading station list ===")
    stations = pd.read_csv(params["station_file"])
    stations = stations[stations["status"] == "KEEP"]
    print("Stations to process:", len(stations))

    buffer_radius = params["buffer_radius_m"]
    scale = params["scale_m"]

    print("=== Building station FeatureCollection ===")
    station_features = []
    for index, station in stations.iterrows():
        location_id = station["location_id"]
        name = station["name"]
        lat = station["latitude"]
        lon = station["longitude"]

        point = ee.Geometry.Point([lon, lat])
        buffer_zone = point.buffer(buffer_radius)

        feature = ee.Feature(buffer_zone, {
            "location_id": location_id,
            "name": name,
            "latitude": lat,
            "longitude": lon,
        })
        station_features.append(feature)

    station_fc = ee.FeatureCollection(station_features)
    region_bounds = station_fc.geometry().bounds()

    print("=== Building period list ===")
    periods = build_period_list(params["study_start"], params["study_end"], params["period_days"])
    print("Total periods:", len(periods))

    rows = []
    failed_periods = []
    zero_image_periods = []

    print("=== Extracting NDVI per period (batch across all stations) ===")
    for period in periods:
        period_index = period["period_index"]
        period_start = period["period_start"]
        period_end = period["period_end"]

        try:
            s2_collection = (
                ee.ImageCollection(params["collection"])
                .filterBounds(region_bounds)
                .filterDate(period_start, ee.Date(period_end).advance(1, "day"))
                .map(mask_s2_clouds)
            )

            image_count = s2_collection.size().getInfo()

            if image_count == 0:
                print("Period", period_index, "(", period_start, "to", period_end, "): 0 images, skipping")
                zero_image_periods.append(period_index)
                continue

            composite = s2_collection.median()
            ndvi = composite.normalizedDifference([NDVI_NIR_BAND, NDVI_RED_BAND]).rename("NDVI")

            result_fc = ndvi.reduceRegions(
                collection=station_fc,
                reducer=ee.Reducer.mean(),
                scale=scale,
            )

            result_features = result_fc.getInfo()["features"]

            for result_feature in result_features:
                props = result_feature["properties"]
                row = {}
                row["period_index"] = period_index
                row["period_start"] = period_start
                row["period_end"] = period_end
                row["location_id"] = props["location_id"]
                row["name"] = props["name"]
                row["latitude"] = props["latitude"]
                row["longitude"] = props["longitude"]
                row["ndvi_mean"] = props.get("mean")
                row["s2_image_count"] = image_count
                rows.append(row)

            print("Period", period_index, "(", period_start, "to", period_end, "): done,", image_count, "images,", len(result_features), "stations")

        except Exception as error:
            print("FAILED period", period_index, "(", period_start, "to", period_end, ") -", error)
            failed_periods.append(period_index)

    print("=== Saving raw output ===")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df = pd.DataFrame(rows)
    output_df.to_csv(args.output, index=False)
    print("Saved to:", args.output)

    print("=== Building extraction summary ===")
    summary_rows = []
    summary_rows.append({"metric": "run_timestamp", "value": datetime.datetime.now().isoformat()})
    summary_rows.append({"metric": "s2_collection", "value": params["collection"]})
    summary_rows.append({"metric": "buffer_radius_m", "value": buffer_radius})
    summary_rows.append({"metric": "scale_m", "value": scale})
    summary_rows.append({"metric": "period_days", "value": params["period_days"]})
    summary_rows.append({"metric": "total_periods", "value": len(periods)})
    summary_rows.append({"metric": "periods_failed", "value": len(failed_periods)})
    summary_rows.append({"metric": "failed_period_indices", "value": str(failed_periods)})
    summary_rows.append({"metric": "periods_with_zero_images", "value": len(zero_image_periods)})
    summary_rows.append({"metric": "zero_image_period_indices", "value": str(zero_image_periods)})
    summary_rows.append({"metric": "stations_expected", "value": len(stations)})
    summary_rows.append({"metric": "total_station_period_rows", "value": len(output_df)})

    if len(output_df) > 0:
        null_ndvi_count = output_df["ndvi_mean"].isna().sum()
        summary_rows.append({"metric": "rows_with_null_ndvi", "value": null_ndvi_count})
        summary_rows.append({"metric": "ndvi_min", "value": output_df["ndvi_mean"].min()})
        summary_rows.append({"metric": "ndvi_max", "value": output_df["ndvi_mean"].max()})
        summary_rows.append({"metric": "ndvi_mean_overall", "value": output_df["ndvi_mean"].mean()})

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)
    print("Saved summary to:", args.summary_output)


if __name__ == "__main__":
    main()