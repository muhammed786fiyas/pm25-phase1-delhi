import argparse
import os
import ee
import pandas as pd
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)["maiac_extract"]

GEE_PROJECT = params["gee_project"]
MAIAC_COLLECTION = params["collection"]
BANDS_TO_PULL = params["bands"]
STUDY_START = params["study_start"]
STUDY_END = params["study_end"]


def extract_station_data(location_id, latitude, longitude):
    # build the point for this station
    point = ee.Geometry.Point([longitude, latitude])

    # same filtering pattern we validated in the Code Editor:
    # date range first, then filterBounds as a coarse geographic filter
    collection = (
        ee.ImageCollection(MAIAC_COLLECTION)
        .filterDate(STUDY_START, STUDY_END)
        .filterBounds(point)
    )

    # this function runs once per image, server-side, via .map() below.
    # it reads the real pixel value at this station's point and tags
    # the image with those values as properties we can read later.
    def tag_image_with_values(img):
        values = img.select(BANDS_TO_PULL).reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=1000,
        )
        feature = ee.Feature(
            None,
            {
                "location_id": location_id,
                "date": img.date().format("YYYY-MM-dd"),
                "image_id": img.get("system:index"),
                "aod_055": values.get("Optical_Depth_055"),
                "aod_047": values.get("Optical_Depth_047"),
                "aod_uncertainty": values.get("AOD_Uncertainty"),
                "aod_qa": values.get("AOD_QA"),
            },
        )
        return feature

    tagged = collection.map(tag_image_with_values)

    # keep only overpasses where we actually got a real AOD value
    # (drops the swath false-positives we saw during testing)
    valid_only = tagged.filter(ee.Filter.neq("aod_055", None))

    # this is the point where the computation actually runs and comes
    # back to Python - everything above this line was just describing
    # the computation, not running it
    result = valid_only.getInfo()

    # result["features"] is a list of dicts, one per valid overpass
    rows = []
    for item in result["features"]:
        rows.append(item["properties"])

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", required=True, help="CSV of station_id, latitude, longitude")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ee.Initialize(project=GEE_PROJECT)

    stations_df = pd.read_csv(args.stations)

    all_rows = []
    for index, row in stations_df.iterrows():
        location_id = row["location_id"]
        latitude = row["latitude"]
        longitude = row["longitude"]

        station_rows = extract_station_data(location_id, latitude, longitude)
        all_rows.extend(station_rows)

        print(f"{location_id}: {len(station_rows)} valid overpasses")

    out_df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.outdir, "maiac_aod_raw.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Total rows: {len(out_df)}")
    print(f"Stations covered: {out_df['location_id'].nunique()}")


if __name__ == "__main__":
    main()