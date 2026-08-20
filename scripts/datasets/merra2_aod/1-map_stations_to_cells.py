import argparse
import os
import ee
import pandas as pd
import yaml


def build_station_features(stations_df):
    features = []
    for row in stations_df.itertuples():
        point = ee.Geometry.Point([row.longitude, row.latitude])
        feature = ee.Feature(point, {
            "location_id": row.location_id,
            "name": row.name,
        })
        features.append(feature)
    return ee.FeatureCollection(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", required=True, help="Path to cpcb_stations_delhi_status.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)["merra2_station_mapping"]

    gee_project = params["gee_project"]
    merra2_collection = params["collection"]
    aot_band = params["aot_band"]
    test_date_start = params["test_date_start"]
    test_date_end = params["test_date_end"]

    ee.Initialize(project=gee_project)

    stations_df = pd.read_csv(args.stations)
    stations_df = stations_df[stations_df["status"] == "KEEP"]
    print(f"Loaded {len(stations_df)} KEEP stations")

    # Grab one MERRA-2 image just to read its native grid (projection).
    # We only need this for the grid spacing, not the AOT values themselves.
    test_image = ee.ImageCollection(merra2_collection) \
        .filterDate(test_date_start, test_date_end) \
        .first() \
        .select(aot_band)
    merra_projection = test_image.projection()

    # An image whose pixel values are each pixel's own center coordinate,
    # snapped to MERRA-2's real ~62km grid (not an arbitrary scale — that
    # was the earlier bug, where scale=1000 snapped to a 1km grid instead).
    pixel_centers = ee.Image.pixelLonLat().reproject(merra_projection)

    station_fc = build_station_features(stations_df)

    stations_with_cells = pixel_centers.reduceRegions(
        collection=station_fc,
        reducer=ee.Reducer.first(),
        crs=merra_projection,
    )

    result = stations_with_cells.getInfo()

    rows = []
    for feature in result["features"]:
        props = feature["properties"]
        rows.append({
            "location_id": props["location_id"],
            "name": props["name"],
            "latitude": feature["geometry"]["coordinates"][1],
            "longitude": feature["geometry"]["coordinates"][0],
            "cell_lat": props["latitude"],
            "cell_lon": props["longitude"],
        })

    mapping_df = pd.DataFrame(rows)
    unique_cells = mapping_df[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    unique_cells["cell_id"] = unique_cells.index + 1
    mapping_df = mapping_df.merge(unique_cells, on=["cell_lat", "cell_lon"])

    out_path = os.path.join(args.outdir, "station_cell_mapping.csv")
    mapping_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    unique_cells = mapping_df[["cell_lat", "cell_lon"]].drop_duplicates()
    print(f"{len(mapping_df)} stations map to {len(unique_cells)} unique MERRA-2 cells")


if __name__ == "__main__":
    main()