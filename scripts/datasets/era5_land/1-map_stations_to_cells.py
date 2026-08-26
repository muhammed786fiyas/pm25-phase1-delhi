import argparse
import os
import yaml
import pandas as pd
import ee


def get_cell_for_station(pixel_lonlat_image, lon, lat):
    point = ee.Geometry.Point(lon, lat)
    result = pixel_lonlat_image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=1000
    ).getInfo()
    return result["longitude"], result["latitude"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", required=True, help="CPCB station status CSV")
    parser.add_argument("--params", required=True, help="params.yaml path")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)

    gee_project = params["era5_land"]["map_stations_to_cells"]["gee_project"]
    collection_id = params["era5_land"]["map_stations_to_cells"]["collection"]

    ee.Initialize(project=gee_project)

    stations = pd.read_csv(args.stations)
    stations = stations[stations["status"] == "KEEP"]
    print(f"Loaded {len(stations)} KEEP stations")

    # pixelLonLat reprojected onto ERA5-Land's own grid snaps any query point
    # to the center of whichever ~11km cell it falls inside
    era5land = ee.ImageCollection(collection_id)
    era5_projection = era5land.first().select("temperature_2m").projection()
    pixel_lonlat = ee.Image.pixelLonLat().reproject(era5_projection)

    rows = []
    for i in range(len(stations)):
        station = stations.iloc[i]
        cell_lon, cell_lat = get_cell_for_station(
            pixel_lonlat, station["longitude"], station["latitude"]
        )
        rows.append({
            "location_id": station["location_id"],
            "name": station["name"],
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "cell_lat": cell_lat,
            "cell_lon": cell_lon,
        })
        print(f"{station['name']} -> cell ({cell_lat:.4f}, {cell_lon:.4f})")

    mapping = pd.DataFrame(rows)

    # assign a simple cell_id per unique (cell_lat, cell_lon) pair — same
    # drop_duplicates + merge pattern as MERRA-2, since pd.factorize broke
    # on this pandas version
    unique_cells = mapping[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    unique_cells["cell_id"] = unique_cells.index + 1
    mapping = mapping.merge(unique_cells, on=["cell_lat", "cell_lon"])

    out_path = os.path.join(args.outdir, "station_cell_mapping.csv")
    mapping.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    print(f"=== {mapping['cell_id'].nunique()} unique ERA5-Land cells ===")
    for cell_id in sorted(mapping["cell_id"].unique()):
        count = len(mapping[mapping["cell_id"] == cell_id])
        print(f"cell {cell_id}: {count} stations")


if __name__ == "__main__":
    main()