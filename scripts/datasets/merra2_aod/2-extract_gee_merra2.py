import argparse
import os
import ee
import pandas as pd
import yaml


def extract_cell_timeseries(collection, band, point, scale):
    # getRegion pulls the whole hourly time series for this point in one
    # GEE call -- no need to loop image by image like the MAIAC script did.
    table = collection.select(band).getRegion(point, scale).getInfo()
    header = table[0]
    rows = table[1:]
    return pd.DataFrame(rows, columns=header)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, help="Path to station_cell_mapping.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)["merra2_extraction"]

    gee_project = params["gee_project"]
    merra2_collection = params["collection"]
    aot_band = params["aot_band"]
    date_start = params["date_start"]
    date_end = params["date_end"]
    scale_meters = params["scale_meters"]

    ee.Initialize(project=gee_project)

    mapping_df = pd.read_csv(args.mapping)
    unique_cells = mapping_df[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates()
    print(f"Extracting MERRA-2 for {len(unique_cells)} unique cells")

    collection = ee.ImageCollection(merra2_collection).filterDate(date_start, date_end)

    all_hourly = []
    for row in unique_cells.itertuples():
        print(f"Cell {row.cell_id}: ({row.cell_lat}, {row.cell_lon})")
        point = ee.Geometry.Point([row.cell_lon, row.cell_lat])

        hourly_df = extract_cell_timeseries(collection, aot_band, point, scale_meters)
        hourly_df["cell_id"] = row.cell_id
        print(f"  {len(hourly_df)} hourly rows")
        all_hourly.append(hourly_df)

    raw_df = pd.concat(all_hourly, ignore_index=True)

    out_path = os.path.join(args.outdir, "merra2_aod_raw.csv")
    raw_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"{raw_df['cell_id'].nunique()} cells, {len(raw_df)} total hourly rows")


if __name__ == "__main__":
    main()