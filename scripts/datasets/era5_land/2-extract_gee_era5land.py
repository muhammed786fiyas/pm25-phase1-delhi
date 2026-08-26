import argparse
import os
import yaml
import pandas as pd
import ee


def build_date_chunks(start_date, end_date, chunk_months):
    # splits the study window into smaller pieces so a single getRegion
    # call doesn't try to pull a full year at once and hit GEE's memory limit
    chunks = []
    current_start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while current_start < end:
        current_end = current_start + pd.DateOffset(months=chunk_months)
        if current_end > end:
            current_end = end
        chunks.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start = current_end
    return chunks


def extract_cell_chunk(era5land_chunk, cell_lat, cell_lon):
    point = ee.Geometry.Point(cell_lon, cell_lat)
    region_data = era5land_chunk.getRegion(point, scale=1000).getInfo()
    header = region_data[0]
    rows = region_data[1:]
    return pd.DataFrame(rows, columns=header)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, help="station_cell_mapping.csv")
    parser.add_argument("--params", required=True, help="params.yaml path")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)

    extract_params = params["era5_land"]["extract"]
    gee_project = extract_params["gee_project"]
    collection_id = extract_params["collection"]
    bands = extract_params["bands"]
    study_start = extract_params["study_start"]
    study_end = extract_params["study_end"]
    chunk_months = extract_params["chunk_months"]

    ee.Initialize(project=gee_project)

    mapping = pd.read_csv(args.mapping)
    unique_cells = mapping[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates()
    print(f"Extracting {len(unique_cells)} unique ERA5-Land cells")

    chunks = build_date_chunks(study_start, study_end, chunk_months)
    print(f"Split study window into {len(chunks)} chunks: {chunks}")

    era5land_base = ee.ImageCollection(collection_id).select(bands)

    all_cells = []
    for i in range(len(unique_cells)):
        cell = unique_cells.iloc[i]
        cell_chunks = []
        for chunk_start, chunk_end in chunks:
            era5land_chunk = era5land_base.filterDate(chunk_start, chunk_end)
            chunk_df = extract_cell_chunk(era5land_chunk, cell["cell_lat"], cell["cell_lon"])
            cell_chunks.append(chunk_df)

        cell_df = pd.concat(cell_chunks, ignore_index=True)
        cell_df["cell_id"] = cell["cell_id"]
        all_cells.append(cell_df)
        print(f"cell {cell['cell_id']}: {len(cell_df)} hourly rows")

    raw = pd.concat(all_cells, ignore_index=True)

    raw["datetime_utc"] = pd.to_datetime(raw["time"], unit="ms")

    out_path = os.path.join(args.outdir, "era5_land_raw.csv")
    raw.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Total rows: {len(raw)}")


if __name__ == "__main__":
    main()