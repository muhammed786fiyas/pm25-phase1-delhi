import argparse
import os
import time

import ee
import pandas as pd
import yaml

# Extracts MAIAC AOD for every prediction grid cell, in BATCHES.
#
# Why this script exists rather than reusing scripts/datasets/maiac_aod/1-:
# that one queries ONE POINT AT A TIME over the whole study window, which is
# fine for 42 stations and hopeless for 2388 grid cells. Measured 2026-10-08:
# 67.6 s per point (82.6 / 60.8 / 59.3 s at three bbox corners), so the station
# run is ~47 min and the 1 km grid would be 44.8 HOURS sequential.
#
# The fix is the pattern scripts/datasets/static_gee/04_extract_ndvi_raw.py
# already uses for NDVI: map over the image collection and reduceRegions over
# MANY points at once, so one request serves many cells. Measured on the same
# day: 100 cells x 1 month returned 3953 rows in 16.7 s, i.e. ~80 min for the
# whole 1 km grid -- a 33x speedup.
#
# Why it must be chunked rather than one big call: Earth Engine aborts a
# collection query after accumulating 5000 elements. 500 points x 1 week failed
# on exactly that. Chunking by (cell block x month) keeps each request's result
# under the limit; CELLS_PER_CHUNK is sized so block x ~30 days x ~1.2 valid
# overpasses per cell-day stays well below 5000.
#
# Output schema is identical to the station extractor's -- aod_047, aod_055,
# aod_qa, aod_uncertainty, date, image_id, location_id, raw unscaled integers --
# so the existing stages 2-5 (filter/scale, decode QA, apply QA filter, daily
# aggregate) consume it with no change. That matters: the model was fitted on
# AOD those stages produced, so the grid's AOD has to come through the same ones.
#
# Chunks are written to a parts directory and skipped if already present, so an
# 80-minute run survives a network failure and resumes instead of restarting.

BAND_TO_COLUMN = {
    "Optical_Depth_055": "aod_055",
    "Optical_Depth_047": "aod_047",
    "AOD_Uncertainty": "aod_uncertainty",
    "AOD_QA": "aod_qa",
}
OUTPUT_COLUMNS = ["aod_047", "aod_055", "aod_qa", "aod_uncertainty",
                  "date", "image_id", "location_id"]

MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 20


def month_ranges(study_start, study_end):
    # Inclusive start, exclusive end, one entry per calendar month.
    starts = pd.date_range(study_start, study_end, freq="MS")
    first = pd.Timestamp(study_start)
    if len(starts) == 0 or starts[0] > first:
        starts = pd.DatetimeIndex([first]).append(starts)
    ranges = []
    for index in range(len(starts)):
        begin = starts[index]
        if index + 1 < len(starts):
            finish = starts[index + 1]
        else:
            finish = pd.Timestamp(study_end) + pd.Timedelta(days=1)
        if begin >= finish:
            continue
        ranges.append((begin.strftime("%Y-%m-%d"), finish.strftime("%Y-%m-%d")))
    return ranges


def extract_chunk(collection_id, bands, cells, date_from, date_to):
    features = []
    for cell in cells.itertuples(index=False):
        features.append(ee.Feature(
            ee.Geometry.Point([float(cell.longitude), float(cell.latitude)]),
            {"location_id": int(cell.location_id)}))
    points = ee.FeatureCollection(features)

    collection = (ee.ImageCollection(collection_id)
                  .filterDate(date_from, date_to)
                  .filterBounds(points.geometry()))

    def tag_image(image):
        reduced = image.select(bands).reduceRegions(
            collection=points, reducer=ee.Reducer.first(), scale=1000)
        image_date = image.date().format("YYYY-MM-dd")
        image_id = image.get("system:index")
        return reduced.map(lambda f: f.set("date", image_date)
                                      .set("image_id", image_id))

    tagged = collection.map(tag_image).flatten()
    # Drop cells the granule did not actually cover, same as the station script.
    valid = tagged.filter(ee.Filter.neq("Optical_Depth_055", None))
    result = valid.getInfo()

    rows = []
    for feature in result["features"]:
        properties = feature["properties"]
        row = {"location_id": properties["location_id"],
               "date": properties["date"],
               "image_id": properties["image_id"]}
        for band in bands:
            row[BAND_TO_COLUMN[band]] = properties.get(band)
        rows.append(row)
    return rows


# Earth Engine's own message when a result collection exceeds 5000 elements.
# This failure is DETERMINISTIC -- the same request will always produce the same
# oversized result -- so retrying it is wasted time. It needs a smaller request.
LIMIT_MARKER = "accumulating over 5000 elements"


def extract_chunk_with_retry(collection_id, bands, cells, date_from, date_to,
                             depth=0):
    # Rows per chunk swing by an order of magnitude across the year: a 100-cell
    # block returned 4659 rows in March and 67 in July, because the monsoon
    # wipes out MAIAC retrievals while winter has near-daily coverage. A single
    # fixed chunk size therefore either overflows in winter or wastes requests
    # in the monsoon. So the size adapts: on the 5000-element error the cell
    # block is halved and each half retried, which keeps big efficient chunks
    # wherever they fit and splits only where they do not.
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return extract_chunk(collection_id, bands, cells, date_from, date_to)
        except Exception as error:
            last_error = error
            if LIMIT_MARKER in str(error):
                if len(cells) <= 1:
                    raise SystemExit(
                        f"ERROR: a single cell exceeds the 5000-element limit for "
                        f"{date_from}..{date_to}; this should be impossible and "
                        f"means something is wrong with the request, not its size")
                half = len(cells) // 2
                print(f"    {len(cells)} cells x {date_from[:7]} exceeds the 5000 "
                      f"row limit -- splitting into {half} + {len(cells) - half}")
                rows = extract_chunk_with_retry(collection_id, bands,
                                                cells.iloc[:half], date_from,
                                                date_to, depth + 1)
                rows.extend(extract_chunk_with_retry(collection_id, bands,
                                                     cells.iloc[half:], date_from,
                                                     date_to, depth + 1))
                return rows
            wait = BACKOFF_SECONDS * (attempt + 1)
            print(f"    attempt {attempt + 1} failed: {type(error).__name__} "
                  f"{str(error)[:90]} -- retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"ERROR: chunk failed after {MAX_ATTEMPTS} attempts: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--grid", required=True, help="grid CSV in station schema")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--parts_dir", required=True,
                        help="per-chunk CSVs, reused on resume")
    args = parser.parse_args()

    for path in [args.output, args.summary_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(args.parts_dir, exist_ok=True)

    with open(args.params) as f:
        all_params = yaml.safe_load(f)
    extract_params = all_params["maiac_extract"]
    grid_params = all_params["prediction"]["maiac_grid"]

    cells_per_chunk = grid_params["cells_per_chunk"]
    collection_id = extract_params["collection"]
    bands = extract_params["bands"]

    ee.Initialize(project=extract_params["gee_project"])

    grid = pd.read_csv(args.grid)
    grid = grid[grid["status"] == "KEEP"].reset_index(drop=True)
    print(f"Loaded {len(grid)} grid cells from {args.grid}")

    months = month_ranges(extract_params["study_start"], extract_params["study_end"])
    blocks = [grid.iloc[i:i + cells_per_chunk]
              for i in range(0, len(grid), cells_per_chunk)]
    print(f"{len(blocks)} cell blocks x {len(months)} months "
          f"= {len(blocks) * len(months)} chunks")

    started_all = time.time()
    n_done = 0
    n_skipped = 0
    total_chunks = len(blocks) * len(months)
    for block_index in range(len(blocks)):
        block = blocks[block_index]
        for month_index in range(len(months)):
            date_from, date_to = months[month_index]
            part_path = os.path.join(
                args.parts_dir, f"block{block_index:03d}_month{month_index:02d}.csv")
            if os.path.exists(part_path):
                n_skipped = n_skipped + 1
                continue
            started = time.time()
            rows = extract_chunk_with_retry(
                collection_id, bands, block, date_from, date_to)
            pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(part_path, index=False)
            n_done = n_done + 1
            elapsed = time.time() - started
            completed = n_done + n_skipped
            rate = (time.time() - started_all) / max(n_done, 1)
            remaining = (total_chunks - completed) * rate / 60.0
            print(f"  block {block_index + 1}/{len(blocks)} "
                  f"month {date_from[:7]}: {len(rows)} rows in {elapsed:.1f}s "
                  f"({completed}/{total_chunks}, ~{remaining:.0f} min left)")

    print(f"Chunks: {n_done} extracted, {n_skipped} reused from {args.parts_dir}")

    frames = []
    for name in sorted(os.listdir(args.parts_dir)):
        if name.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(args.parts_dir, name)))
    combined = pd.concat(frames, ignore_index=True)
    # A cell on a tile edge can appear in two granules in the same month; the
    # station pipeline keeps one row per (cell, date, image), so dedupe on that.
    before = len(combined)
    combined = combined.drop_duplicates(subset=["location_id", "date", "image_id"])
    print(f"Combined {before} rows, {before - len(combined)} duplicates dropped")

    combined = combined[OUTPUT_COLUMNS].sort_values(["location_id", "date"])
    combined.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(combined)} rows, "
          f"{combined['location_id'].nunique()} cells, "
          f"{combined['date'].nunique()} dates")

    summary = pd.DataFrame([{
        "n_cells": int(combined["location_id"].nunique()),
        "n_dates": int(combined["date"].nunique()),
        "n_rows": len(combined),
        "rows_per_cell_day": round(len(combined) /
                                   max(combined["location_id"].nunique() *
                                       combined["date"].nunique(), 1), 4),
        "n_chunks": total_chunks,
        "cells_per_chunk": cells_per_chunk,
        "minutes_total": round((time.time() - started_all) / 60.0, 1),
        "aod_055_min": float(combined["aod_055"].min()),
        "aod_055_max": float(combined["aod_055"].max()),
        "cells_with_no_data": int(len(grid) - combined["location_id"].nunique()),
    }])
    summary.to_csv(args.summary_output, index=False)
    print(f"Wrote {args.summary_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
