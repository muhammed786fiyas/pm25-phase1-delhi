import argparse
import os
import time
import urllib.request

import geopandas as gpd
import osmium
import pandas as pd
import shapely
import yaml

# Builds a PREDICTION-ONLY OpenStreetMap extract covering the prediction grid
# plus the 1km feature buffer, written to data/prediction/raw/osm/.
#
# Why separate from the training extract rather than extending it: the training
# exports were hand-drawn around the 42 stations and every station sits well
# inside them, so TRAINING HAS NO GAP and needs no fix. Only the grid reaches
# outside -- 440 of 2388 one-kilometre cells fall outside the training
# industrial export and would otherwise report a false zero. Extending the
# training files would modify a DVC-tracked training input, dirty every
# downstream training stage, and leave one file holding two OSM vintages, all to
# deliver something training never uses.
#
# The feature DEFINITIONS stay shared: scripts/datasets/osm/01..09 compute road
# density, industrial fraction and powerplant distance from whichever geojson
# they are pointed at. Only the source file differs. That is deliberate -- the
# model was fitted on features those exact scripts produced, so a second copy of
# the computation would drift and silently feed it a different quantity.
#
# SOURCE: a Geofabrik regional .osm.pbf, not the Overpass API.
# Overpass was tried first and is not usable for this. Measured 2026-10-08
# against overpass-api.de: a 5km box in dense Delhi returns 2721 ways / 1.63 MB
# in 7.8s, a 6km box returns 504 in 12s. Those 504s arrive far too fast to be
# the 120s query budget expiring, and `out ids;` on a failing tile succeeds and
# returns all 23180 ways -- so the gateway refuses large RESPONSES rather than
# timing out on computation. Tiling down to 4km made each request legal, but the
# failures were intermittent (a tile that 504'd inside a run succeeded on its own
# moments later) and across 182 tiles the projected runtime climbed 91 -> 158 ->
# 176 minutes as it ran. No mirror helps: kumi.systems and private.coffee return
# 500 on every attempt, osm.jp and maps.mail.ru fail TLS verification. Rate
# limiting was ruled out -- the status endpoint reported 4 free slots throughout.
#
# A pre-built extract avoids all of it: one 224 MB download, no rate limits, and
# -- the real win for reproducibility -- a single published timestamp, so the
# vintage is an exact documented fact rather than "whenever the 182 tiles ran".
#
# TWO zones are needed, not one. Geofabrik's "Northern Zone" does NOT include
# Uttar Pradesh, and that matters twice over:
#   - the power-plant bbox reaches lon 79.0, deep into UP. Northern Zone alone
#     found 112 plants against the training export's 170, missing all 57 east of
#     lon 77.36 -- including Dadri, which docs/logs/tasks/6-OSM_Features.md names
#     as one of the key out-of-Delhi plants the wide bbox exists to catch.
#   - Noida and Ghaziabad are in UP and inside the STUDY bbox. Northern Zone
#     alone returned only 76% of the training export's roads in the eastern
#     strip (lon 77.30-77.36), because the zone's clip boundary cuts through it.
# Northern (Delhi, Haryana, Rajasthan, Uttarakhand, Punjab, HP) plus Central
# (Uttar Pradesh, Madhya Pradesh, Chhattisgarh) covers the study bbox and the
# whole plant bbox. Each is small enough to index in memory on its own, which
# the 1.7 GB all-India extract would not reliably be.
#
# Tag selectors replicate docs/logs/tasks/6-OSM_Features.md exactly:
#   roads       highway=*            (every mapped path type, as before)
#   industrial  landuse=industrial
#   powerplants power=plant          (NOT power=generator -- generator nodes tag
#                                     rooftop diesel sets and would corrupt
#                                     "distance to nearest power plant")

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
USER_AGENT = "pm25-phase1-delhi/1.0 (academic research; PM2.5 satellite-ground calibration)"


def download_pbf(url, destination):
    if os.path.exists(destination):
        print(f"  reusing cached {destination} "
              f"({os.path.getsize(destination) / 1e6:.0f} MB)")
        return
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.time()
    written = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(destination, "wb") as f:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                f.write(chunk)
                written = written + len(chunk)
    print(f"  wrote {written / 1e6:.0f} MB of {total / 1e6:.0f} MB "
          f"in {time.time() - started:.0f}s")


def intersects(bounds, bbox):
    # bounds = (minx, miny, maxx, maxy) from shapely; bbox = (south, west, north, east)
    south, west, north, east = bbox
    return not (bounds[2] < west or bounds[0] > east
                or bounds[3] < south or bounds[1] > north)


class LayerHandler(osmium.SimpleHandler):
    # All three layers are collected in ONE pass. Reading a 224 MB regional
    # extract three times would triple the slowest step for no reason.
    def __init__(self, feature_bbox, plant_bbox):
        super().__init__()
        self.feature_bbox = feature_bbox
        self.plant_bbox = plant_bbox
        self.factory = osmium.geom.GeoJSONFactory()
        self.roads = []
        self.industrial = []
        self.plants = []
        self.n_skipped = 0

    def node(self, n):
        tags = n.tags
        if tags.get("power") == "plant":
            point = shapely.Point(n.location.lon, n.location.lat)
            if intersects(point.bounds, self.plant_bbox):
                self.plants.append(("node/%d" % n.id, tags.get("name"), point))
        if tags.get("landuse") == "industrial":
            point = shapely.Point(n.location.lon, n.location.lat)
            if intersects(point.bounds, self.feature_bbox):
                self.industrial.append(("node/%d" % n.id, tags.get("name"), point))

    def way(self, w):
        if "highway" not in w.tags:
            return
        try:
            geometry = shapely.from_geojson(self.factory.create_linestring(w))
        except Exception:
            self.n_skipped = self.n_skipped + 1
            return
        if intersects(geometry.bounds, self.feature_bbox):
            self.roads.append(("way/%d" % w.id, w.tags.get("name"), geometry))

    def area(self, a):
        # osmium assembles areas from closed ways AND multipolygon relations, so
        # this one callback covers both. The Overpass version had to rebuild
        # rings by hand and could not reconstruct relation holes; this can.
        tags = a.tags
        is_industrial = tags.get("landuse") == "industrial"
        is_plant = tags.get("power") == "plant"
        if not is_industrial and not is_plant:
            return
        try:
            geometry = shapely.from_geojson(self.factory.create_multipolygon(a))
        except Exception:
            self.n_skipped = self.n_skipped + 1
            return
        if geometry.is_empty:
            return
        if a.from_way():
            osm_id = "way/%d" % a.orig_id()
        else:
            osm_id = "relation/%d" % a.orig_id()
        if is_industrial and intersects(geometry.bounds, self.feature_bbox):
            self.industrial.append((osm_id, tags.get("name"), geometry))
        if is_plant and intersects(geometry.bounds, self.plant_bbox):
            # The training export holds power plants as 170 Points because its
            # Overpass Turbo query used `out center;`. Script 07 collapses any
            # geometry to a centroid anyway, but matching the training shape
            # keeps the two extracts directly comparable.
            self.plants.append((osm_id, tags.get("name"), geometry.centroid))


def to_geodataframe(records, layer_name):
    if len(records) == 0:
        raise SystemExit(f"ERROR: layer {layer_name} came back empty")
    frame = pd.DataFrame([{"id": r[0], "name": r[1]} for r in records])
    return gpd.GeoDataFrame(frame, geometry=[r[2] for r in records], crs="EPSG:4326")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--roads_output", required=True)
    parser.add_argument("--industrial_output", required=True)
    parser.add_argument("--powerplants_output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--pbf_cache", required=True,
                        help="directory holding the regional .osm.pbf between runs")
    args = parser.parse_args()

    for path in [args.roads_output, args.industrial_output,
                 args.powerplants_output, args.summary_output]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)["prediction"]["osm_export"]

    feature_bbox = (params["bbox_south"], params["bbox_west"],
                    params["bbox_north"], params["bbox_east"])
    plant_bbox = (params["powerplant_bbox_south"], params["powerplant_bbox_west"],
                  params["powerplant_bbox_north"], params["powerplant_bbox_east"])

    print("=== Source extracts ===")
    pbf_paths = []
    for source in params["pbf_sources"]:
        pbf_path = os.path.join(args.pbf_cache, source["filename"])
        download_pbf(source["url"], pbf_path)
        pbf_paths.append(pbf_path)

    print("=== Reading pbf (one pass per zone, all three layers) ===")
    print(f"  feature bbox (S,W,N,E) {feature_bbox}")
    print(f"  plant bbox   (S,W,N,E) {plant_bbox}")
    handler = LayerHandler(feature_bbox, plant_bbox)
    for pbf_path in pbf_paths:
        started = time.time()
        # locations=True builds way geometries from node positions; flex_mem
        # keeps the node index in memory, which one zone fits in comfortably.
        handler.apply_file(pbf_path, locations=True, idx="flex_mem")
        print(f"  {os.path.basename(pbf_path)}: read in {time.time() - started:.0f}s "
              f"(running totals: {len(handler.roads)} roads, "
              f"{len(handler.industrial)} industrial, {len(handler.plants)} plants)")
    print(f"  {handler.n_skipped} features skipped for unusable geometry")

    layers = [
        ("roads", handler.roads, args.roads_output),
        ("industrial", handler.industrial, args.industrial_output),
        ("powerplants", handler.plants, args.powerplants_output),
    ]

    summary_rows = []
    for name, records, output_path in layers:
        gdf = to_geodataframe(records, name)
        # A closed way tagged highway can arrive from both way() and area(), so
        # the same OSM id can appear twice; keep one.
        gdf = gdf.drop_duplicates(subset=["id"]).reset_index(drop=True)
        gdf.to_file(output_path, driver="GeoJSON")
        bounds = gdf.total_bounds
        counts = gdf.geometry.type.value_counts().to_dict()
        print(f"  {name}: {len(gdf)} features -> {output_path}")
        print(f"    geometry {counts}")
        print(f"    extent lon {bounds[0]:.4f}-{bounds[2]:.4f} "
              f"lat {bounds[1]:.4f}-{bounds[3]:.4f}")
        summary_rows.append({
            "layer": name,
            "source": ";".join(x["url"] for x in params["pbf_sources"]),
            "n_features": len(gdf),
            "geometry_types": str(counts),
            "min_lon": round(float(bounds[0]), 5),
            "min_lat": round(float(bounds[1]), 5),
            "max_lon": round(float(bounds[2]), 5),
            "max_lat": round(float(bounds[3]), 5),
            "extracted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    pd.DataFrame(summary_rows).to_csv(args.summary_output, index=False)
    print(f"Wrote {args.summary_output}")

    print("=== Coverage gate ===")
    # The entire point of this extract is that every grid cell's 1km buffer is
    # covered. A regional pbf reaches far past the bbox, so the extracted
    # features must span the required corners; if they do not, something
    # filtered wrongly and the false-zero bug is back.
    for row in summary_rows:
        if row["layer"] == "powerplants":
            continue
        covered = (row["min_lon"] <= feature_bbox[1]
                   and row["max_lon"] >= feature_bbox[3]
                   and row["min_lat"] <= feature_bbox[0]
                   and row["max_lat"] >= feature_bbox[2])
        print(f"  {row['layer']}: spans every required corner: {covered}")
        if not covered:
            print(f"    WARNING: {row['layer']} does not span the required bbox. "
                  f"For a sparse layer this can be legitimate (no feature near an "
                  f"edge); the grid QC stage is the authoritative gate.")


if __name__ == "__main__":
    main()
