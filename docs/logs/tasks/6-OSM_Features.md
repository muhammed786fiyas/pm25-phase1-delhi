# Task Log: OSM Features
_Last updated: 2026-08-28_

## Scope
Non-GEE static covariates sourced from OpenStreetMap for Delhi Phase 1: road density, industrial land-use fraction, distance to nearest power plant. Follows the Static GEE Layers module. Stretch features (bus stop density, fuel station density) explicitly deferred to a later session.

## Completed

**All three features -- 2026-08-28**
- Data source: manual Overpass Turbo exports (GeoJSON), not a live API call -- see "Network constraint" below.
- Roads: bbox (28.40,76.91,28.84,77.34) -- tight, around the 42-station cluster.
- Industrial land-use: same tight bbox.
- Power plants: bbox (27.0,75.5,30.0,79.0) -- wide, covers ~110-160km margin around the station cluster so nearby out-of-Delhi plants (Dadri, Jhajjar, Panipat) aren't missed.
- Pipeline: `scripts/datasets/osm/01-09`, extract -> QC -> compute/finalize per feature, wired into `dvc.yaml` (9 new stages, 46 total) and `params.yaml` (`osm_layers.{roads,industrial,powerplants}`).
- Result: 42/42 KEEP stations, 0 hard fails, 0 QC flags across all three features.
- Tagged `delhi-phase1-v16`.

## Key decisions

- **Buffer radius: 1km for roads and industrial land-use** (2026-08-28) -- consistent with the static_gee convention (WorldCover/NDVI/SRTM), matches MAIAC's native 1km grid.
- **Power plant tag: `power=plant` only, not `power=generator`** (2026-08-28) -- generator nodes tag small distributed/backup generation everywhere (rooftop diesel gensets etc.) and would corrupt "distance to nearest power plant" as an emissions point-source proxy.
- **Split bbox by feature sparsity, not one bbox for everything** (2026-08-28) -- roads/industrial need a tight bbox around the stations (dense data; a wide bbox overloaded the Overpass server on the first attempt -- see gotcha below). Power plants need a wide bbox (sparse data, cheap regardless of area, and the true nearest plant to a Delhi station is often outside Delhi).
- **Manual Overpass Turbo export instead of a live API call** (2026-08-28) -- both the cloud sandbox's and the device shell's network are allowlisted and block Overpass/Nominatim/openstreetmap.org entirely (only package registries pass). The user ran the queries in their own browser and dropped the GeoJSON exports into `data/raw/osm/`.
- **`dist_to_nearest_powerplant_km` computed to plant centroid, not footprint edge** (2026-08-28) -- power plants can be mapped as points, ways, or relations in OSM; collapsing everything to a centroid keeps the "nearest" comparison consistent regardless of how a given plant happens to be mapped.
- **Edge-effect QC gate at 110km** (2026-08-28) -- set below the ~130-160km margin between the station cluster and the download bbox edge, so a flag would mean "possibly missing a closer real-world plant outside the download area." No stations triggered it (max observed distance: 8.2km).

## Data notes & gotchas

- **First roads query (full power-plant-search bbox, `way["highway"]` over ~3 states) failed with an Overpass "Ajax Error" / parsererror** -- the server couldn't handle the data volume. Fixed by using a much smaller bbox for roads/industrial (just around the station cluster, ~2km padding beyond the outermost stations' 1km buffers) since that's all road density and industrial fraction actually need.
- **Road density (5.6-41.4 km/km² across stations) is higher than naive intuition but not a bug** -- OSM's `highway=*` tag covers every mapped path type (motorways down to footways, service roads, parking-lot aisles), and Delhi's colonies are mapped in fine detail. All values sit under the QC ceiling (50 km/km²) and the spatial pattern is sensible: lowest at peripheral stations (Najafgarh 6.1, Alipur 8.1), highest at dense urban ones (Patparganj 41.4, Jahangirpuri 36.3).
- **Industrial fraction lines up with known Delhi industrial geography**: Okhla Phase-2 (52.3%), Narela (46.7%), Wazirpur (27.2%), Bawana (28.1%) are all recognized industrial estates. 19/42 stations show exactly 0%, expected for residential-area monitors.
- **`power=plant` conflates facility scales.** Several stations' nearest "power plant" came back as a small biogas unit or solar farm rather than a grid-scale thermal/CCGT station -- the tag doesn't encode capacity. Genuine combined-cycle gas stations did show up too (Pragati-III Combined Cycle Power Plant, Rithala Power Plant, Indraprastha Gas Turbine Power Station, BSES). This means the raw "distance to nearest power plant" feature may sometimes reflect a small distributed-generation unit instead of a real point-source emitter. Flagged as a measurement-validity caveat worth a line in the eventual manuscript -- not fixed automatically, since resolving it would require a judgment call (e.g. filtering by plant capacity/type tags, which many OSM entries don't even have) rather than a QC-level correction.
- **No station triggered the edge-effect QC flag** -- max observed nearest-plant distance was 8.2km, far under the 110km threshold, confirming the wide bbox had ample margin.
- **Discovered (not caused by this module) a repo-wide CRLF/LF line-ending drift** across ~80 pre-existing files, unrelated to OSM work -- see DAY7 daily log for detail. Left untouched.

## Pending
- Bus stop density, fuel station density (stretch features) -- explicitly deferred by the user; not started, no scripts/params scaffolding added yet.
- If the study area later expands beyond Delhi Phase 1, the Overpass Turbo bboxes used here will need to be redrawn to match the new station set.

## Ideas / under consideration
- None raised this session.
