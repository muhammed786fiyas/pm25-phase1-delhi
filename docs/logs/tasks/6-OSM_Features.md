# Task Log: OSM Features
_Last updated: 2026-08-28_

## Scope
Non-GEE static covariates sourced from OpenStreetMap for Delhi Phase 1: road density, industrial land-use fraction, distance to nearest power plant. Follows the Static GEE Layers module. Stretch features (bus stop density, fuel station density) explicitly deferred to a later session.

## Completed

**Data source (all three features)** -- 2026-08-28
- Manual Overpass Turbo exports (GeoJSON), not a live API call -- see Data notes below for why.
- Roads + industrial land-use: bbox (28.40,76.91,28.84,77.34) -- tight, around the 42-station cluster.
- Power plants: bbox (27.0,75.5,30.0,79.0) -- wide, ~110-160km margin around the station cluster so nearby out-of-Delhi plants (Dadri, Jhajjar, Panipat) aren't missed.

**Road density (km per km²)** -- 2026-08-28
- Script 01 (`01_extract_road_lengths.py`): loads the 42 KEEP stations and the roads GeoJSON (`highway=*` ways), reprojects to UTM 43N, clips each station's 1km-radius buffer against the road network, and sums the clipped segment lengths per station. Writes raw length (m) + segment count per station, plus an extraction summary (min/max/median length, failed stations).
- Script 02 (`02_qc_roads.py`): QC gate -- missing-station check against the KEEP list, zero-length flag, and an implausible-density flag (`max_plausible_density_km_per_km2` from params). Hard-fails (`SystemExit`) on any missing station.
- Script 03 (`03_compute_road_density.py`): converts each station's raw length into road density (km / km², dividing by the 1km buffer's circular area) and writes the final `road_density_km_per_km2` to processed.
- Result: 42/42 stations, 0 hard fails, 0 zero-length, 0 implausible-density flags. Density range 5.6-41.4 km/km², matching Delhi's urban-density gradient (Najafgarh/Alipur lowest, Patparganj/Jahangirpuri highest).

**Industrial land-use fraction** -- 2026-08-28
- Script 04 (`04_extract_industrial_area.py`): loads the industrial land-use GeoJSON (`landuse=industrial` polygons/relations), reprojects to UTM 43N, fixes invalid geometries (`buffer(0)`), clips each station's 1km buffer against the polygons, and sums the clipped area (m²) per station alongside the buffer's own area.
- Script 05 (`05_qc_industrial.py`): QC gate -- missing-station check, and a geometry-bug check (industrial area exceeding buffer area, which would mean a bad clip). Zero-industrial stations are logged but not treated as an error (expected for residential-area monitors).
- Script 06 (`06_compute_industrial_fraction.py`): divides industrial area by buffer area per station to produce the final `industrial_landuse_fraction`, written to processed.
- Result: 42/42 stations, 0 hard fails, 0 geometry-area bugs, 19/42 zero-industrial (expected). Fractions match known Delhi industrial estates (Okhla Phase-2 52.3%, Narela 46.7%, Wazirpur 27.2%, Bawana 28.1%).

**Distance to nearest power plant** -- 2026-08-28
- Script 07 (`07_extract_powerplant_distances.py`): loads the power plants GeoJSON (`power=plant` nodes/ways/relations), collapses every plant geometry to its centroid, and computes the straight-line distance from each station to its single nearest plant (plus that plant's name, when tagged).
- Script 08 (`08_qc_powerplants.py`): QC gate -- missing-station check, and an edge-effect check (flags a station whose nearest distance exceeds a threshold close to the download bbox's own margin, meaning a closer real-world plant might sit outside the downloaded area).
- Script 09 (`09_finalize_powerplant_distance.py`): straight-copy finalize step (mirrors SRTM script 10's pattern) -- converts the QC-passed raw distance from meters to km and writes the final `dist_to_nearest_powerplant_km` + nearest plant name to processed.
- Result: 42/42 stations, 0 hard fails, 0 edge-effect flags. Distances range 0.17-8.2km, all well under the 110km edge-effect threshold.

**Pipeline wiring**: all 9 scripts wired into `dvc.yaml` (9 new stages, 46 total) and `params.yaml` (`osm_layers.{roads,industrial,powerplants}`), following the extract -> QC -> compute/finalize pattern from static_gee. Tagged `delhi-phase1-v16`.

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
