// === Setup ===
var COLLECTION_ID = 'ECMWF/ERA5_LAND/HOURLY';
var TEST_POINT = ee.Geometry.Point(77.2090, 28.6139); // central Delhi

var era5land = ee.ImageCollection(COLLECTION_ID);

// === 1. Band names ===
var firstImage = era5land.first();
print('Band names:', firstImage.bandNames());
// Manually check this list for temperature_2m, dewpoint_temperature_2m,
// u_component_of_wind_10m, v_component_of_wind_10m
// and confirm nothing boundary-layer-related is present.

// === 2. Hourly image count for one day ===
var oneDay = era5land.filterDate('2025-06-15', '2025-06-16');
print('Images on 2025-06-15 (expect 24):', oneDay.size());

// === 3. Resolution check ===
var proj = firstImage.select('temperature_2m').projection();
print('Projection:', proj);
print('Nominal scale (meters, expect ~11132):', proj.nominalScale());

// === 4. Summer test date — Delhi, pre-monsoon (hot, dry) ===
var summerImage = era5land
  .filterDate('2025-06-15T07:00:00', '2025-06-15T08:00:00')
  .first();

var summerValues = summerImage.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: TEST_POINT,
  scale: 1000
});
print('Summer (2025-06-15, 07:00 UTC) values:', summerValues);

// === 5. Monsoon test date — Delhi, wet season ===
var monsoonImage = era5land
  .filterDate('2025-07-15T07:00:00', '2025-07-15T08:00:00')
  .first();

var monsoonValues = monsoonImage.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: TEST_POINT,
  scale: 1000
});
print('Monsoon (2025-07-15, 07:00 UTC) values:', monsoonValues);

// === 6. Manual checks to do on the printed values ===
// - temperature_2m should be ~300-315K in June, ~295-305K in July (Kelvin, not Celsius)
// - dewpoint_temperature_2m should be LOWER than temperature_2m in both cases
// - u_component_of_wind_10m and v_component_of_wind_10m should each be small,
//   roughly single-digit m/s (either sign is fine, it's direction)
