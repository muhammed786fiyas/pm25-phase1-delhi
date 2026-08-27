// === Setup ===
var COLLECTION_ID = 'ECMWF/ERA5/HOURLY';
var TEST_POINT = ee.Geometry.Point(77.2090, 28.6139); // central Delhi

var era5 = ee.ImageCollection(COLLECTION_ID);

// === 1. Band names ===
var firstImage = era5.first();
print('Band names:', firstImage.bandNames());
// Manually check this list contains 'boundary_layer_height'

// === 2. Hourly image count for one day ===
var oneDay = era5.filterDate('2025-06-15', '2025-06-16');
print('Images on 2025-06-15 (expect 24):', oneDay.size());

// === 3. Resolution check ===
var proj = firstImage.select('boundary_layer_height').projection();
print('Projection:', proj);
print('Nominal scale (meters, expect ~27830-31000):', proj.nominalScale());

// === 4. Summer test date - Delhi, pre-monsoon (hot, dry) ===
var summerImage = era5
  .filterDate('2025-06-15T07:00:00', '2025-06-15T08:00:00')
  .first();

var summerValue = summerImage.select('boundary_layer_height').reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: TEST_POINT,
  scale: 30000
});
print('Summer BLH (2025-06-15, 07:00 UTC):', summerValue);

// === 5. Monsoon test date - Delhi, wet season ===
var monsoonImage = era5
  .filterDate('2025-07-15T07:00:00', '2025-07-15T08:00:00')
  .first();

var monsoonValue = monsoonImage.select('boundary_layer_height').reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: TEST_POINT,
  scale: 30000
});
print('Monsoon BLH (2025-07-15, 07:00 UTC):', monsoonValue);

// === 6. Manual checks to do on the printed values ===
// - bandNames() should contain 'boundary_layer_height'
// - image count should be 24
// - nominal scale should be ~27830m (matches ERA5's ~31km grid)
// - summer BLH should be HIGHER than monsoon BLH (stronger convective
//   mixing in hot/dry conditions vs. cloud-suppressed monsoon mixing)
// - both values should be physically plausible: roughly hundreds to a
//   few thousand meters, not near-zero and not absurdly large