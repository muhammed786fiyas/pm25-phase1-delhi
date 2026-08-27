// ============================================================
// GEE Console Sanity Check — Static Layers Module
// Station: R K Puram, Delhi | location_id: 17
// ============================================================

var station = ee.Geometry.Point([77.186937, 28.563262]);
var buffer1km = station.buffer(1000);

Map.centerObject(station, 14);
Map.addLayer(buffer1km, {color: 'red'}, '1km buffer');

// ------------------------------------------------------------
// 1. ESA WorldCover v200 (2021) — land-use % fractions
// ------------------------------------------------------------
var worldcover = ee.Image('ESA/WorldCover/v200/2021').select('Map');

Map.addLayer(worldcover.clip(buffer1km), {}, 'WorldCover clipped');

var wcHist = worldcover.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: buffer1km,
  scale: 10,
  maxPixels: 1e9
});

print('WorldCover class pixel counts (raw):', wcHist);

// Convert counts to % fractions
var classCounts = ee.Dictionary(wcHist.get('Map'));
var totalPixels = classCounts.values().reduce(ee.Reducer.sum());
var classFractions = classCounts.map(function(key, value) {
  return ee.Number(value).divide(totalPixels).multiply(100);
});
print('WorldCover class % fractions (should sum to ~100):', classFractions);

// WorldCover class legend for reference while sanity checking:
// 10 Tree cover | 20 Shrubland | 30 Grassland | 40 Cropland
// 50 Built-up   | 60 Bare/sparse veg | 70 Snow/ice | 80 Water
// 90 Wetland herbaceous | 95 Mangroves | 100 Moss/lichen

// ------------------------------------------------------------
// 2. Sentinel-2 NDVI — nearest 5-day composite check
// ------------------------------------------------------------
var testDate = ee.Date('2024-11-15'); // pick any date in your study window
var windowDays = 5;

function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask);
}

var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(station)
  .filterDate(testDate.advance(-windowDays, 'day'), testDate.advance(windowDays, 'day'))
  .map(maskS2clouds);

print('Number of S2 images in +/-5 day window:', s2Collection.size());

var s2Composite = s2Collection.median();

var ndvi = s2Composite.normalizedDifference(['B8', 'B4']).rename('NDVI');

Map.addLayer(ndvi.clip(buffer1km), {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI');

var ndviValue = ndvi.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: station,
  scale: 10,
  maxPixels: 1e9
});

print('NDVI value at station point:', ndviValue);

// ------------------------------------------------------------
// 3. SRTM — elevation + slope
// ------------------------------------------------------------
var srtm = ee.Image('USGS/SRTMGL1_003');
var elevation = srtm.select('elevation');
var slope = ee.Terrain.slope(elevation);

Map.addLayer(elevation.clip(buffer1km), {min: 150, max: 300}, 'Elevation');

var terrainValues = elevation.addBands(slope).reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: station,
  scale: 30,
  maxPixels: 1e9
});

print('Elevation (m) and slope (deg) at station point:', terrainValues);