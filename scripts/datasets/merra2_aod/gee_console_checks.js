// ===== MERRA-2 Sanity Check Script =====
// Run this in the GEE Code Editor (code.earthengine.google.com), not via DVC/CLI.
// Collection: NASA/GSFC/MERRA/aer/2 (M2T1NXAER, hourly aerosol diagnostics)
// Band: TOTEXTTAU (total aerosol extinction AOT at 550nm)

var merra2 = ee.ImageCollection('NASA/GSFC/MERRA/aer/2');

// ----- 1. Confirm hourly images exist for a test day -----
var oneDay = merra2.filterDate('2025-06-01', '2025-06-02');
print('Images on 2025-06-01 (expect 24, one per hour):', oneDay.size());

// ----- 2. Grab one image, check band and native resolution -----
var firstImage = oneDay.first().select('TOTEXTTAU');
print('TOTEXTTAU image:', firstImage);
print('Native pixel size in meters:', firstImage.projection().nominalScale());

// ----- 3. Visualize over Delhi -----
Map.setCenter(77.2090, 28.6139, 7);
Map.addLayer(firstImage, {min: 0, max: 1, palette: ['blue', 'yellow', 'red']}, 'MERRA-2 TOTEXTTAU');

// ----- 4. Sample TOTEXTTAU at a Delhi test point -----
var delhiPoint = ee.Geometry.Point([77.2090, 28.6139]);
var valueAtPoint = firstImage.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: delhiPoint,
  scale: 1000
});
print('TOTEXTTAU at test point (2025-06-01):', valueAtPoint);

// ----- 5. Monsoon date check (expect a clear drop from the June value) -----
var monsoonDay = merra2.filterDate('2025-07-15', '2025-07-16');
var monsoonImage = monsoonDay.first().select('TOTEXTTAU');
var monsoonValue = monsoonImage.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: delhiPoint,
  scale: 1000
});
print('TOTEXTTAU on monsoon date (2025-07-15):', monsoonValue);

// ----- 6. Grid cell-center check -----
// NOTE: reduceRegion with scale:1000 here snaps pixelLonLat() to a 1km grid,
// NOT MERRA-2's real ~62km grid -- it will look plausible but be wrong.
// See 1-map_stations_to_cells.py for the correct version, which reprojects
// to MERRA-2's own projection instead of guessing a scale.
var merraProjection = firstImage.projection();
var pixelCenters = ee.Image.pixelLonLat().reproject(merraProjection);

var narela = ee.Geometry.Point([77.101981, 28.822836]);
var ayaNagar = ee.Geometry.Point([77.131606, 28.474261]);

var narelaCell = pixelCenters.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: narela,
  crs: merraProjection
});
print('Narela true cell center:', narelaCell);

var ayaNagarCell = pixelCenters.reduceRegion({
  reducer: ee.Reducer.first(),
  geometry: ayaNagar,
  crs: merraProjection
});
print('Aya Nagar true cell center:', ayaNagarCell);