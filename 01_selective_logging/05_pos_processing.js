
var asset = 'projects/ee-simex/assets/classification';

var assetUf = 'projects/mapbiomas-workspace/AUXILIAR/estados-2016'

var assetLulc = 'projects/mapbiomas-workspace/public/collection8/mapbiomas_collection80_integration_v1'







var roi = ee.FeatureCollection(assetUf).filter('NM_ESTADO == "PARأپ"')


var lulc = ee.Image(assetLulc).select('classification_2022').clip(roi.geometry());

var water = lulc.eq(33).distance(ee.Kernel.euclidean(5)).gt(0);

var forest = lulc.eq(3);


var freq = ee.ImageCollection(asset)
    .map(function(img){return img.gt(50)}).sum()
    .clip(roi.geometry())
    .mask(forest.eq(1)).where(water.eq(1), 0);
    



var vis = {
    'min': 0,
    'max': 16,
    'palette': [
      "#000000",
      "ffffcc",
      "ffeda0",
      "fed976",
      "feb24c",
      "fd8d3c",
      "fc4e2a",
      "e31a1c",
      "bd0026",
      "800026"
    ],
    //'format': 'png'
};

Map.addLayer(freq, vis, 'freq');









