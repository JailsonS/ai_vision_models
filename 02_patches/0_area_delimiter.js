

// import libs
var palettes = require('users/gena/packages:palettes')
var palettesMb = require('users/mapbiomas/modules:Palettes.js').get('classification8');



// assets

var assetClsBr = 'projects/mapbiomas-workspace/public/collection8/mapbiomas_collection80_integration_v1';
var assetClsOthers = 'projects/mapbiomas-raisg/public/collection5/mapbiomas_raisg_panamazonia_collection5_integration_v1';
var assetRoi = 'WWF/HydroATLAS/v1/Basins/level03'




// config

var years = [
    //1985, 1986, 1987, 1988, 1989, 1990,
    //1991, 1992, 1993, 1994, 1995, 1996,
    //1997, 1998, 1999, 2000, 2001, 2002,
    //2003, 2004, 2005, 2006, 2007, 2008,
    //2009, 2010, 2011, 2012, 2013, 2014,
    //2015, 2016, 2017, 2018, 2019, 2020,
    1995, 2000, 2022, 
    // 2023
]






// input data

var roi = ee.FeatureCollection(assetRoi)
var collectionBr = ee.Image(assetClsBr);
var collectionRaisg = ee.Image(assetClsOthers);

Map.addLayer(roi)



// vis params

var visLulc = {
  palette:palettesMb,
  min:0,max:62
};



// iterate years

var mask = ee.Image(1).clip(roi)

years.forEach(function(year) {
  
  
    var classificationRaisg = collectionRaisg.select('classification_' + String(year));
    
    
    var forestCls = classificationRaisg.eq(3).or(classificationRaisg.eq(6)).mask(mask);
    
    
    
    var objId = forestCls.connectedComponents({
        connectedness: ee.Kernel.plus(5),
        maxSize: 256
    });
    
      
    var objectSize = objId.select('labels').connectedPixelCount({
        maxSize: 256, eightConnected: false
    });
    
    var pixelArea = ee.Image.pixelArea().divide(10000);
    
    
    var objectArea = objectSize.multiply(pixelArea);
    
    var maskPatches = objectArea.lte(1).unmask(0).reproject({crs:'epsg:4326', scale:30});
    
    
    forestCls = forestCls.mask(maskPatches.eq(0)).mask(mask)
      

    //Map.addLayer(classificationRaisg, visLulc, 'map raisg: ' + String(year), false);
    Map.addLayer(forestCls.selfMask(), {palette:['#22ff42']}, 'forest: ' + String(year), false);
    
    
    
    
    // export examples
    
    var name = 'examples_' + String(year);
    
    Export.image.toDrive({
      image: forestCls, 
      description: name, 
      folder:'FRAGMENTACAO', 
      fileNamePrefix:name,
      region: mask.geometry(), 
      scale:30,
      fileFormat:'GeoTIFF', priority:1000})
  
});

















