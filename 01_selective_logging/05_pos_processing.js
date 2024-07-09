

var asset = 'projects/ee-simex/assets/logging_predictions';
var assetOutput = 'projects/ee-simex/assets/predictions'

var mb = ee.Image('projects/mapbiomas-workspace/public/collection8/mapbiomas_collection80_integration_v1')
    mb = mb.select('classification_2020')
    mb = mb.eq(3).or(mb.eq(6))

var collection = ee.ImageCollection(asset)

var collectionLoaded = ee.ImageCollection(assetOutput) 

var image = collection.reduce(ee.Reducer.sum());


var idsLoaded = collectionLoaded.reduceColumns(ee.Reducer.toList(), ['image_id']).get('list').getInfo();

var ids = collection
    .filter(ee.Filter.inList('image_id', idsLoaded).not())
    .reduceColumns(ee.Reducer.toList(), ['image_id'])
    .get('list').getInfo();






var high = image.gt(3);

var kernel = ee.Kernel.square({radius: 10});
  
var buffer = high
             .focalMax({kernel: kernel, iterations: 2})
             .reproject({scale:30, crs:'epsg:4326'})
          
var binaryMask = buffer.unmask(0);


ids.forEach(function(id) {
  
  var image = ee.Image(collection.filter('image_id == "' + id +'"').first())
  
  
  var fixImage = image.where(binaryMask.eq(0), 0);
      fixImage = fixImage.mask(mb).selfMask()
      fixImage = fixImage.copyProperties(image);
      
      fixImage = fixImage.set('system:time_end', image.get('system:time_end'));
      fixImage = fixImage.set('system:time_start', image.get('system:time_start'));
      
  Export.image.toAsset({
    image: fixImage,
    description: id + '_1',
    assetId: assetOutput + '/'+ id + '_1',
    pyramidingPolicy: {'.default': 'mode'},
    region: image.geometry(),
    scale:10,
    maxPixels:1e13
  });
  
})


image = image.where(binaryMask.eq(0), 0);



