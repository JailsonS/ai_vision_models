import ee

# PROJECT = 'ee-mapbiomas-imazon'
PROJECT = 'ee-simex'

ee.Authenticate()
ee.Initialize(project=PROJECT)

asset = 'projects/ee-simex/assets/logging_predictions'
assetOutput = 'projects/ee-simex/assets/predictions'

mb = ee.Image('projects/mapbiomas-workspace/public/collection8/mapbiomas_collection80_integration_v1')
mb = mb.select('classification_2020')
mb = ee.Image(mb.eq(3).Or(mb.eq(6)))

collection = ee.ImageCollection(asset)
collectionLoaded = ee.ImageCollection(assetOutput)
image = collection.reduce(ee.Reducer.sum())

idsLoaded = collectionLoaded.reduceColumns(ee.Reducer.toList(), ['image_id']).get('list').getInfo()
ids = collection.filter(ee.Filter.inList('image_id', idsLoaded).Not()) \
    .reduceColumns(ee.Reducer.toList(), ['image_id']) \
    .get('list').getInfo()

high = image.gt(3)
kernel = ee.Kernel.square(radius=10)
buffer = high.focalMax(kernel=kernel, iterations=2).reproject(crs='EPSG:4326', scale=30)
binaryMask = buffer.unmask(0)

for id in ids:
    image = ee.Image(collection.filter(ee.Filter.eq('image_id', id)).first())
    fixImage = ee.Image(image.where(binaryMask.eq(0), 0))
    fixImage = ee.Image(fixImage.mask(mb.eq(1)).selfMask())
    fixImage = ee.Image(fixImage.copyProperties(image))
    fixImage = fixImage.set('system:time_end', image.get('system:time_end'))
    fixImage = fixImage.set('system:time_start', image.get('system:time_start'))

    print(f'processing {id}')

    task = ee.batch.Export.image.toAsset(
        image=fixImage,
        description=id,
        assetId=assetOutput + '/' + id.replace('.tif','') + '_1',
        pyramidingPolicy={'.default': 'mode'},
        region=image.geometry(),
        scale=10,
        maxPixels=1e13
    )
    task.start()
