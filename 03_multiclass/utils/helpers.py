import ee

def remove_cloud_s2(collection: ee.imagecollection.ImageCollection) -> ee.imagecollection.ImageCollection:

    CLEAR_THRESHOLD = 0.60

    cloud_prob = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')    

    colFreeCloud = collection.linkCollection(cloud_prob, ['cs'])\
        .map(lambda image: 
            image.updateMask(image.select('cs').gte(CLEAR_THRESHOLD))
                    .copyProperties(image)
                    .copyProperties(image, ['system:footprint'])
                    .copyProperties(image, ['system:time_start'])
        )
    
    return colFreeCloud
