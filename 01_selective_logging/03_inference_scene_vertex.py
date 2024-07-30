
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))


import ee

from pprint import pprint
from utils.metrics import *
from utils.index import *

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='sad-deep-learning-274812')



'''

    Config

'''

ASSET_IMAGE_T0 = ''
ASSET_IMAGE_T1 = ''

PROJECT = '1012853489050'

REGION = 'us-central1'

ENDPOINT_ID = '2920195131233533952'

BANDS = []




'''

    Get Image

'''

imaget0 = ee.Image(ASSET_IMAGE_T0).select(BANDS)
imaget1 = ee.Image(ASSET_IMAGE_T1).select(BANDS)

'''

    Normalize

'''

def normalize(image):

    min_dict = image.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=image.geometry(),
        scale=10,
        maxPixels=1e9
    )
    min_value = ee.Image.constant(min_dict.values().get(0))
    

    max_dict = image.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=image.geometry(),
        scale=10,
        maxPixels=1e9
    )

    max_value = ee.Image.constant(max_dict.values().get(0))
    

    normalized = image.subtract(min_value).divide(max_value.subtract(min_value))
    
    return normalized

imaget0 = normalize(imaget0).rename('R_0', 'G_0', 'B_0')
imaget1 = normalize(imaget1).rename('R_1', 'G_1', 'B_1')

image = imaget0.addBands(imaget1)


'''

    Run

'''

endpoint_path = (
    'projects/' + PROJECT + '/locations/' + REGION + '/endpoints/' + str(ENDPOINT_ID)
)

# connect to the hosted model.
vertex_model = ee.Model.fromVertexAi(**{
  'endpoint': endpoint_path,
  'inputTileSize': [64, 64],
  'inputOverlapSize': [32, 32],
  'proj': ee.Projection('EPSG:4326').atScale(10),
  'fixInputProj': True,
  'outputBands': {'output': {
      'type': ee.PixelType.float(),
      'dimensions': 1
    }
  }
})

predictions = vertex_model.predictImage(image.float())
labels = predictions.arrayArgmax().arrayGet(0).rename('label')