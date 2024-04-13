
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

import numpy as np
import tensorflow as tf
import ee, io

from tensorflow.keras import backend as  K

from utils.metrics import *
from models.UnetDefault import Unet


from numpy.lib.recfunctions import structured_to_unstructured
from retry import retry


ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='sad-deep-learning-274812')



'''

    Assets

'''



ASSET_COLLECTION = 'COPERNICUS/S2_HARMONIZED'
#ASSET_TILES = 'projects/imazon-simex/MAPASREFERENCIA/GRID_SENTINEL'
ASSET_TILES = 'projects/mapbiomas-workspace/AUXILIAR/SENTINEL2/grid_sentinel'



'''
    Config Info
'''

TILES = {
    '21LXH': []
}


BANDS = [
    'red_t0','green_t0', 'blue_t0','nir_t0','swir_t0'
    'red_t1','green_t1', 'blue_t1','nir_t1','swir1_t1'
]

TARGET_BANDS = [0,1,2,5,6,7]

KERNEL_SIZE = 512

NUM_CLASSES = 1

HYPER_PARAMS = {
    'optimizer': tf.keras.optimizers.Nadam(learning_rate=0.001),
    'loss': 'MeanSquaredError',
    'metrics': ['RootMeanSquaredError']
}

MODEL_NAME = 'model/model_v3'




'''

    Request Template

'''



# image resolution in meters
SCALE = 10

# pre-compute a geographic coordinate system.
proj = ee.Projection('EPSG:4326').atScale(SCALE).getInfo()

# get scales in degrees out of the transform.
SCALE_X = proj['transform'][0]
SCALE_Y = -proj['transform'][4]

# patch size in pixels.
PATCH_SIZE = 512

# offset to the upper left corner.
OFFSET_X = -SCALE_X * PATCH_SIZE / 2
OFFSET_Y = -SCALE_Y * PATCH_SIZE / 2


# request template.
REQUEST = {
      'fileFormat': 'NPY',
      'grid': {
          'dimensions': {
              'width': PATCH_SIZE,
              'height': PATCH_SIZE
          },
          'affineTransform': {
              'scaleX': SCALE_X,
              'shearX': 0,
              'shearY': 0,
              'scaleY': SCALE_Y,
          },
          'crsCode': proj['crs']
      }
  }


BAND_NAMES = ['R', 'G', 'B', 'N']
NEW_BAND_NAMES = ['red', 'green', 'blue', 'nir']


'''

    Functions

'''

def serialize(data: np.ndarray) -> bytes:
    features = {
        'probabilities': tf.train.Feature(
            float_list=tf.train.FloatList(value=[])
        )
    }

    example = tf.train.Example(
        features=tf.train.Features(
            feature=features['probabilities'].float_list.value.extend(data)
        )
    )
    
    return example.SerializeToString()


@retry()
def get_patch(items):

    """Get a patch centered on the coordinates, as a numpy array."""

    coords = items[1]
    
    image = ee.Image(
        ee.ImageCollection(ASSET_COLLECTION).filter(f'system_index == "{items[0]}"').first()
    )

    request = dict(REQUEST)
    request['expression'] = image
    request['grid']['affineTransform']['translateX'] = coords[0] + OFFSET_X
    request['grid']['affineTransform']['translateY'] = coords[1] + OFFSET_Y

    return np.load(io.BytesIO(ee.data.computePixels(request)))


def flatten_extend(matrix):
    flat_list = []
    for row in matrix: flat_list.extend(row)
    return flat_list

'''

    Implementation

'''

for k, v in TILES.items():


    grid = ee.FeatureCollection(ASSET_TILES).filter(f'NAME == "{k}"')
    grid_feat = ee.Feature(grid.first()).set('id', 1)
    grid_img = ee.FeatureCollection([grid_feat]).reduceToImage(['id'], ee.Reducer.first())


    # get centroids
    seeds = grid_img.sample(region=grid_feat.geometry(), scale=10 * PATCH_SIZE, geometries=True)
    coords = seeds.reduceColumns(ee.Reducer.toList(), ['.geo']).get('list').getInfo()
    coords = [x['coordinates'] for x in coords]


    # if not specified get all scenes from grid
    if len(v) == 0:
        col = ee.ImageCollection(ASSET_COLLECTION).filter(f'MGRS_TILE == "{k}"')
        v = col.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()
    
    items = [list(zip([x] * len(coords), coords))
                for x in v]

    items = flatten_extend(items)
    









