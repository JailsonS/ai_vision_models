
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

from pprint import pprint

import numpy as np
import tensorflow as tf
import ee, io, rasterio, keras
import concurrent

from tensorflow.keras import backend as  K

from utils.metrics import *
from models.UnetDefault import Unet

from rasterio.merge import merge
from numpy.lib.recfunctions import structured_to_unstructured
from retry import retry
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

APPLY_BRIGHT = True

TILES = {
    '22MGA': [
        '20220724T133851_20220724T134243_T22MGA'
    ]
}

KERNEL_SIZE = 512

NUM_CLASSES = 1

MODEL_PATH = '01_selective_logging/model/model_v4.keras'

OUTPUT_CHIPS = '01_selective_logging/predictions/{}/{}_{}.tif'
OUTPUT_TILE = '01_selective_logging/predictions'

'''

    Request Template

'''

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=30)


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


BAND_NAMES = [
    'B2', 'B3', 'B4',
    #'B8', 'B11'
]

NEW_BAND_NAMES = [
    'red','green', 'blue',
    #'nir','swir1'
]

BANDS = [
    'red_t0','green_t0', 'blue_t0',#'nir_t0','swir_t0',
    'red_t1','green_t1', 'blue_t1',#'nir_t1','swir1_t1'
]



'''

    Functions

'''

def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized_array = (array - min_value) / (max_value - min_value)
    return normalized_array


def apply_brightness(array):
    brightness_factor = 1.5
    # increase brightness by multiplying
    brightened = np.clip(array * brightness_factor, 0, 1)
    return brightened

def get_image(items):

    coords = items[1][1]

    t1_band_names = [x + '_t1' for x in NEW_BAND_NAMES]
    t0_band_names = [x + '_t0' for x in NEW_BAND_NAMES]

    col = ee.ImageCollection(ASSET_COLLECTION).filter(f'system:index == "{items[1][0]}"')

    # t1
    image_t1 = ee.Image(col.first())\
        .select(BAND_NAMES, NEW_BAND_NAMES).rename(t1_band_names)
    

    # get relative dates for t0
    tmp_date = ee.Date(image_t1.get('system:time_start')).advance(-12, 'months')
    t0 = ee.Date.fromYMD(tmp_date.get('year'), 6, 1)
    t0_ = ee.Date.fromYMD(tmp_date.get('year'), 10, 30)


    col_t0 = ee.ImageCollection(ASSET_COLLECTION)\
        .filterDate(t0, t0_)\
        .filterBounds(ee.Geometry.Point(coords))\
        .filter('CLOUDY_PIXEL_PERCENTAGE < 30')
    
    # t0
    image_t0 = ee.Image(col_t0.median())\
        .select(BAND_NAMES, NEW_BAND_NAMES).rename(t0_band_names)
    

    image = image_t0.addBands(image_t1)

    return image


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

    response = {'error': '', 'item':items}
    
    coords = items[1][1]

    image = get_image(items)

    request = dict(REQUEST)
    request['expression'] = image
    request['grid']['affineTransform']['translateX'] = coords[0] + OFFSET_X
    request['grid']['affineTransform']['translateY'] = coords[1] + OFFSET_Y

    affine = (
        request['grid']['affineTransform']['scaleX'],
        request['grid']['affineTransform']['shearX'],
        request['grid']['affineTransform']['translateX'],
        request['grid']['affineTransform']['scaleY'],
        request['grid']['affineTransform']['shearY'],
        request['grid']['affineTransform']['translateY'],
    )

    # for georeference convertion
    response['affine'] = affine
    
    try:
        data = np.load(io.BytesIO(ee.data.computePixels(request)))
    except ee.ee_exception.EEException as e:
        response['error']= e
        return None, response
    return data, response, items[0]


def predict(items):

    future_to_point = {EXECUTOR.submit(get_patch, item): item for item in items}

    for future in concurrent.futures.as_completed(future_to_point):
        
        data, response, idx = future.result()

        if data is not None:

            data = structured_to_unstructured(data)

            data_norma = np.stack([normalize_array(data[:,:,x]) for x in range(0, len(BANDS))])


            if APPLY_BRIGHT: 
                data_norma = apply_brightness(data_norma)

            data_transposed = np.transpose(data_norma, (1,2,0))
            data_transposed = np.expand_dims(data_transposed, axis=0)


            # add exception here
            # for prediction its necessary to have an extra dimension representing the bach
            probabilities = model.predict(data_transposed)


            # it only checks the supposed prediction and skip it if there is no logging
            prediction = np.copy(probabilities)
            prediction[prediction < 0.01] = 0
            prediction[prediction >= 0.01] = 1

            if np.max(prediction[0]) == 0.0:
                continue


            probabilities = probabilities[0]
            probabilities = np.transpose(probabilities, (2,0,1))

            with rasterio.open(
                OUTPUT_CHIPS.format(k, response['item'][1][0], idx),
                'w',
                driver = 'GTiff',
                count = NUM_CLASSES,
                height = probabilities.shape[1],
                width  = probabilities.shape[2],
                dtype  = probabilities.dtype,
                crs    = rasterio.crs.CRS.from_epsg(4326),
                transform=rasterio.transform.from_origin(response['item'][1][1][0] + OFFSET_X,
                                                         response['item'][1][1][1] + OFFSET_Y,
                                                         SCALE_X,
                                                         SCALE_Y)
            ) as output:
                output.write(probabilities)



def flatten_extend(matrix):
    flat_list = []
    for row in matrix: flat_list.extend(row)
    return flat_list



'''

    Implementation

'''


#outputs = smlayer(tf.keras.layers.Input(shape=[None, None, len(BANDS)]))

#model = tf.keras.Model(inputs=tf.keras.layers.Input(shape=[None, None, len(BANDS)]), outputs=outputs)
#model.compile(optimizer='adam', loss=soft_dice_loss)


model = keras.saving.load_model(MODEL_PATH, compile=False)



model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), 
    loss=soft_dice_loss, 
    metrics=[
        running_recall, 
        running_f1, 
        running_precision, 
        tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

for k, v in TILES.items():

    if not os.path.isdir(f'01_selective_logging/predictions/{k}'):
        os.mkdir(os.path.abspath(f'01_selective_logging/predictions/{k}'))


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

    items = enumerate(items)


    # run predictions
    predict(items)

    # create mosaic 
    # path_chips = [rasterio.open(x) for x in glob(f'{OUTPUT_TILE}/{k}/*')]

    # mosaic, out_trans = merge(path_chips)

    #with rasterio.open(
    #    f'{OUTPUT_TILE}/{k}_pred.tif',
    #    'w',
    #    driver = 'GTiff',
    #    count = NUM_CLASSES,
    #    height = mosaic.shape[1],
    #    width  = mosaic.shape[2],
    #    dtype  = mosaic.dtype,
    #    crs    = rasterio.crs.CRS.from_epsg(4326),
    #    transform = out_trans 
    #) as dest:
    #    dest.write(mosaic)
#
    ## delete files
    #for f in glob(f'{OUTPUT_TILE}/{k}/*.tif'):
    #    os.remove(f)






