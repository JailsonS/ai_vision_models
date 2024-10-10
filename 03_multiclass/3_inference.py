
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

import numpy as np
import os
import ee, io
import concurrent
import tensorflow as tf
import keras, rasterio


from retry import retry
from numpy.lib.recfunctions import structured_to_unstructured
from utils.helpers import *
from utils.index import *
from utils.metrics import *
from pprint import pprint
from rasterio.transform import Affine

PROJECT = 'ee-mapbiomas-imazon'

# ee.Authenticate()
ee.Initialize(project=PROJECT)


'''

    Config Session

'''


config = {

    'base_path': '03_multiclass',

    'channels': {
        'gv':0, 
        'npv':1, 
        'soil':2, 
        'cloud':3,
        'gvs':4,
        'ndfi':5, 
        #'csfi':6
    },

    'chip_size': 256,

    'number_output_classes': 8,

    'model_params': {
        'model_name':'multiclass_s2_v1',
        'loss': soft_dice_loss,
        'metrics':[
            running_recall, 
            running_f1, 
            running_precision, 
            tf.keras.metrics.OneHotIoU(
                num_classes=8,
                target_class_ids=[0,1,2,3,4,5,6,7],
            )
        ],
        'batch_size':15,
        'epochs': 50,
        'output_model': '03_multiclass/model',
        'output_ckpt':'03_multiclass/model/ckpt',
        'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001)
    }
}


ASSET_REFERENCE = 'projects/ee-mapbiomas-imazon/assets/lulc/reference_map/editted_classification_2020_14'

SENTINEL_NEW_NAMES = [
    'blue',
    'green',
    'red',
    'red_edge_1',
    'nir',
    'swir1',
    'swir2',
    'pixel_qa'
]

FEATURES = [
    'gv', 
    'npv', 
    'soil', 
    'cloud',
    'gvs',
    'ndfi', 
    #'csfi'
]


ASSET_IMAGES = {
    's2':{
        'idCollection': 'COPERNICUS/S2_HARMONIZED',
        'bandNames': ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'QA60'],
        'newBandNames': SENTINEL_NEW_NAMES,
    }
}

'''

    Request Template    

'''

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=29)

# image resolution in meters
SCALE = 10

# pre-compute a geographic coordinate system.
proj = ee.Projection('EPSG:4326').atScale(SCALE).getInfo()

# get scales in degrees out of the transform.
SCALE_X = proj['transform'][0]
SCALE_Y = -proj['transform'][4]

# patch size in pixels.
PATCH_SIZE = 256

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


'''

    Functions

'''

def normalize_channels(data):

    feature_index = list(config['channels'].values())

    data_filtered = tf.gather(data, tf.constant(feature_index), axis=-1)

    unstacked = tf.unstack(data_filtered, axis=2)

    data_norm = []

    for i in unstacked:
        min_arr = tf.reduce_min(i)
        max_arr = tf.reduce_max(i)

        tensor = tf.divide(
            tf.subtract(i, min_arr),
            tf.subtract(max_arr, min_arr)
        )

        data_norm.append(tensor)

    data_normalized = tf.stack(data_norm, axis=2)

    return data_normalized

@retry(delay=0.5)
def get_patch(items):

    """Get a patch centered on the coordinates, as a numpy array."""

    response = {'data':None, 'error':'', 'affine':[]}
    
    coords = items

    image = image_sensor

    request = dict(REQUEST)
    request['expression'] = image
    request['grid']['affineTransform']['translateX'] = coords[0] + OFFSET_X
    request['grid']['affineTransform']['translateY'] = coords[1] + OFFSET_Y


    # criação do objeto Affine usando os parâmetros fornecidos
    transform = Affine(
        request['grid']['affineTransform']['scaleX'], 
        request['grid']['affineTransform']['shearX'], 
        request['grid']['affineTransform']['translateX'],
        request['grid']['affineTransform']['shearY'],
        request['grid']['affineTransform']['scaleY'], 
        request['grid']['affineTransform']['translateY']
    )

    # for georeference convertion
    response['affine'] = transform
    
    try:
        data = np.load(io.BytesIO(ee.data.computePixels(request)))
        response['data'] = data
    except ee.ee_exception.EEException as e:
        print(e)
        return response
    
    return data

def predict(items, year, month, k):


    future_to_point = {EXECUTOR.submit(get_patch, item): item for item in items}

    for future in concurrent.futures.as_completed(future_to_point):
        # check_memory_usage()
        
        response = future.result()

        if response['data'] is not None:

            data = structured_to_unstructured(data)
            
            data_norma = normalize_channels(data)
            
            data_transposed = np.transpose(data_norma, (1,2,0))
            
            data_transposed = np.expand_dims(data_transposed, axis=0)


            # add exception here
            # for prediction its necessary to have an extra dimension representing the bach
            probabilities = model.predict(data_transposed)


            # it only checks the supposed prediction and skip it if there is no logging
            prediction = np.copy(probabilities)
            prediction[prediction < 0.2] = 0
            prediction[prediction >= 0.2] = 1

            if np.max(prediction[0]) == 0.0 or np.max(prediction[0]) == 0:
                continue
            

            probabilities = probabilities * 100
            probabilities = probabilities[0].astype(int)


            probabilities = np.transpose(probabilities, (2,0,1))

            name = ''

            print(name)
            
            with rasterio.open(
                name,
                'w',
                driver = 'COG',
                count = 1,
                height = probabilities.shape[1],
                width  = probabilities.shape[2],
                dtype  = probabilities.dtype,
                crs    = rasterio.crs.CRS.from_epsg(4326),
                transform=response['affine']
            ) as output:
                output.write(probabilities)


'''

    Input 

'''

reference_data = ee.Image(ASSET_REFERENCE).rename('label')\
    .rename('label')


roi = reference_data.geometry()

collection = ee.ImageCollection(ASSET_IMAGES['s2']['idCollection'])\
    .filterDate('2020-05-30', '2020-10-31')\
    .filterBounds(roi)\
    .filter('CLOUDY_PIXEL_PERCENTAGE < 30')\
    .select(ASSET_IMAGES['s2']['bandNames'], ASSET_IMAGES['s2']['newBandNames'])


collection_w_cloud = remove_cloud_s2(collection)

collection_w_cloud = collection_w_cloud\
    .map(lambda image: get_fractions(image))\
    .map(lambda image: get_ndfi(image))\
    .map(lambda image: get_csfi(image))

image_sensor = ee.Image(collection_w_cloud.reduce(ee.Reducer.median())).clip(roi)



'''
    
    Get Seeds

'''

# get centroids
seeds = image_sensor.sample(region=roi, scale=10 * config['chip_size'], geometries=True)
coords = seeds.reduceColumns(ee.Reducer.toList(), ['.geo']).get('list').getInfo()
coords = [x['coordinates'] for x in coords]

'''
    Load model
'''

model = keras.saving.load_model(config['model_params']['output_model'] + '/lulc_v1.keras', compile=False)

model.compile(
    optimizer=config['model_params']['optimizer'], 
    loss=config['model_params']['loss'], 
    metrics=config['model_params']['metrics']
)