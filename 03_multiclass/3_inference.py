
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

import numpy as np
import os
import ee, io
import concurrent, gc
import tensorflow as tf
import keras, rasterio

from glob import glob
from rasterio.merge import merge
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

OUTPUT_PATH = '03_multiclass/predictions'
OUTPUT_TILE = '03_multiclass/predictions/tiles'

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

    response = {'id': items[0],'data': None, 'error':'', 'affine':[]}
    
    coords = items[1]

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
    
    return response

def predict(items, year):


    future_to_point = {EXECUTOR.submit(get_patch, item): item for item in items}

    for future in concurrent.futures.as_completed(future_to_point):
        # check_memory_usage()
        
        response = future.result()

        if response['data'] is not None:

            data = structured_to_unstructured(response['data'])

            print(data.shape)
            
            data_norma = normalize_channels(data)

            print(data_norma.shape)

         
            data_transposed = np.expand_dims(data_norma, axis=0)

            print(data_transposed.shape)
            # add exception here
            # for prediction its necessary to have an extra dimension representing the bach
            probabilities = model.predict(data_transposed)            

            probabilities = probabilities * 100
            probabilities = probabilities[0].astype(int)


            probabilities = np.transpose(probabilities, (2,0,1))

            name = f'{response["id"]}_{str(year)}_pred_chunk.tif'

            print(name)
            
            with rasterio.open(
                OUTPUT_TILE + '/' +name,
                'w',
                driver = 'COG',
                count = config['number_output_classes'],
                height = probabilities.shape[1],
                width  = probabilities.shape[2],
                dtype  = probabilities.dtype,
                crs    = rasterio.crs.CRS.from_epsg(4326),
                transform=response['affine']
            ) as output:
                output.write(probabilities)

def merge_rasters_in_batches(files, batch_size=100):
    merged_image = None
    merged_transform = None
    merged_profile = None

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        
        # Open files in the current batch
        with rasterio.Env():
            chunks = [rasterio.open(x) for x in batch_files]
            
            # Merge the current batch
            batch_image, batch_transform = merge(chunks)
            
            # Use the profile from the first chunk in the batch
            if merged_profile is None:
                merged_profile = chunks[0].profile

            # If this is the first batch, set the initial merged image and transform
            if merged_image is None:
                merged_image = batch_image
                merged_transform = batch_transform
            else:
                # Merge the current batch result with the previous result
                merged_image, merged_transform = merge(
                    [rasterio.io.MemoryFile().open() for _ in [merged_image, batch_image]], 
                    out=merged_image, 
                    transform=merged_transform
                )
            
            # Close all files in the batch
            for chunk in chunks:
                chunk.close()

    return merged_image, merged_transform, merged_profile

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

image_sensor = ee.Image(collection_w_cloud.reduce(ee.Reducer.median()))\
    .select([x + '_median' for x in FEATURES])\
    .clip(roi)



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

'''
    Run
'''

items = enumerate(coords)

print(len(list(items)))

# predict(items=items, year=2020)

# mosaic chunks
list_chunks = [rasterio.open(x) for x in glob(OUTPUT_TILE + '/*')]


if len(list_chunks) == 0: exit()


image_mosaic, out_trans = merge(list_chunks)


# save mosaic pred
name_image = f'{OUTPUT_PATH}_predicted.tif'

with rasterio.open(
    name_image,
    'w',
    driver = 'COG',
    count = config['number_output_classes'],
    height = image_mosaic.shape[1],
    width  = image_mosaic.shape[2],
    dtype  = 'uint8',
    crs    = rasterio.crs.CRS.from_epsg(4326),
    transform=out_trans
) as output:
    output.write(image_mosaic)



# delete chunks
for i in glob(OUTPUT_TILE + '/*'):
    os.remove(i)

print('image processed')

# Coletar lixo após cada imagem processada
gc.collect()

# Desalocar variáveis específicas
del items
del image_mosaic
del out_trans


# https://code.earthengine.google.com/c21d517a17dab8e1ab4257de54ebb166