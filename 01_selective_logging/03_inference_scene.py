
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

from pprint import pprint

import numpy as np
import tensorflow as tf
import ee, io, rasterio, keras
import concurrent, gc

from tensorflow.keras import backend as  K

from utils.metrics import *
from utils.index import *
from models.UnetDefault import Unet

from rasterio.merge import merge
from numpy.lib.recfunctions import structured_to_unstructured
from retry import retry
from glob import glob
from pprint import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='sad-deep-learning-274812')


'''

    Assets

'''



ASSET_COLLECTION = 'COPERNICUS/S2_HARMONIZED'
#ASSET_TILES = 'projects/imazon-simex/MAPASREFERENCIA/GRID_SENTINEL'

ASSET_TILES = 'projects/mapbiomas-workspace/AUXILIAR/SENTINEL2/grid_sentinel'

ASSET_LEGAL_AMAZON = 'users/jailson/brazilian_legal_amazon'

ASSET_UF = 'projects/mapbiomas-workspace/AUXILIAR/estados-2016'

ASSET_SIMEX = 'users/jailson/simex/te_amz_legal_exp_simex_2020'

'''
    Config Info
'''

ADD_NDFI = False
APPLY_BRIGHT = False

TILES = []

TILES_FINISHED = [
    '21LXF',
    '21LXG',
    '21LXH',
    '21LXJ',
    '21LXK',
    '21LXL',
    #'21LYC',
    '21LYD',
    '22MCS',
    '21NYC',
    '19MEV',
    '20MLD',
    '22LCM',
    '22MEB',
    '22NEG',
    '19MFU',
    '19LDH',
    '18MXT',
    '21MUQ',
    '20NMK', # trash
    '22NDL', # trash
    '21LWK',
    '21LZK',
    '21LVC',
    '21LUJ',
    '22LCK',
    '21LWJ',
    '21LUF',
    '21KZB',
    '21KVB',
    '20LRP',
    '20LRH',
    '22LBH',
    '22LBP',
    '22LDL',
    '22LDQ',
    '21LTD',
    '21LUK',
    '20LQJ',
    '21LVJ',
    '21LUD',
    '21LTH',
    '20LPP',
    '22LCL',
    '21LVH',
    '22LEP',
    '21LTL',
    '20LRM'
    # gerar pdf da versçai di artugi 
]

KERNEL_SIZE = 512

NUM_CLASSES = 1

MODEL_PATH = '01_selective_logging/model/model_v5.keras'

OUTPUT_CHIPS = '01_selective_logging/predictions/{}/{}_{}.tif'
OUTPUT_TILE = '01_selective_logging/predictions'

'''

    Request Template


    - qual a vetorização mais eficiente (spline)

    - salvar área do pixe e área ajustada por poligono

    

'''

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=40)


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
    'B2', 'B3', 'B4', 'B8', 'B11', 'B12'
]

NEW_BAND_NAMES = [
    'red','green', 'blue', 'nir','swir1', 'swir2'
]

FEATURES = [
    'red_t0','green_t0', 'blue_t0', #'ndfi_t0',
    'red_t1','green_t1', 'blue_t1', #'ndfi_t1'
]

FEATURES_INDEX = [
    0, 1, 2,
    3, 4, 5
]



T0 = '2022-08-01'
T1 = '2022-08-30'

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

    if ADD_NDFI:
        t1_band_names = [x + '_t1' for x in NEW_BAND_NAMES + ['ndfi']]
        t0_band_names = [x + '_t0' for x in NEW_BAND_NAMES + ['ndfi']]
    else:
        t1_band_names = [x + '_t1' for x in NEW_BAND_NAMES]
        t0_band_names = [x + '_t0' for x in NEW_BAND_NAMES]


    col = ee.ImageCollection(ASSET_COLLECTION).filter(f'system:index == "{items[1][0]}"')

    # t1
    image_t1 = ee.Image(col.first()).select(BAND_NAMES, NEW_BAND_NAMES)
    


    if ADD_NDFI:
        image_t1 = get_fractions(image_t1)
        image_t1 = get_ndfi(image_t1)
        image_t1 = ee.Image(image_t1).select(NEW_BAND_NAMES + ['ndfi'])

    image_t1 = image_t1.rename(t1_band_names)

    # get relative dates for t0
    tmp_date = ee.Date(image_t1.get('system:time_start')).advance(-12, 'months')
    t0 = ee.Date.fromYMD(tmp_date.get('year'), 6, 1)
    t0_ = ee.Date.fromYMD(tmp_date.get('year'), 10, 30)


    col_t0 = ee.ImageCollection(ASSET_COLLECTION)\
        .filterDate(t0, t0_)\
        .filterBounds(ee.Geometry.Point(coords))\
        .filter('CLOUDY_PIXEL_PERCENTAGE < 30')
    
    # t0
    image_t0 = ee.Image(col_t0.median()).select(BAND_NAMES, NEW_BAND_NAMES)
    
    
    if ADD_NDFI:
        image_t0 = get_fractions(image_t0)
        image_t0 = get_ndfi(image_t0)
        image_t0 = ee.Image(image_t0).select(NEW_BAND_NAMES + ['ndfi'])
        
    image_t0 = image_t0.rename(t0_band_names)

    image = image_t0.addBands(image_t1).select(FEATURES)

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
        pprint(response)
        return None, response, items[0]
    return data, response, items[0]


def predict(items):

    with tf.device('/GPU:0'):

        future_to_point = {EXECUTOR.submit(get_patch, item): item for item in items}

        for future in concurrent.futures.as_completed(future_to_point):
            
            data, response, idx = future.result()

            if data is not None:

                data = structured_to_unstructured(data)

                data_norma = np.stack([normalize_array(data[:,:,x]) for x in FEATURES_INDEX])


                if APPLY_BRIGHT: data_norma = apply_brightness(data_norma)

                data_transposed = np.transpose(data_norma, (1,2,0))
                data_transposed = np.expand_dims(data_transposed, axis=0)


                # add exception here
                # for prediction its necessary to have an extra dimension representing the bach
                probabilities = model.predict(data_transposed)


                # it only checks the supposed prediction and skip it if there is no logging
                prediction = np.copy(probabilities)
                prediction[prediction < 0.5] = 0
                prediction[prediction >= 0.5] = 1

                if np.max(prediction[0]) == 0.0 or np.max(prediction[0]) == 0:
                    continue


                probabilities = probabilities[0]
                probabilities = np.transpose(probabilities, (2,0,1))
                
                prediction = prediction[0].astype(np.uint8)

                # data_output = tf.unstack(data, axis=2)
                # data_output.append(prediction[:,:,0])
                # data_output = np.stack(data_output, axis=2)


                prediction = np.transpose(prediction, (2,0,1))
                # prediction = np.transpose(data_output, (2,0,1))

                name = OUTPUT_CHIPS.format(k, response['item'][1][0], idx)

                print(name)

                output = rasterio.open(
                    name,
                    'w',
                    driver = 'COG',
                    count = 1,
                    height = prediction.shape[1],
                    width  = prediction.shape[2],
                    dtype  = prediction.dtype,
                    crs    = rasterio.crs.CRS.from_epsg(4326),
                    transform=rasterio.transform.from_origin(response['item'][1][1][0] + OFFSET_X,
                                                            response['item'][1][1][1] + OFFSET_Y,
                                                            SCALE_X,
                                                            SCALE_Y)
                )
                
                output.write(prediction)
                output.close()

                

                # Liberar recursos
                del data, data_norma, data_transposed, probabilities, prediction, response, idx
                gc.collect()

def flatten_extend(matrix):
    flat_list = []
    for row in matrix: flat_list.extend(row)
    return flat_list



'''

    Implementation

'''


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

'''

    Running Session

'''

# roi = ee.FeatureCollection(ASSET_LEGAL_AMAZON)

roi = ee.FeatureCollection(ASSET_UF).filter('NM_ESTADO == "MATO GROSSO"')

simex = ee.FeatureCollection(ASSET_SIMEX).filter('nm_estad_1 == "MATO GROSSO"') 

if len(TILES) == 0:
    TILES = ee.FeatureCollection(ASSET_TILES).filterBounds(simex.geometry())\
        .reduceColumns(ee.Reducer.toList(), ['NAME']).get('list').getInfo()

TILES = list(set(TILES) - set(TILES_FINISHED))

print(len(TILES))

# for k, v in TILES.items():

for k in TILES:

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
    # if len(v) == 0:
    #     col = ee.ImageCollection(ASSET_COLLECTION)\
    #         .filter(f'MGRS_TILE == "{k}"')\
    #         .filterDate(T0, T1)

    #     v = col.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()
    
    col = ee.ImageCollection(ASSET_COLLECTION)\
        .filter(f'MGRS_TILE == "{k}"')\
        .filterDate(T0, T1)

    v = col.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()

    # identify loaded images
    loaded = ['_'.join(x.split('/')[-1].split('_')[:3]) 
                for x in glob(f'{OUTPUT_TILE}/{k}/*')]

    loaded = list(set(loaded))

    list_image_id = [x for x in v if x not in loaded]

    print(len(list_image_id))

    for img_id in list_image_id:

        items = list(zip([img_id] * len(coords), coords))
        items = enumerate(items)

        # run predictions
        predict(items)

    '''

    # create mosaic 
    path_chips = [rasterio.open(x) for x in glob(f'{OUTPUT_TILE}/{k}/*')]

    mosaic, out_trans = merge(path_chips)

    with rasterio.open(
        f'{OUTPUT_TILE}/{k}_pred.tif',
        'w',
        driver = 'COG',
        count = NUM_CLASSES,
        height = mosaic.shape[1],
        width  = mosaic.shape[2],
        dtype  = mosaic.dtype,
        crs    = rasterio.crs.CRS.from_epsg(4326),
        transform = out_trans 
    ) as dest:
        dest.write(mosaic)






var ASSET_TILES = 'projects/mapbiomas-workspace/AUXILIAR/SENTINEL2/grid_sentinel'

var ASSET_LEGAL_AMAZON = 'users/jailson/brazilian_legal_amazon'


var roi = ee.FeatureCollection(ASSET_LEGAL_AMAZON);
var tiles = ee.FeatureCollection(ASSET_TILES).filterBounds(roi.geometry());

print(tiles.size()) - 46.650

    '''
#
    ## delete files
    #for f in glob(f'{OUTPUT_TILE}/{k}/*.tif'):
    #    os.remove(f)







