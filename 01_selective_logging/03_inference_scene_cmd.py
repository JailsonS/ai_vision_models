
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
import logging
import psutil
import argparse

from tensorflow.keras import backend as  K

from utils.metrics import *
from utils.index import *

from numpy.lib.recfunctions import structured_to_unstructured
from retry import retry
from glob import glob
from rasterio.merge import merge
from pprint import pprint
from rasterio.transform import Affine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='sad-deep-learning-274812')


'''

    Assets

'''



ASSET_COLLECTION = 'COPERNICUS/S2_HARMONIZED'

ASSET_TILES = 'projects/mapbiomas-workspace/AUXILIAR/cartas'

ASSET_LEGAL_AMAZON = 'users/jailson/brazilian_legal_amazon'

ASSET_UF = 'projects/mapbiomas-workspace/AUXILIAR/estados-2016'

ASSET_SIMEX = 'users/jailson/simex/te_amz_legal_exp_simex_2020'

ASSET_CLASSIFICATION = 'projects/ee-simex/assets/classification'

'''
    Config Info
'''

ADD_NDFI = False
APPLY_BRIGHT = False


TILES = []

KERNEL_SIZE = 512

NUM_CLASSES = 1

MODEL_PATH = '01_selective_logging/model/model_v5.keras'

OUTPUT_CHIPS = '01_selective_logging/predictions/{}/{}/{}/{}_{}.tif'
OUTPUT_TILE = '01_selective_logging/predictions'

'''

    Request Template    

'''

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=35)
#EXECUTOR = concurrent.futures.ProcessPoolExecutor(max_workers=10)

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
    'blue','green', 'red', 'nir','swir1', 'swir2'
]

FEATURES = [
    'red_t0','green_t0', 'blue_t0', 
    #'ndfi_t0',
    'red_t1','green_t1', 'blue_t1', 
    #'ndfi_t1'
    #'gv_t0', 'npv_t0','soil_t0','shade_t0','cloud_t0', 'ndfi_t0',
    #'gv_t1', 'npv_t1','soil_t1','shade_t1','cloud_t1', 'ndfi_t1'
]

FEATURES_INDEX = [
    0, 1, 2, 
    3, 4, 5
    #6, 7, 8, 9, 10, 11
]


'''

    Functions

'''



def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory Usage: RSS={mem_info.rss / (1024 ** 2):.2f} MB, VMS={mem_info.vms / (1024 ** 2):.2f} MB")


def check_memory_usage(threshold=0.5):
    mem_info = psutil.virtual_memory()
    available_memory_gb = mem_info.available / (1024 ** 3)
    print(f'availabel memo {available_memory_gb}')
    if available_memory_gb < threshold:
        logger.warning(f"Memory usage is high. Available memory: {available_memory_gb:.2f} GB. Restarting...")
        sys.exit(1)



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
    yeart = items[1][2] - 1
    point = ee.Geometry.Point(coords)

    if ADD_NDFI:
        t1_band_names = [x + '_t1' for x in NEW_BAND_NAMES + ['ndfi', 'gv', 'soil', 'cloud', 'shade','npv']]
        t0_band_names = [x + '_t0' for x in NEW_BAND_NAMES + ['ndfi', 'gv', 'soil', 'cloud', 'shade','npv']]
    else:
        t1_band_names = [x + '_t1' for x in NEW_BAND_NAMES]
        t0_band_names = [x + '_t0' for x in NEW_BAND_NAMES]


    col = ee.ImageCollection(ASSET_COLLECTION)\
        .filterBounds(point)\
        .filter(f'system:index == "{items[1][0]}"')
    
    # check if it has data
    if int(col.size().getInfo()) == 0:
        return None

    # t1
    image_t1 = ee.Image(col.first()).select(BAND_NAMES, NEW_BAND_NAMES)
    


    if ADD_NDFI:
        image_t1 = get_fractions(image_t1)
        image_t1 = get_ndfi(image_t1)
        image_t1 = ee.Image(image_t1).select(NEW_BAND_NAMES + ['ndfi', 'gv', 'soil', 'cloud', 'shade','npv'])

    image_t1 = image_t1.rename(t1_band_names)

    # get relative dates for t0
    # tmp_date = ee.Date(image_t1.get('system:time_start')).advance(-12, 'months')
    # t0 = ee.Date.fromYMD(tmp_date.get('year'), 5, 1)
    # t0_ = ee.Date.fromYMD(tmp_date.get('year'), 7, 30)

    t0 = ee.Date.fromYMD(yeart, 5, 1)
    t0_ = ee.Date.fromYMD(yeart, 7, 30)


    col_t0 = ee.ImageCollection(ASSET_COLLECTION)\
        .filterDate(t0, t0_)\
        .filterBounds(ee.Geometry.Point(coords))\
        .filter('CLOUDY_PIXEL_PERCENTAGE < 20')
    
    # t0
    image_t0 = ee.Image(col_t0.median()).select(BAND_NAMES, NEW_BAND_NAMES)
    
    
    if ADD_NDFI:
        image_t0 = get_fractions(image_t0)
        image_t0 = get_ndfi(image_t0)
        image_t0 = ee.Image(image_t0).select(NEW_BAND_NAMES + ['ndfi', 'gv', 'soil', 'cloud', 'shade','npv'])
        
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


@retry(delay=0.5)
def get_patch(items):

    """Get a patch centered on the coordinates, as a numpy array."""

    response = {'error': '', 'item':items}
    
    coords = items[1][1]

    image = get_image(items)

    if image == None:
        return None, None, None

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
    except ee.ee_exception.EEException as e:
        response['error']= e
        pprint(response)
        return None, response, items[0]
    return data, response, items[0]


def predict(items, year, month, k):


    future_to_point = {EXECUTOR.submit(get_patch, item): item for item in items}

    for future in concurrent.futures.as_completed(future_to_point):
        check_memory_usage()
        
        data, response, idx = future.result()

        if data is not None:

            data = structured_to_unstructured(data)

            
            data_norma = np.stack([normalize_array(data[:,:,x]) 
                                        for x in FEATURES_INDEX]) 

            if APPLY_BRIGHT: 
                data_norma = apply_brightness(data_norma)

            
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

            
            #prediction = prediction[0].astype(np.uint8)
            #prediction = np.transpose(prediction, (2,0,1))

            name = OUTPUT_CHIPS.format(year,month,k,response['item'][1][0], idx)

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

#model.save('01_selective_logging/model/vertex')

#exit()

'''

    Running Session

'''

# roi = ee.FeatureCollection(ASSET_LEGAL_AMAZON)

roi = ee.FeatureCollection(ASSET_UF).filter('NM_ESTADO == "MATO GROSSO"')

simex = ee.FeatureCollection(ASSET_SIMEX).filter('nm_estad_1 == "PARA"') 

if len(TILES) == 0:
    TILES = ee.FeatureCollection(ASSET_TILES)\
        .filterBounds(simex.geometry())\
        .reduceColumns(ee.Reducer.toList(), ['grid_name']).get('list').getInfo()

# for k, v in TILES.items():


def main(yeartarget, year, month):

    year = str(year)

    for k in TILES:
            last_day = '28' if month == '02' else '30'

            check_memory_usage()  # Check memory at the start of each month loop


            T0 = '{}-{}-{}'.format(year, month, '01')
            T1 = '{}-{}-{}'.format(year, month, last_day)


            # identify loaded images from asset
            list_loaded_cls = ee.ImageCollection(ASSET_CLASSIFICATION)\
                .filter(f'version == "1"')\
                .filterDate(T0, T1)\
                .reduceColumns(ee.Reducer.toList(), ['image_id']).get('list').getInfo()
            
            tiles_loaded_cls = [x[-5:] for x in list_loaded_cls]

            # if tile is already processed, skip
            if k in tiles_loaded_cls: continue



            if not os.path.isdir(f'01_selective_logging/predictions/{year}'):
                os.mkdir(os.path.abspath(f'01_selective_logging/predictions/{year}'))

            if not os.path.isdir(f'01_selective_logging/predictions/{year}/{month}'):
                os.mkdir(os.path.abspath(f'01_selective_logging/predictions/{year}/{month}'))

            if not os.path.isdir(f'01_selective_logging/predictions/{year}/{month}/{k}'):
                os.mkdir(os.path.abspath(f'01_selective_logging/predictions/{year}/{month}/{k}'))
            else: continue



            grid = ee.FeatureCollection(ASSET_TILES).filter(f'grid_name == "{k}"')
            grid_feat = ee.Feature(grid.first()).set('id', 1)
            grid_img = ee.FeatureCollection([grid_feat]).reduceToImage(['id'], ee.Reducer.first())


            # get centroids
            seeds = grid_img.sample(region=grid_feat.geometry(), scale=10 * PATCH_SIZE, geometries=True)
            coords = seeds.reduceColumns(ee.Reducer.toList(), ['.geo']).get('list').getInfo()
            coords = [x['coordinates'] for x in coords]


            col = ee.ImageCollection(ASSET_COLLECTION)\
                .filter('CLOUDY_PIXEL_PERCENTAGE <= 80')\
                .filterBounds(grid)\
                .filterDate(T0, T1)

            v = col.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()

            # identify loaded images
            loaded = [x.split('/')[-1].replace('pred_','')
                        for x in glob(f'01_selective_logging/predictions/{year}/{month}/*/pred*')]

            loaded = list(set(loaded))


            list_image_id = [x for x in v if x not in loaded]
            list_image_id = [x for x in v if x not in list_loaded_cls]

            for img_id in list_image_id:

                items = list(zip([img_id] * len(coords), coords, [yeartarget] * len(coords)))
                items = enumerate(items)

                # run predictions
                predict(items, year, month, k)


                # mosaic chunks
                list_chunks = [rasterio.open(x) for x in glob(f'{OUTPUT_TILE}/{str(year)}/{month}/{k}/{img_id}*')]
                image_mosaic, out_trans = merge(list_chunks)


                # save mosaic pred
                name_image = f'{OUTPUT_TILE}/{str(year)}/{month}/{k}/pred_{img_id}_{k}.tif'
                
                with rasterio.open(
                    name_image,
                    'w',
                    driver = 'COG',
                    count = 1,
                    height = image_mosaic.shape[1],
                    width  = image_mosaic.shape[2],
                    dtype  = 'uint8',
                    crs    = rasterio.crs.CRS.from_epsg(4326),
                    transform=out_trans
                ) as output:
                    output.write(image_mosaic)


                # delete chunks
                for i in glob(f'{OUTPUT_TILE}/{str(year)}/{month}/{k}/{img_id}*'):
                    os.remove(i)

                print(f'image {img_id} processed')

                

                # Coletar lixo após cada imagem processada
                gc.collect()

                # Desalocar variáveis específicas
                del items
                del list_chunks
                del image_mosaic
                del out_trans


            gc.collect()


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--yeartarget', type=int, help="this is the year of calendar")
    parser.add_argument('--month', type=str,help="month to be processed (ex: 01)")
    parser.add_argument('--year', type=int, help="this is the year to search images for t1")
    
    args = parser.parse_args()

    main(yeartarget=args.yeartarget, year=args.year, month=args.month)

# pid
# 1904
# 5744
# 