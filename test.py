
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

from pprint import pprint


import ee
import concurrent, gc
import logging
import psutil


from numpy.lib.recfunctions import structured_to_unstructured
from retry import retry
from glob import glob
from pprint import pprint
from rasterio.transform import Affine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='sad-deep-learning-274812')



asset = 'projects/imazon-simex/LULC/COLLECTION9/integrated-transversal'

col = ee.ImageCollection(asset).filter('version == "18"')

ids = col.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()

for id in ids:
    asset_del = f'{asset}/{id}'
    print(f'deleting asset: {asset_del}')
    ee.data.deleteAsset(asset_del)
