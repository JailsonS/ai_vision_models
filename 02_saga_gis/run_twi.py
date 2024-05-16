'''
    Before Run - Install Saga
'''


'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))

from pprint import pprint

import numpy as np
import ee, rasterio
import concurrent, gc

from retry import retry
from glob import glob
from pprint import pprint
from PySAGA_cmd import (SAGA)



'''
    Config Session
'''

# your google cloud project
PROJECT_ID = 'ee-simex'


ASSET_DEM = 'projects/sat-io/open-datasets/FABDEM'

ASSET_TILES = 'users/mapbiomas_c1/bigGrids_paises'

PATH_BASE_OUTPUT = '02_saga_gis/data'

LIST_ALREADY_PROCESSED = []


# workers to paralell threads
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


ee.Initialize(project=PROJECT_ID)




'''
    Set SAGA TOOLS
'''



saga = SAGA('/usr/bin/saga_cmd')

# choosing libraries.
preprocessor = saga / 'terrain_analysis'

# choosing tools.
topographic_wetness_index = saga / 'terrain_analysis' / 'Topographic Wetness Index (One Step)'

# check method's param
print(topographic_wetness_index.execute(ignore_stderr=True).stdout)




'''
    Input Data
'''

dem = ee.ImageCollection(ASSET_DEM).mosaic().toInt16()

tiles = ee.FeatureCollection(ASSET_TILES).filter(ee.Filter.inList('grid_name', LIST_ALREADY_PROCESSED).Not())

print('tiles to process:', tiles.size().getInfo())

list_images_dem = list(glob(f'{PATH_BASE_OUTPUT}/*'))


'''

    Functions

'''


# items: [path_dem]
@retry()
def get_patch(path):

    """Get Image and Process."""

    image_name = path.split('/')[-1]
    image = rasterio.open(path)

    meta = image.meta

    image_arr = image.read()

    output_path = f'{PATH_BASE_OUTPUT}/TWI_{image_name}'

    try:

        twi = topographic_wetness_index(
            DEM=path,
            TWI=output_path,
            FLOW_METHOD=4
        )

        output = twi.execute(verbose=True, ignore_stderr=True)
  
    except ee.ee_exception.EEException as e:
        pprint(e)
        print(f'error at: {image_name}')
        return None, image_name

    return output.rasters['TWI'], image_name



def run(items):

    future_to_point = {EXECUTOR.submit(get_patch, item): item for item in items}

    for future in concurrent.futures.as_completed(future_to_point):

        data, image_name = future.result()

        if data is None: continue

        print(data)


'''

    Run Process

'''