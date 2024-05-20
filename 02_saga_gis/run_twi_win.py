'''
    author: Jailson S. (Imazon)
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
PROJECT_ID = 'your-project'

# base path of your input images 
PATH_INPUT = '02_saga_gis\\data\\input'

# base path of your output
PATH_OUTPUT = '02_saga_gis\\data\\output'

# locate your .exe saga file
PATH_SAGA = 'C:\\Program Files\\SAGA\\saga_cmd'

# workers to paralell threads
# depending on you machine, you may increase the number of workers
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


ee.Initialize(project=PROJECT_ID)




'''
    Set SAGA TOOLS
'''



saga = SAGA(PATH_SAGA)

# choosing libraries.
preprocessor = saga / 'ta_hydrology'

# choosing tools.
topographic_wetness_index = saga / 'ta_hydrology' / 'Topographic Wetness Index (One Step)'

# check method's param
print(topographic_wetness_index.execute(ignore_stderr=True).stdout)




'''
    Input Data
'''

list_images_dem = list(glob(f'{PATH_INPUT}\\*'))
list_images_out = list(glob(f'{PATH_OUTPUT}\\*'))
list_images_out = [x.replace('TWI_', '') for x in list_images_out]

# skip files processed
list_images_dem = list(set(list_images_dem) - set(list_images_out))


'''

    Functions

'''


# items: [path_dem]
@retry()
def get_patch(path):

    """Get Image and Process."""

    image_name = path.split('\\')[-1]

    output_path = f'{PATH_OUTPUT}\\TWI_{image_name}'

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

        print(f'image finished: {data}')


'''

    Run Process

'''

run(list_images_dem)