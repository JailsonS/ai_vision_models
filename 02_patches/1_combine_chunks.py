import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label
from collections import defaultdict
import rasterio
from glob import glob
from scipy.ndimage import label as label_ndimage
from pprint import pprint

PATH_IMAGES = '02_patches/data'

YEARS = list(range(1985, 2000, 1))

chunk_size = 900

'''
    Funcions
'''

import numpy as np
from skimage.measure import label


def relabel_array(array, chunk_size=5):
    labeled_array, num_features = label_ndimage(array)
    rows, cols = array.shape

    for i in range(0, rows, chunk_size):
        for j in range(0, cols, chunk_size):
            if i + chunk_size < rows:
                update_labels(labeled_array, i + chunk_size - 1, j, i + chunk_size, j)
            if j + chunk_size < cols:
                update_labels(labeled_array, i, j + chunk_size - 1, i, j + chunk_size)
            if i + chunk_size < rows and j + chunk_size < cols:
                update_labels(labeled_array, i + chunk_size - 1, j + chunk_size - 1, i + chunk_size, j + chunk_size)

    return relabel_connected_components(labeled_array)

def update_labels(array, x1, y1, x2, y2):
    label1 = array[x1, y1]
    label2 = array[x2, y2]
    if label1 != 0 and label2 != 0 and label1 != label2:
        array[array == label2] = label1

def relabel_connected_components(array):
    unique_labels = np.unique(array)
    relabeled_array = np.zeros_like(array)
    new_label = 1

    for label in unique_labels:
        if label != 0:
            relabeled_array[array == label] = new_label
            new_label += 1

    return relabeled_array









for year in YEARS:

    path = f'{PATH_IMAGES}/chunks/chunks_{str(year)}.tif'

    array_obj = rasterio.open(path)

    proj = {
        'crs':array_obj.crs,
        'transform':array_obj.transform
    }


    array = array_obj.read()[0]

    array = relabel_array(array, chunk_size=chunk_size)

    data = np.expand_dims(array, axis=0)

    print(f"Classified array at time {year}:\n{array}")

    
    
    name = f'{PATH_IMAGES}/chunks_combined_{str(year)}.tif'

    with rasterio.open(
        name,
        'w',
        driver = 'COG',
        count = 1,
        height = np.array(data).shape[1],
        width  = np.array(data).shape[2],
        dtype  = data.dtype,
        crs    = rasterio.crs.CRS.from_epsg(4326),
        transform = proj['transform']
    ) as output:
        output.write(data)

    print(f'shape {data.shape}')
    
