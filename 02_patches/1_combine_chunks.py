import numpy as np
import rasterio.dtypes
import rasterio

from scipy.ndimage import find_objects
from skimage.measure import label
from collections import defaultdict
from glob import glob
from scipy.ndimage import label as label_ndimage
from pprint import pprint

PATH_IMAGES = '02_patches/data'

YEARS = [
    1985, 1986, 1987, 1988, 1989,
    1990, 1991, 1992, 1993, 1994, 
    1996, 
    1997, 1998, 1999,
    2000, 2001, 2002, 2003, 2004,
    2005, 2006, 2007, 2008, 2009,
    2010, 2011, 2012, 2013, 2014,
    2015, 2016, 2017, 2018, 2019,
    2020, 2021, 2022
]

SET_AREA_LABEL = False

chunk_size = 600

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

    if SET_AREA_LABEL:
        return set_area_label_connected_components(labeled_array)
    else:
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


def set_area_label_connected_components(array):
    unique_labels = np.unique(array)
    relabeled_array = np.zeros_like(array)
    new_label = 1

    for label in unique_labels:
        count = np.count_nonzero(array[array == label])
        print('count px',count)
        if label != 0:
            relabeled_array[array == label] = (count * 900) / 10000
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
    
    name = f'{PATH_IMAGES}/chunks_combined_area_{str(year)}.tif' if SET_AREA_LABEL else f'{PATH_IMAGES}/chunks_combined_{str(year)}.tif'

    with rasterio.open(
        name,
        'w',
        driver = 'COG',
        count = 1,
        height = np.array(data).shape[1],
        width  = np.array(data).shape[2],
        dtype  = rasterio.dtypes.float32,
        crs    = rasterio.crs.CRS.from_epsg(4326),
        transform = proj['transform']
    ) as output:
        output.write(data)

    print(f'shape {data.shape}')
    
# rm -r 02_patches/data/examples* 
# 02_patches/data
# python3 02_patches/0_mosaic.py
# vim 02_patches/0_mosaic.py
# gsutil mv 02_patches/data/mosaics/mosaic_2002.tif gs://imazon/mapbiomas/degradation/fragmentation/forest_mosaic/