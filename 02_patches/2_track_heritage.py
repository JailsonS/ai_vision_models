import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label
from collections import defaultdict
import rasterio
from glob import glob
from scipy.ndimage import label as label_ndimage
from pprint import pprint

'''
    
    Config

'''


PATH_IMAGES = '02_patches/data'

YEARS = [
    1985, 1986, 1987, 1988, 1989,
    1990, 1991, 1992, 1993, 1994
]

CHUNK_SIZE = 900



'''
    
    Input

'''

list_images = list(glob(f'{PATH_IMAGES}/chunks_combined_*'))

'''
    
    Functions

'''

def generate_heritage(years):
    for layer_index, year in enumerate(years):

        if layer_index == len(years) - 1: 
            break

        path_current = f'{PATH_IMAGES}/chunks_combined_{str(year)}.tif'
        path_next = f'{PATH_IMAGES}/chunks_combined_{str(year + 1)}.tif'

        current_layer = rasterio.open(path_current)
        next_layer = rasterio.open(path_next)

        # proj_current = {'crs':current_layer.crs,'transform':current_layer.transform}
        # proj_next = {'crs':next_layer.crs,'transform':next_layer.transform}

        current_layer = current_layer.read()[0]
        next_layer = next_layer.read()[0]

        # iterate over chunks 
        for i in range(0, current_layer.shape[0], CHUNK_SIZE):
            for j in range(0, current_layer.shape[1], CHUNK_SIZE):
                current_chunk = current_layer[i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]

                # find unique IDs in the current chunk
                current_ids, current_counts = np.unique(current_chunk, return_counts=True)

                for curr_id, count_curr in zip(current_ids, current_counts):
                    # if curr_id == 0: continue

                    # mask for current ID in current chunk
                    curr_mask = current_chunk == curr_id
                    
                    # check overlapping regions in the next layer chunk
                    overlapping_ids, counts = np.unique(next_layer[i:i+CHUNK_SIZE, j:j+CHUNK_SIZE][curr_mask], return_counts=True)


                    # check if no overlap and add as lost value
                    if len(overlapping_ids) == 0:
                        yield ((i, j), (layer_index, curr_id, count_curr), (layer_index + 1, None, 0))

                    for next_id, count in zip(overlapping_ids, counts):
                        if next_id == 0: continue
                        
                        yield ((i, j), (layer_index, curr_id, count_curr), (layer_index + 1, next_id, count))
                        


'''
    
    Running

'''


combined = defaultdict(lambda: [0, 0])


for i in generate_heritage(YEARS):

    _, (layer_index, curr_id, count_curr), (next_layer_index, next_id, count) = i

    key = (layer_index, curr_id, next_layer_index, next_id)

    combined[key][0] += count_curr
    combined[key][1] += count


for key, (total_curr, total_next) in combined.items():
    if key[1] == 0 and key[0] <= 1:
        print(f"From Layer {key[0]} with ID {key[1]} to Layer {key[2]} with ID {key[3]}: Current Count = {total_curr}, Next Count = {total_next}")