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

YEARS = list(range(1985, 2000, 1))

CHUNK_SIZE = 900



'''
    
    Input

'''


'''
    
    Functions

'''

def track_heritage(arrays):
    result = []
    
    for layer_index, layer in enumerate(arrays):
        if layer_index == len(arrays) - 1:
            break
        
        current_layer = layer[0]
        next_layer = arrays[layer_index + 1][0]
        
        # Iterate over chunks of size 5x5
        for i in range(0, current_layer.shape[0], 5):
            for j in range(0, current_layer.shape[1], 5):
                current_chunk = current_layer[i:i+5, j:j+5]
                
                # Find unique IDs in the current chunk
                current_ids, current_counts = np.unique(current_chunk, return_counts=True)
                
                for curr_id, count_curr in zip(current_ids, current_counts):
                    if curr_id == 0: continue
                    
                    # Mask for current ID in current chunk
                    curr_mask = current_chunk == curr_id
                    
                    # Check overlapping regions in the next layer chunk
                    overlapping_ids, counts = np.unique(next_layer[i:i+5, j:j+5][curr_mask], return_counts=True)
                    
                    for next_id, count in zip(overlapping_ids, counts):
                        if next_id == 0: continue
                        
                        result.append(((i, j), (layer_index, curr_id, count_curr), (layer_index + 1, next_id, count)))
                        
                    # Check if no overlap and add as lost value
                    if len(overlapping_ids) == 0:
                        result.append(((i, j), (layer_index, curr_id, count_curr), (layer_index + 1, None, 0)))
    
    return result


'''
    
    Running

'''