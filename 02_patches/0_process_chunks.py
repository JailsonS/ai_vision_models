import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label, regionprops
from collections import defaultdict
import rasterio
from glob import glob



PATH_IMAGES = '02_patches/data'

YEARS = [
    1985, 1986, 1987, 1988, 1989,
    1990, 1991, 1992, 1993, 1994,
    # 1995, 1996, 1997, 1998, 1999,
    # 2000, 2001, 2002, 2003, 2004,
    # 2005, 2006, 2007, 2008, 2009,
    # 2010, 2011, 2012, 2013, 2014,
    # 2015, 2016, 2017, 2018, 2019,
    # 2020, 2021, 2022
]

chunk_size = 1500

'''
    Funcions
'''

import numpy as np
from skimage.measure import label




def create_chunks(arr, chunk_size):
    chunks = []
    for i in range(0, arr.shape[0], chunk_size):
        for j in range(0, arr.shape[1], chunk_size):
            chunk = arr[i:i + chunk_size, j:j + chunk_size]
            chunks.append(chunk)
    return chunks

def reassemble_chunks(chunks, original_shape, chunk_size):
    reassembled = np.zeros(original_shape, dtype=int)
    
    chunk_idx = 0
    for i in range(0, original_shape[0], chunk_size):
        for j in range(0, original_shape[1], chunk_size):
            reassembled[i:i + chunk_size, j:j + chunk_size] = chunks[chunk_idx]
            chunk_idx += 1
    
    return reassembled

def update_labels(prev_labels, prev_chunk, curr_labels, combined_array):
    if prev_labels is None:
        return combined_array

    props_prev = regionprops(prev_labels)
    props_curr = regionprops(curr_labels)
    
    prev_slices = find_objects(prev_labels)
    curr_slices = find_objects(curr_labels)
    
    label_mapping = {}
    
    for prop_curr, curr_slice in zip(props_curr, curr_slices):
        overlap_labels = set(prev_labels[curr_slice].flatten())
        overlap_labels.discard(0)
        
        if len(overlap_labels) == 1:
            prev_label = overlap_labels.pop()
            label_mapping[prop_curr.label] = prev_label
        else:
            label_mapping[prop_curr.label] = np.max(prev_labels) + 1
    
    for label_value, mapped_label in label_mapping.items():
        combined_array[curr_labels == label_value] = mapped_label
    
    return combined_array


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


previous_labels = None

for idx, year in enumerate(YEARS):

    path = f'{PATH_IMAGES}/forest_{str(year)}.tif'

    array_obj = rasterio.open(path)

    proj = {
        'crs':array_obj.crs,
        'transform':array_obj.transform
    }

    arr = array_obj.read()[0]
    
    processed_chunks = []
    chunks = create_chunks(arr, chunk_size)



    if idx > 0:
        arr_prev = rasterio.open(f'{PATH_IMAGES}/forest_{str(year-1)}.tif').read()[0]
        prev_chunks = prev_chunk = create_chunks(arr_prev, chunk_size)
    else:
        prev_chunks = [None] * len(chunks)


    for chunk_idx, chunk in enumerate(chunks):
        combined_array = np.zeros_like(chunk)
        labels = label(chunk, connectivity=1)
        combined_array[:labels.shape[0], :labels.shape[1]] = labels


        if idx > 0:
            prev_chunk = prev_chunks[chunk_idx]
            prev_labels_chunk = previous_labels[chunk_idx] if previous_labels else None
            combined_array = update_labels(prev_labels_chunk, prev_chunk, labels, combined_array)
        
        processed_chunks.append(combined_array)


    previous_labels = processed_chunks
    processed_array = reassemble_chunks(processed_chunks, arr.shape, chunk_size)


    # export 
    data = np.expand_dims(processed_array, axis=0)

    print(f"Classified array at time {str(year)}:\n{processed_array}")
    
    name = f'{PATH_IMAGES}/chunks/chunks_{str(year)}.tif'

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