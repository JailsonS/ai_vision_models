import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label, regionprops
import rasterio
import os

PATH_IMAGES = '02_patches/data'
YEARS = [1985, 1986]  # Adicione outros anos conforme necessário
chunk_size = 600

def create_chunks(arr, chunk_size):
    for i in range(0, arr.shape[0], chunk_size):
        for j in range(0, arr.shape[1], chunk_size):
            yield arr[i:i + chunk_size, j:j + chunk_size], i, j

def update_labels(prev_labels, curr_labels, combined_array, prev_chunk_shape, curr_chunk_shape):
    if prev_labels is None:
        return combined_array

    prev_props = regionprops(prev_labels)
    curr_props = regionprops(curr_labels)
    
    prev_slices = find_objects(prev_labels)
    curr_slices = find_objects(curr_labels)
    
    label_mapping = {}
    
    for prop_curr, curr_slice in zip(curr_props, curr_slices):
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

def process_chunk(chunk, prev_chunk, prev_labels_chunk, chunk_coords, year, proj):
    combined_array = np.zeros_like(chunk)
    labels = label(chunk, connectivity=1)
    combined_array[:labels.shape[0], :labels.shape[1]] = labels

    if prev_chunk is not None:
        combined_array = update_labels(prev_labels_chunk, labels, combined_array, prev_chunk.shape, chunk.shape)

    chunk_name = f'{PATH_IMAGES}/chunks/chunk_{year}_{chunk_coords[0]}_{chunk_coords[1]}.npy'
    np.save(chunk_name, combined_array)

def main():
    previous_labels = None

    for idx, year in enumerate(YEARS):
        path = f'{PATH_IMAGES}/mosaic_{str(year)}.tif'

        with rasterio.open(path) as array_obj:
            proj = {
                'crs': array_obj.crs,
                'transform': array_obj.transform
            }
            arr = array_obj.read(1)

        if idx > 0:
            prev_path = f'{PATH_IMAGES}/mosaic_{str(year - 1)}.tif'
            with rasterio.open(prev_path) as prev_array_obj:
                arr_prev = prev_array_obj.read(1)
        else:
            arr_prev = None

        for chunk, i, j in create_chunks(arr, chunk_size):
            if idx > 0:
                prev_chunk = arr_prev[i:i + chunk_size, j:j + chunk_size]
                prev_labels_chunk = np.load(f'{PATH_IMAGES}/chunks/chunk_{year-1}_{i}_{j}.npy') if os.path.exists(f'{PATH_IMAGES}/chunks/chunk_{year-1}_{i}_{j}.npy') else None
            else:
                prev_chunk = None
                prev_labels_chunk = None

            process_chunk(chunk, prev_chunk, prev_labels_chunk, (i, j), year, proj)

        del arr, arr_prev

if __name__ == '__main__':
    main()
