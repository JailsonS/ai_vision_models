import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label, regionprops
import rasterio
from glob import glob
from scipy.ndimage import label as label_ndimage

# Definindo o caminho para as imagens e os anos
PATH_IMAGES = '02_patches/data'

YEARS = [
    1985, 1986, 
]

SET_AREA_LABEL = False
chunk_size = 600

# Funções auxiliares para criar chunks
def create_chunks(arr, chunk_size):
    chunks = []
    for i in range(0, arr.shape[0], chunk_size):
        for j in range(0, arr.shape[1], chunk_size):
            chunk = arr[i:i + chunk_size, j:j + chunk_size]
            chunks.append((chunk, i, j))
    return chunks

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

def relabel_array(array, chunk_size=5):
    labeled_array, num_features = label_ndimage(array)
    rows, cols = array.shape

    for i in range(0, rows, chunk_size):
        for j in range(0, cols, chunk_size):
            if i + chunk_size < rows:
                update_labels_local(labeled_array, i + chunk_size - 1, j, i + chunk_size, j)
            if j + chunk_size < cols:
                update_labels_local(labeled_array, i, j + chunk_size - 1, i, j + chunk_size)
            if i + chunk_size < rows and j + chunk_size < cols:
                update_labels_local(labeled_array, i + chunk_size - 1, j + chunk_size - 1, i + chunk_size, j + chunk_size)

    if SET_AREA_LABEL:
        return set_area_label_connected_components(labeled_array)
    else:
        return relabel_connected_components(labeled_array)

def update_labels_local(array, x1, y1, x2, y2):
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

previous_labels = None

for idx, year in enumerate(YEARS):
    path = f'{PATH_IMAGES}/mosaic_{str(year)}.tif'

    array_obj = rasterio.open(path)

    proj = {
        'crs': array_obj.crs,
        'transform': array_obj.transform
    }

    arr = array_obj.read(1)
    
    chunks = create_chunks(arr, chunk_size)

    if idx > 0:
        arr_prev = rasterio.open(f'{PATH_IMAGES}/mosaic_{str(year-1)}.tif').read(1)
        prev_chunks = create_chunks(arr_prev, chunk_size)
    else:
        prev_chunks = [(None, None, None)] * len(chunks)

    for chunk_idx, (chunk, i, j) in enumerate(chunks):
        combined_array = np.zeros_like(chunk)
        labels = label(chunk, connectivity=1)
        combined_array[:labels.shape[0], :labels.shape[1]] = labels

        if idx > 0:
            prev_chunk, prev_i, prev_j = prev_chunks[chunk_idx]
            prev_labels_chunk = previous_labels
