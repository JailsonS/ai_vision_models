import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label, regionprops
import rasterio
from glob import glob

# Definindo o caminho para as imagens e os anos
PATH_IMAGES = '02_patches/data'

YEARS = [
    1985, 1986, 
]

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
            prev_labels_chunk = previous_labels[chunk_idx] if previous_labels else None
            combined_array = update_labels(prev_labels_chunk, prev_chunk, labels, combined_array)
        
        # Exportando cada chunk processado como arquivo TIFF
        data = np.expand_dims(combined_array, axis=0)
        chunk_name = f'{PATH_IMAGES}/chunks/chunk_{str(year)}_{i}_{j}.tif'

        with rasterio.open(
            chunk_name,
            'w',
            driver='GTiff',
            count=1,
            height=combined_array.shape[0],
            width=combined_array.shape[1],
            dtype=combined_array.dtype,
            crs=proj['crs'],
            transform=rasterio.Affine(proj['transform'][0], proj['transform'][1], proj['transform'][2] + j,
                                      proj['transform'][3], proj['transform'][4], proj['transform'][5] + i)
        ) as output:
            output.write(data)

        print(f'Exported {chunk_name} with shape {data.shape}')

    previous_labels = [chunk for chunk, _, _ in chunks]
