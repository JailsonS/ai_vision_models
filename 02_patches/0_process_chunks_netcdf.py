import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label, regionprops
from netCDF4 import Dataset
import rasterio
from glob import glob
from rasterio.transform import from_origin

PATH_IMAGES = '02_patches/data'

YEARS = [1985, 1986]

chunk_size = 200

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

previous_labels = None

for idx, year in enumerate(YEARS):
    path = f'{PATH_IMAGES}/netcdf_{str(year)}.nc'

    with Dataset(path, 'r') as nc_file:

        # Acessar o DataArray (assumindo que o nome da variável é 'variable_name')
        da = nc_file['variable_name']

        print(da)

        # Extrair os dados e as coordenadas
        data_ = da.values
        lat = da['lat'].values
        lon = da['lon'].values

        # Calcular a transformação affine a partir das coordenadas
        transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

        processed_chunks = []
        chunk_shape = (chunk_size, chunk_size)

        for i in range(0, data_.shape[0], chunk_size):
            for j in range(0, data_.shape[1], chunk_size):
                arr_chunk = data_[i:i + chunk_size, j:j + chunk_size]

                if idx > 0:
                    with Dataset(f'{PATH_IMAGES}/netcdf_{str(year-1)}.nc', 'r') as prev_nc_file:
                        var_prev = prev_nc_file.variables['variable_name']
                        arr_prev_chunk = var_prev[i:i + chunk_size, j:j + chunk_size]
                    prev_chunks = create_chunks(arr_prev_chunk, chunk_size)
                else:
                    prev_chunks = [None] * len(arr_chunk)

                combined_array = np.zeros_like(arr_chunk)
                labels = label(arr_chunk, connectivity=1)
                combined_array[:labels.shape[0], :labels.shape[1]] = labels

                if idx > 0:
                    prev_chunk = prev_chunks[0]  # Usando apenas o primeiro chunk anterior
                    prev_labels_chunk = previous_labels[0] if previous_labels else None
                    combined_array = update_labels(prev_labels_chunk, prev_chunk, labels, combined_array)
                
                processed_chunks.append(combined_array)

        previous_labels = processed_chunks
        processed_array = reassemble_chunks(processed_chunks, data_.shape, chunk_size)

        # export 
        data = np.expand_dims(processed_array, axis=0)

        print(f"Classified array at time {str(year)}:\n{processed_array}")
        
        name = f'{PATH_IMAGES}/chunks/chunks_{str(year)}.tif'

        with rasterio.open(
            name,
            'w',
            driver='COG',
            count=1,
            height=np.array(data).shape[1],
            width=np.array(data).shape[2],
            dtype=data.dtype,
            crs=rasterio.crs.CRS.from_epsg(4326),
            transform=transform
        ) as output:
            output.write(data)

        print(f'Shape {data.shape}')
