import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label, regionprops
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
import xarray as xr

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

    nc_file = xr.open_dataset(path)

    # Acessar o DataArray (assumindo que o nome da variável é 'variable_name')
    da = nc_file['variable_name']

    # Extrair os dados e as coordenadas
    data_ = da.values
    lat = da['lat'].values
    lon = da['lon'].values

    # Calcular a transformação affine a partir das coordenadas
    transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))

    height, width = data_.shape

    with rasterio.open(
        f'{PATH_IMAGES}/chunks/chunks_{str(year)}.tif',
        'w',
        driver='COG',
        count=1,
        height=height,
        width=width,
        dtype=np.int32,  # Alterei para int32 para economizar memória
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=transform
    ) as dst:
        
        for i in range(0, height, chunk_size):
            for j in range(0, width, chunk_size):
                arr_chunk = data_[i:i + chunk_size, j:j + chunk_size]

                if idx > 0:
                    prev_nc_file = xr.open_dataset(f'{PATH_IMAGES}/netcdf_{str(year-1)}.nc')
                    prev_da = prev_nc_file['variable_name']
                    arr_prev_chunk = prev_da.values[i:i + chunk_size, j:j + chunk_size]
                    prev_labels_chunk = label(arr_prev_chunk, connectivity=1)
                else:
                    prev_labels_chunk = None

                combined_array = np.zeros_like(arr_chunk, dtype=np.int32)
                labels = label(arr_chunk, connectivity=1)
                combined_array[:labels.shape[0], :labels.shape[1]] = labels

                if idx > 0:
                    combined_array = update_labels(prev_labels_chunk, arr_prev_chunk, labels, combined_array)

                dst.write(combined_array, 1, window=Window(j, i, chunk_size, chunk_size))

                if previous_labels is None:
                    previous_labels = np.zeros((height, width), dtype=np.int32)
                previous_labels[i:i + chunk_size, j:j + chunk_size] = combined_array

    print(f'Finished processing year {year}')
