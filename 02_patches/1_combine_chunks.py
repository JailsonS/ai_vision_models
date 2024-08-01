import numpy as np
import rasterio
from skimage.measure import label as label_ndimage
from glob import glob
import os

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

def process_chunk(chunk, chunk_idx, chunk_size):
    relabelled_chunk = relabel_connected_components(chunk)
    return relabelled_chunk

def relabel_array(array, chunk_size=600):
    rows, cols = array.shape

    for i in range(0, rows, chunk_size):
        for j in range(0, cols, chunk_size):
            if i + chunk_size < rows:
                update_labels(array, i + chunk_size - 1, j, i + chunk_size, j)
            if j + chunk_size < cols:
                update_labels(array, i, j + chunk_size - 1, i, j + chunk_size)
            if i + chunk_size < rows and j + chunk_size < cols:
                update_labels(array, i + chunk_size - 1, j + chunk_size - 1, i + chunk_size, j + chunk_size)

    return relabel_connected_components(array)

# Definindo o caminho para as imagens e os anos
PATH_IMAGES = '02_patches/data'

YEARS = [
    1985, 1986, 
]

chunk_size = 600

for year in YEARS:
    # Coletando todos os arquivos de chunks
    chunk_files = glob(f'{PATH_IMAGES}/chunks/chunk_{year}_*.tif')

    for chunk_file in chunk_files:
        with rasterio.open(chunk_file) as src:
            chunk = src.read(1)
            basename = os.path.basename(chunk_file)
            _, i, j = basename.split('_')[1], basename.split('_')[2], basename.split('_')[3].split('.tif')[0]
            i = int(i)
            j = int(j)

            # Processando e reetiquetando o chunk
            relabelled_chunk = process_chunk(chunk, (i, j), chunk_size)

            # Salvando o chunk reetiquetado
            output_path = f'{PATH_IMAGES}/reassembled/reassembled_{year}_{i}_{j}.tif'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                count=1,
                height=relabelled_chunk.shape[0],
                width=relabelled_chunk.shape[1],
                dtype=relabelled_chunk.dtype,
                crs=src.crs,
                transform=src.transform
            ) as output:
                output.write(relabelled_chunk, 1)

            print(f'Exported reassembled and relabelled chunk to {output_path}')

# Agora, combinando os chunks reetiquetados incrementalmente
for year in YEARS:
    reassembled_array = None
    transform = None
    crs = None

    chunk_files = glob(f'{PATH_IMAGES}/reassembled_{year}_*.tif')

    for chunk_file in chunk_files:
        with rasterio.open(chunk_file) as src:
            chunk = src.read(1)
            if reassembled_array is None:
                reassembled_array = np.zeros_like(chunk)
                transform = src.transform
                crs = src.crs
            reassembled_array += chunk

    relabelled_final = relabel_array(reassembled_array, chunk_size)

    output_path = f'{PATH_IMAGES}/mosaics/reassembled_{year}.tif'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        count=1,
        height=relabelled_final.shape[0],
        width=relabelled_final.shape[1],
        dtype=relabelled_final.dtype,
        crs=crs,
        transform=transform
    ) as output:
        output.write(relabelled_final, 1)

    print(f'Exported final reassembled and relabelled image to {output_path}')
