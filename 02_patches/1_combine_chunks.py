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

def reassemble_chunks(chunks, original_shape, chunk_size):
    reassembled = np.zeros(original_shape, dtype=int)
    for chunk, i, j in chunks:
        reassembled[i:i + chunk_size, j:j + chunk_size] = chunk
    return reassembled

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

    # Criando a estrutura para reassemblar a imagem completa
    chunks = []
    max_i = max_j = 0

    for chunk_file in chunk_files:
        with rasterio.open(chunk_file) as src:
            chunk = src.read(1)
            _, i, j = os.path.basename(chunk_file).split('_')
            i = int(i)
            j = int(j.split('.tif')[0])
            chunks.append((chunk, i, j))
            if i > max_i:
                max_i = i
            if j > max_j:
                max_j = j

    original_shape = (max_i + chunk_size, max_j + chunk_size)

    # Reassemblando a imagem completa
    reassembled_array = reassemble_chunks(chunks, original_shape, chunk_size)

    # Reetiquetando a imagem completa
    relabelled_array = relabel_array(reassembled_array, chunk_size)

    # Salvando a imagem final reassemblada e reetiquetada
    output_path = f'{PATH_IMAGES}/reassembled/reassembled_{year}.tif'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        count=1,
        height=relabelled_array.shape[0],
        width=relabelled_array.shape[1],
        dtype=relabelled_array.dtype,
        crs=src.crs,
        transform=src.transform
    ) as output:
        output.write(relabelled_array, 1)

    print
