import rasterio
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_dilation, generate_binary_structure

def update_labels(previous_labels, current_data, next_id):
    """
    Atualiza os labels dos patches conforme as regras especificadas.
    """
    # Estrutura para conexão de 8 vizinhos
    structure = generate_binary_structure(2, 2)
    
    # Identificar patches na imagem atual
    current_labels, num_features = label(current_data, structure=structure, return_num=True)
    
    # Mapear IDs anteriores para IDs atuais
    id_map = {}
    for current_label in range(1, num_features + 1):
        current_patch = current_labels == current_label
        
        # Patches que existiam anteriormente
        overlap_ids = np.unique(previous_labels[current_patch])
        overlap_ids = overlap_ids[overlap_ids != 0]  # Remover o fundo
        
        if len(overlap_ids) == 1:
            # Patch preserva o ID se não houve fragmentação
            id_map[current_label] = overlap_ids[0]
        else:
            # Novo patch ou fragmentação
            id_map[current_label] = next_id
            next_id += 1

    # Atualiza a imagem com os novos IDs
    for current_label, new_id in id_map.items():
        current_labels[current_labels == current_label] = new_id
    
    return current_labels, next_id

def process_forest_patches(input_data):
    next_id = 1  # Começa com 1 para IDs únicos
    
    # Carregar o primeiro ano
    with rasterio.open(input_data[0]) as src:
        previous_data = src.read(1)
        previous_labels, _ = label(previous_data, return_num=True)
    
    for i in range(1, len(input_data)):
        with rasterio.open(input_data[i]) as src:
            current_data = src.read(1)
        
        # Aplicar dilatação binária para identificar novas áreas conectadas
        dilated = binary_dilation(previous_labels > 0, structure=generate_binary_structure(2, 2))
        new_patches = (current_data == 1) & (dilated == 0)
        
        current_data[new_patches] = 0  # Marca novos patches desconectados como 0 temporariamente
        
        # Atualizar labels com base nos critérios
        updated_labels, next_id = update_labels(previous_labels, current_data, next_id)
        
        # Identificar novos patches desconectados e atribuir novos IDs
        new_patch_labels, num_new_patches = label(new_patches, structure=generate_binary_structure(2, 2), return_num=True)
        for new_label in range(1, num_new_patches + 1):
            next_id += 1
            updated_labels[new_patch_labels == new_label] = next_id
        
        # Salvar resultado
        output_filename = f'forest_patches_{1985 + i}.tif'
        with rasterio.open(
            output_filename, 'w',
            driver='GTiff',
            height=updated_labels.shape[0],
            width=updated_labels.shape[1],
            count=1,
            dtype=rasterio.uint32,
            crs=src.crs,
            transform=src.transform
        ) as dst:
            dst.write(updated_labels, 1)
        
        previous_labels = updated_labels

# Lista de arquivos de entrada
input_data = [
    'forest_1985.tif',
    'forest_1986.tif',
    'forest_1987.tif',
    # Adicione outros arquivos até 2022
]

# Processa os patches de floresta
process_forest_patches(input_data)
