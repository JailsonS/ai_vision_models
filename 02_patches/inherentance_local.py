import numpy as np

from skimage.measure import label
from scipy.ndimage import find_objects
from pprint import pprint


def track_objects(stack):
    previous_labels = None
    current_id = 1
    tracking_info = {}
    edges = []

    for i, current_labels in enumerate(stack):
        if previous_labels is None:
            # Primeira camada, rotular os objetos
            labeled_array, num_features = label(current_labels, return_num=True)
            previous_labels = labeled_array

            # Armazenar os objetos mãe
            for obj_id in range(1, num_features + 1):
                tracking_info[current_id] = {'layer': i, 'size': np.sum(labeled_array == obj_id)}
                current_id += 1
            continue

        # Identificar objetos na camada atual
        labeled_array, num_features = label(current_labels, return_num=True)
        objects_current = find_objects(labeled_array)
        objects_previous = find_objects(previous_labels)

        for current_obj_id in range(1, num_features + 1):
            current_slice = objects_current[current_obj_id - 1]
            if current_slice is None:
                continue
            current_object = labeled_array[current_slice] == current_obj_id

            # Verificar interseção com objetos da camada anterior
            parents = []
            for previous_obj_id in range(1, len(objects_previous) + 1):
                previous_slice = objects_previous[previous_obj_id - 1]
                if previous_slice is None:
                    continue

                # Calcular a interseção dos slices
                intersection_slice = tuple(
                    slice(max(current_slice[d].start, previous_slice[d].start),
                          min(current_slice[d].stop, previous_slice[d].stop))
                    for d in range(len(current_slice))
                )

                # Verificar se a interseção é válida (dimensão > 0)
                if any(s.start >= s.stop for s in intersection_slice):
                    continue

                # Extrair as regiões de interseção
                current_intersection = current_object[tuple(
                    slice(max(0, previous_slice[d].start - current_slice[d].start),
                          min(current_slice[d].stop - current_slice[d].start,
                              previous_slice[d].stop - current_slice[d].start))
                    for d in range(len(current_slice))
                )]

                previous_intersection = previous_labels[intersection_slice] == previous_obj_id

                intersection = np.logical_and(current_intersection, previous_intersection)
                if np.any(intersection):
                    parents.append(previous_obj_id)

            if len(parents) > 0:
                # Se há interseção, registrar como filho do(s) objeto(s) mãe(s)
                for parent in parents:
                    edges.append((parent, current_id))
                tracking_info[current_id] = {'layer': i, 'size': np.sum(current_object)}
            else:
                # Se não há interseção, é um objeto órfão
                tracking_info[current_id] = {'layer': i, 'size': np.sum(current_object)}
            current_id += 1

        # Atualizar a camada anterior
        previous_labels = labeled_array

    return tracking_info, edges

# Exemplo de uso com mais camadas:
layer1 = np.array([[0, 0, 1, 1, 0],
                   [0, 1, 1, 0, 0],
                   [1, 0, 0, 1, 1],
                   [0, 0, 1, 1, 0],
                   [1, 1, 0, 0, 0]])

layer2 = np.array([[0, 1, 1, 0, 0],
                   [1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1],
                   [1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1]])

layer3 = np.array([[0, 1, 0, 0, 0],
                   [1, 1, 0, 1, 1],
                   [0, 0, 0, 1, 1],
                   [0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1]])



stack = [layer1, layer2, layer3]
tracking_info, edges = track_objects(stack)





def prepare_for_neo4j(tracking_info, edges):
    nodes = []
    for obj_id, info in tracking_info.items():
        nodes.append({
            'id': obj_id,
            'layer': info['layer'],
            'size': info['size']
        })

    return nodes, edges

# Preparar os dados
nodes, edges = prepare_for_neo4j(tracking_info, edges)

# Função para imprimir a informação de herança na ordem cronológica
def print_heritage(tracking_info, edges):
    # Organizar nós por camada
    layers = {}
    for obj_id, info in tracking_info.items():
        layer = info['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append((obj_id, info))
    
    # Ordenar camadas por ordem cronológica e organizar heranças
    heritage_dict = {}
    for layer in sorted(layers.keys()):
        for obj_id, info in sorted(layers[layer]):
            heritage_dict[obj_id] = {'layer': layer, 'size': info['size'], 'children': []}

    for parent, child in edges:
        heritage_dict[parent]['children'].append(child)

    def print_heritage_recursive(obj_id, heritage_dict, level=0):
        info = heritage_dict[obj_id]
        print("  " * level + f"Object {obj_id}: Layer {info['layer']}, Size {info['size']}")
        for child in sorted(info['children']):
            print_heritage_recursive(child, heritage_dict, level + 1)
    
    # Imprimir a herança para cada objeto na camada inicial
    initial_layer = min(layers.keys())
    for obj_id, info in sorted(layers[initial_layer]):
        print_heritage_recursive(obj_id, heritage_dict)

print("Tracking Info:")
print_heritage(tracking_info, edges)

pprint(nodes)
