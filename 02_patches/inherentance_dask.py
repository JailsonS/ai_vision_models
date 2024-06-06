import numpy as np
from skimage.measure import label
from scipy.ndimage import find_objects
from dask import delayed, compute
from dask.distributed import Client
from neo4j import GraphDatabase

# Configuração do cliente Dask
client = Client()


'''







# Função para processar uma camada e encontrar heranças
@delayed
def process_layer(layer, previous_labels, current_id, layer_index):

    labeled_array, num_features = label(layer, return_num=True)
    objects_current = find_objects(labeled_array)

    edges = []
    tracking_info = {}

    for current_obj_id in range(1, num_features + 1):
        current_slice = objects_current[current_obj_id - 1]

        if current_slice is None: continue

        current_object = labeled_array[current_slice] == current_obj_id

        parents = []
        if previous_labels is not None:
            objects_previous = find_objects(previous_labels)
            for previous_obj_id in range(1, len(objects_previous) + 1):
                
                previous_slice = objects_previous[previous_obj_id - 1]

                if previous_slice is None: continue

                intersection_slice = tuple(
                    slice(max(current_slice[d].start, previous_slice[d].start),
                          min(current_slice[d].stop, previous_slice[d].stop))
                    for d in range(len(current_slice))
                )

                if any(s.start >= s.stop for s in intersection_slice):
                    continue

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
            for parent in parents:
                edges.append((parent, current_id))
            tracking_info[current_id] = {'layer': layer_index, 'size': np.sum(current_object)}
        else:
            tracking_info[current_id] = {'layer': layer_index, 'size': np.sum(current_object)}

        current_id += 1

    return labeled_array, tracking_info, edges

# Função principal para rastrear objetos através das camadas
def track_objects(stack):
    previous_labels = None
    current_id = 1
    tracking_info = {}
    edges = []

    tasks = []
    for i, current_labels in enumerate(stack):
        task = process_layer(current_labels, previous_labels, current_id, i)
        tasks.append(task)
        previous_labels, new_tracking_info, new_edges = task.compute()

        current_id += len(new_tracking_info)

        tracking_info.update(new_tracking_info)
        edges.extend(new_edges)

    return tracking_info, edges

# Função para armazenar os dados no Neo4j
def store_in_neo4j(tracking_info, edges, uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # Criar nodos de objetos
        for obj_id, info in tracking_info.items():
            session.run(
                "CREATE (o:Object {id: $id, layer: $layer, size: $size})",
                id=obj_id, layer=info['layer'], size=info['size']
            )

        # Criar arestas de herança
        for parent, child in edges:
            session.run(
                "MATCH (p:Object {id: $parent}), (c:Object {id: $child}) "
                "CREATE (p)-[:HERITAGE]->(c)",
                parent=parent, child=child
            )

    driver.close()








# Exemplo de uso com mais camadas
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

layer4 = np.array([[0, 1, 0, 0, 1],
                   [1, 0, 0, 1, 1],
                   [0, 0, 1, 1, 1],
                   [0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1]])

layer5 = np.array([[0, 1, 0, 0, 0],
                   [1, 1, 0, 1, 0],
                   [0, 0, 0, 1, 1],
                   [0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1]])

stack = [layer1, layer2, layer3, layer4, layer5]
tracking_info, edges = track_objects(stack)

# Configuração do Neo4j
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

store_in_neo4j(tracking_info, edges, uri, user, password)

def print_heritage(tracking_info, edges):
    layers = {}
    for obj_id, info in tracking_info.items():
        layer = info['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append((obj_id, info))

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

    initial_layer = min(layers.keys())
    for obj_id, info in sorted(layers[initial_layer]):
        print_heritage_recursive(obj_id, heritage_dict)

print("Tracking Info:")
print_heritage(tracking_info, edges)
'''