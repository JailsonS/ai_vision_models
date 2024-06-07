import numpy as np
from skimage.measure import label
from scipy.ndimage import find_objects
from neo4j import GraphDatabase

# Função para processar uma camada e encontrar heranças
def process_layer(layer, previous_labels, current_id, layer_index):
    labeled_array, num_features = label(layer, return_num=True, background=0,connectivity=1)
    objects_current = find_objects(labeled_array)

    edges = []
    tracking_info = {}
    new_labels = np.zeros_like(layer, dtype=int)
    parent_map = {}

    for current_obj_id in range(1, num_features + 1):
        current_slice = objects_current[current_obj_id - 1]
        if current_slice is None:
            continue
        current_object = labeled_array[current_slice] == current_obj_id

        parents = []
        if previous_labels is not None:
            objects_previous = find_objects(previous_labels)
            for previous_obj_id in range(1, len(objects_previous) + 1):
                previous_slice = objects_previous[previous_obj_id - 1]
                if previous_slice is None:
                    continue

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

        if len(parents) == 1:
            parent_id = parents[0]
            parent_map[current_obj_id] = parent_id
            new_labels[current_slice][current_object] = parent_id
            if parent_id not in tracking_info:
                tracking_info[parent_id] = {'layer': layer_index, 'size': 0}
            tracking_info[parent_id]['size'] = np.sum(current_object)  # Atualiza o tamanho do objeto
        else:
            tracking_info[current_id] = {'layer': layer_index, 'size': np.sum(current_object)}
            new_labels[current_slice][current_object] = current_id
            for parent in parents:
                edges.append((parent, current_id))
            current_id += 1

    if previous_labels is not None:
        # Mantém os labels dos objetos que não geraram novos filhos e não perderam área
        new_labels[previous_labels != 0] = previous_labels[previous_labels != 0]

    return new_labels, tracking_info, edges

def process_layer_(layer, previous_labels, current_id, layer_index):
    labeled_array, num_features = label(layer, return_num=True, background=0, connectivity=1)
    objects_current = find_objects(labeled_array)

    edges = []
    tracking_info = {}
    new_labels = np.zeros_like(layer, dtype=int)
    parent_map = {}

    for current_obj_id in range(1, num_features + 1):
        current_slice = objects_current[current_obj_id - 1]
        if current_slice is None:
            continue
        current_object = labeled_array[current_slice] == current_obj_id

        parents = []
        if previous_labels is not None:
            objects_previous = find_objects(previous_labels)
            for previous_obj_id in range(1, len(objects_previous) + 1):
                previous_slice = objects_previous[previous_obj_id - 1]
                if previous_slice is None:
                    continue

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

        if len(parents) == 1:
            parent_id = parents[0]
            parent_map[current_obj_id] = parent_id
            new_labels[current_slice][current_object] = parent_id
            if parent_id not in tracking_info:
                tracking_info[parent_id] = {'layer': layer_index, 'size': 0}
            tracking_info[parent_id]['size'] = np.sum(current_object)  # Atualiza o tamanho do objeto
        else:
            # Verifica se os novos objetos se tocam
            touch_objects = []
            if previous_labels is not None:  # Adiciona a verificação aqui
                for label_value in np.unique(previous_labels[current_slice]):
                    if label_value != 0 and np.any(previous_labels[current_slice] == label_value):
                        touch_objects.append(label_value)


            if len(touch_objects) == 1:
                parent_id = touch_objects[0]
                new_labels[current_slice][current_object] = parent_id
                if parent_id not in tracking_info:
                    tracking_info[parent_id] = {'layer': layer_index, 'size': 0}
                tracking_info[parent_id]['size'] += np.sum(current_object)  # Atualiza o tamanho do objeto
            else:
                # Atribui um novo rótulo único para todos os objetos que se tocam
                new_label_value = current_id
                new_labels[current_slice][current_object] = new_label_value
                for touch_obj in touch_objects:
                    new_labels[new_labels == touch_obj] = new_label_value
                tracking_info[new_label_value] = {'layer': layer_index, 'size': np.sum(current_object)}
                current_id += 1


            if len(touch_objects) == 1:
                parent_id = touch_objects[0]
                new_labels[current_slice][current_object] = parent_id
                if parent_id not in tracking_info:
                    tracking_info[parent_id] = {'layer': layer_index, 'size': 0}
                tracking_info[parent_id]['size'] += np.sum(current_object)  # Atualiza o tamanho do objeto
            else:
                # Atribui um novo rótulo único para todos os objetos que se tocam
                new_label_value = current_id
                new_labels[current_slice][current_object] = new_label_value
                for touch_obj in touch_objects:
                    new_labels[new_labels == touch_obj] = new_label_value
                tracking_info[new_label_value] = {'layer': layer_index, 'size': np.sum(current_object)}
                current_id += 1


            else:
                # Encontra todos os rótulos conectados aos novos objetos
                connected_labels = set()
                if previous_labels is not None:  
                    connected_labels = set(previous_labels[current_slice][previous_labels[current_slice] != 0])


                if connected_labels:
                    # Atualiza os rótulos dos objetos atuais para o rótulo conectado
                    new_label_value = min(connected_labels)
                    new_labels[current_slice][current_object] = new_label_value

                    # Atualiza o tamanho do objeto conectado
                    tracking_info[new_label_value]['size'] += np.sum(current_object)
                else:
                    # Se não houver objetos conectados, atribui um novo rótulo
                    new_labels[current_slice][current_object] = current_id
                    tracking_info[current_id] = {'layer': layer_index, 'size': np.sum(current_object)}
                    current_id += 1

                for parent in parents:
                    edges.append((parent, current_id - 1))

    if previous_labels is not None:
        # Mantém os labels dos objetos que não geraram novos filhos e não perderam área
        new_labels[previous_labels != 0] = previous_labels[previous_labels != 0]


    return new_labels, tracking_info, edges


# Função principal para rastrear objetos através das camadas
def track_objects(stack):
    previous_labels = None
    current_id = 1
    tracking_info = {}
    edges = []
    resulting_stack = []

    for i, current_labels in enumerate(stack):
        new_labels, new_tracking_info, new_edges = process_layer_(current_labels, previous_labels, current_id, i)
        resulting_stack.append(new_labels)

        for obj_id, info in new_tracking_info.items():
            if obj_id not in tracking_info:
                tracking_info[obj_id] = info
            else:
                tracking_info[obj_id]['size'] = info['size']  # Atualiza o tamanho do objeto

        current_id = max(current_id, max(new_tracking_info.keys(), default=current_id))

        edges.extend(new_edges)

        previous_labels = new_labels

    return resulting_stack, tracking_info, edges

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
resulting_stack, tracking_info, edges = track_objects(stack)
'''
# Configuração do Neo4j
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

store_in_neo4j(tracking_info, edges, uri, user, password)
'''
def print_heritage(tracking_info, edges):
    # Criando um dicionário para armazenar as informações de herança de cada objeto
    heritage_dict = {}

    # Populando o dicionário com as informações de herança
    for obj_id, info in tracking_info.items():
        heritage_dict[obj_id] = {'layer': info['layer'], 'size': info['size'], 'children': []}

    for parent, child in edges:
        heritage_dict[parent]['children'].append(child)

    # Pilha para armazenar os objetos a serem processados
    stack = [(obj_id, 0) for obj_id in heritage_dict.keys()]

    # Iterando sobre a pilha e imprimindo as informações de herança em profundidade
    while stack:
        obj_id, level = stack.pop()
        info = heritage_dict[obj_id]
        print("  " * level + f"Object {obj_id}: Layer {info['layer']}, Size {info['size']}")

        # Adicionando os filhos do objeto atual à pilha com um nível mais profundo
        for child_id in sorted(info['children'], reverse=True):
            stack.append((child_id, level + 1))
print("Tracking Info:")
#print_heritage(tracking_info, edges)

print("\nResulting Stack:")
for i, layer in enumerate(resulting_stack):
    print(f"Layer {i}:\n{layer}")
