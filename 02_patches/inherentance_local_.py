import numpy as np
from skimage.measure import label
from scipy.ndimage import find_objects
from collections import defaultdict
from neo4j import GraphDatabase

def classify_objects(arrays):
    obj_id = 1
    parent_map = {}  # Map object id to its parent
    obj_map = {}  # Map object id to its slice
    history = []  # List to hold the classified arrays

    def get_unique_id():
        nonlocal obj_id
        obj_id += 1
        return obj_id - 1

    def update_object_map(labeled_array, current_objects):
        objects = find_objects(labeled_array)
        obj_dict = {}
        for obj_idx, obj_slice in enumerate(objects, start=1):
            if obj_idx in current_objects:
                obj_id = current_objects[obj_idx]
            else:
                obj_id = get_unique_id()
            obj_dict[obj_idx] = obj_id
            obj_map[obj_id] = obj_slice
        return obj_dict

    def find_connected_objects(prev_array, current_array):
        connections = defaultdict(set)
        for prev_label in np.unique(prev_array):
            if prev_label == 0:
                continue
            prev_mask = prev_array == prev_label
            overlap_labels = current_array[prev_mask]
            overlap_labels = overlap_labels[overlap_labels != 0]
            unique_labels = np.unique(overlap_labels)
            for label in unique_labels:
                connections[label].add(prev_label)
        return connections

    result_arrays = []
    for t, array in enumerate(arrays):
        labeled_array, num_features = label(array, return_num=True, connectivity=1)
        current_objects = update_object_map(labeled_array, {})
        if t == 0:
            parent_map = {obj_id: obj_id for obj_id in current_objects.values()}
        else:
            prev_labeled_array, _ = history[-1]
            connections = find_connected_objects(prev_labeled_array, labeled_array)
            for curr_label, prev_labels in connections.items():
                if len(prev_labels) == 1:
                    prev_label = list(prev_labels)[0]
                    current_objects[curr_label] = prev_label
                else:
                    current_objects[curr_label] = get_unique_id()
            current_objects = update_object_map(labeled_array, current_objects)

        history.append((labeled_array, current_objects))
        result_array = np.zeros_like(labeled_array)
        for obj_idx, obj_id in current_objects.items():
            result_array[labeled_array == obj_idx] = obj_id
        result_arrays.append(result_array)
    
    return result_arrays, parent_map, history

# Função para preparar dados para Neo4J
def prepare_data_for_neo4j(history, parent_map):
    nodes = []
    relationships = []
    for t, (labeled_array, objects) in enumerate(history):
        for obj_idx, obj_id in objects.items():
            nodes.append((obj_id, t))
            parent_id = parent_map.get(obj_id)
            if parent_id and parent_id != obj_id:
                relationships.append((parent_id, obj_id, t))
    return nodes, relationships

# Função para imprimir herança dos objetos
def print_object_heritage(history, parent_map):
    for t, (labeled_array, objects) in enumerate(history):
        print(f"Tempo {t}:")
        for obj_idx, obj_id in objects.items():
            parent_id = parent_map.get(obj_id, None)
            print(f"Objeto {obj_id} (Parent: {parent_id})")

# Exemplo de uso
arrays = [
    np.array([
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0]
    ]),
    np.array([
        [0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1]
    ]),
    np.array([
        [0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1]
    ]),
    np.array([
        [0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1]
    ]),
]

classified_arrays, parent_map, history = classify_objects(arrays)
print_object_heritage(history, parent_map)


'''
# Conectar ao Neo4J e inserir dados
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

def insert_data_into_neo4j(nodes, relationships):
    with driver.session() as session:
        session.write_transaction(create_nodes, nodes)
        session.write_transaction(create_relationships, relationships)

def create_nodes(tx, nodes):
    for node_id, t in nodes:
        tx.run("CREATE (n:Object {id: $id, time: $time})", id=node_id, time=t)

def create_relationships(tx, relationships):
    for parent_id, child_id, t in relationships:
        tx.run("""
        MATCH (p:Object {id: $parent_id, time: $time})
        MATCH (c:Object {id: $child_id, time: $time})
        CREATE (p)-[:PARENT_OF]->(c)
        """, parent_id=parent_id, child_id=child_id, time=t)

nodes, relationships = prepare_data_for_neo4j(history, parent_map)
insert_data_into_neo4j(nodes, relationships)
driver.close()
'''
# Print final classified arrays
for t, array in enumerate(classified_arrays):
    print(f"Classified array at time {t}:\n{array}")
