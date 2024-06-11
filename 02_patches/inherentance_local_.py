import numpy as np
import rasterio


from pprint import pprint
from skimage.measure import label
from scipy.ndimage import find_objects
from collections import defaultdict
from neo4j import GraphDatabase


from utils.Fragmentation import ClassifyPatches


'''
    Config Info
'''

PATH_IMAGES = ''

PATH_OUTPUT = ''

# Conectar ao Neo4J e inserir dados
URI_DB = "bolt://localhost:7687"


'''

    Input Data

'''



# layer1 = rasterio.open('02_patches/data/examples_1995.tif')
# layer2 = rasterio.open('02_patches/data/examples_2000.tif')
# layer3 = rasterio.open('02_patches/data/examples_2022.tif')
# 
# arrays = [layer1,layer2,layer3]


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



'''
    Helpers
'''

# função para preparar dados para Neo4J
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

# função para imprimir herança dos objetos
def print_object_heritage(history, parent_map):
    for t, (labeled_array, objects) in enumerate(history):
        print(f"Tempo {t}:")
        for obj_idx, obj_id in objects.items():
            parent_id = parent_map.get(obj_id, None)
            print(f"Objeto {obj_id} (Parent: {parent_id})")




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




'''
    Main Running
'''


frag = ClassifyPatches(arrays)
classified_arrays, parent_map, history = frag.classify()

# print_object_heritage(history, parent_map)



# print final classified arrays
'''
for t, array in enumerate(classified_arrays):
    
    data = np.expand_dims(array[0], axis=0)
    proj = array[1]

    print(f"Classified array at time {t}:\n{array}")
    
    name = f'{t}_output.tif'

    with rasterio.open(
        name,
        'w',
        driver = 'COG',
        count = 1,
        height = np.array(data).shape[1],
        width  = np.array(data).shape[2],
        dtype  = data.dtype,
        crs    = rasterio.crs.CRS.from_epsg(4326),
        transform = proj['transform']
    ) as output:
        output.write(data)

    print(f'shape {data.shape}')
'''


'''
    Connect to Database
'''


#driver = GraphDatabase.driver(URI_DB, auth=("neo4j", "password"))
# nodes, relationships = prepare_data_for_neo4j(history, parent_map)

#with driver.session() as session:
#    session.write_transaction(create_nodes, nodes)
#    session.write_transaction(create_relationships, relationships)

#driver.close()

'''


'''

