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



layer1 = rasterio.open('02_patches/data/examples_1995.tif')
layer2 = rasterio.open('02_patches/data/examples_2000.tif')
layer3 = rasterio.open('02_patches/data/examples_2022.tif')
# 
arrays = [layer1,layer2,layer3]


'''
    Helpers
'''



def track_heritage(arrays):
    result = []
    
    for i in range(len(arrays) - 1):
        current_layer = arrays[i][0]
        next_layer = arrays[i + 1][0]
        current_ids = np.unique(current_layer)
        next_ids = np.unique(next_layer)
        
        for curr_id in current_ids:
            if curr_id == 0: continue
            
            curr_mask = current_layer == curr_id
            count_curr = np.sum(curr_mask)
            
            # find the overlapping regions in the next layer
            overlapping_ids, counts = np.unique(next_layer[curr_mask], return_counts=True)
            for next_id, count in zip(overlapping_ids, counts):

                if next_id == 0: continue
                
                result.append((i, (curr_id, count_curr), (next_id, count)))
            
            # find lost values if no overlap
            if len(overlapping_ids) == 0:
                result.append((i, (curr_id, count_curr), (None, 0)))
        
        for next_id in next_ids:
            
            if next_id == 0: continue
            
            next_mask = next_layer == next_id
            count_next = np.sum(next_mask)
            
            # Find the parent regions in the current layer
            overlapping_ids, counts = np.unique(current_layer[next_mask], return_counts=True)
            for curr_id, count in zip(overlapping_ids, counts):
                
                if curr_id == 0: continue
                
                # Check if this tuple already exists in result
                if not any(entry == (i, (curr_id, np.sum(current_layer == curr_id)), (next_id, count)) for entry in result):
                    result.append((i, (curr_id, np.sum(current_layer == curr_id)), (next_id, count)))
                
            # Find gained values if no overlap
            if len(overlapping_ids) == 0:
                result.append((i, (None, 0), (next_id, count_next)))
    
    return result



def ingest_data_to_neo4j(data):
    # Conexão com o banco de dados Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        # Criar nós para cada camada e patch
        for position_layer, (parent_info, child_info) in enumerate(data):
            parent_id, parent_count = parent_info
            child_id, child_count = child_info
            
            # Criação de nós para os patches
            session.run(
                "MERGE (p:Patch {layer: $layer, id: $parent_id, count: $parent_count}) "
                "MERGE (c:Patch {layer: $layer, id: $child_id, count: $child_count})",
                layer=position_layer,
                parent_id=parent_id,
                parent_count=parent_count,
                child_id=child_id,
                child_count=child_count
            )
            
            # Criação de relação de herança entre os patches, se houver
            if child_id:
                session.run(
                    "MATCH (p:Patch {layer: $parent_layer, id: $parent_id}) "
                    "MATCH (c:Patch {layer: $child_layer, id: $child_id}) "
                    "MERGE (p)-[:INHERITS]->(c)",
                    parent_layer=position_layer,
                    parent_id=parent_id,
                    child_layer=position_layer + 1,
                    child_id=child_id
                )
    
    driver.close()

'''
    Main Running
'''


frag = ClassifyPatches(arrays)
classified_arrays = frag.classify()

result = track_heritage(classified_arrays)

pprint(result)

'''
for t, array in enumerate(classified_arrays):
    
    data = np.expand_dims(array[0], axis=0)
    proj = array[1]

    print(f"Classified array at time {t}:\n{array[0]}")
'''

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

