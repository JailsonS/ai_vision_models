import numpy as np
from scipy.ndimage import find_objects
from skimage.measure import label
from collections import defaultdict
import rasterio
from glob import glob
from scipy.ndimage import label as label_ndimage
from pprint import pprint

from neo4j import GraphDatabase

'''
    
    Config

'''


PATH_IMAGES = '02_patches/data'

YEARS = [
    1985, 1986, 1987, 1988, 1989,
    1990, 1991, 1992, 1993, 1994
]

CHUNK_SIZE = 900

URI = "neo4j+s://0bc8d08b.databases.neo4j.io"
USERNAME = "neo4j"
PSW = "z6GumyPyEp066Olp7uFfSpLdsmgbi4yh6VOXoixKJJo"  # Substitua por sua senha


'''
    
    Input

'''

list_images = list(glob(f'{PATH_IMAGES}/chunks_combined_*'))

'''
    
    Functions

'''

def generate_heritage(years):
    for layer_index, year in enumerate(years):

        if layer_index == len(years) - 1: 
            break

        path_current = f'{PATH_IMAGES}/chunks_combined_{str(year)}.tif'
        path_next = f'{PATH_IMAGES}/chunks_combined_{str(year + 1)}.tif'

        current_layer = rasterio.open(path_current)
        next_layer = rasterio.open(path_next)

        # proj_current = {'crs':current_layer.crs,'transform':current_layer.transform}
        # proj_next = {'crs':next_layer.crs,'transform':next_layer.transform}

        current_layer = current_layer.read()[0]
        next_layer = next_layer.read()[0]

        # iterate over chunks 
        for i in range(0, current_layer.shape[0], CHUNK_SIZE):
            for j in range(0, current_layer.shape[1], CHUNK_SIZE):
                current_chunk = current_layer[i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]

                # find unique IDs in the current chunk
                current_ids, current_counts = np.unique(current_chunk, return_counts=True)

                for curr_id, count_curr in zip(current_ids, current_counts):
                    # if curr_id == 0: continue

                    # mask for current ID in current chunk
                    curr_mask = current_chunk == curr_id
                    
                    # check overlapping regions in the next layer chunk
                    overlapping_ids, counts = np.unique(next_layer[i:i+CHUNK_SIZE, j:j+CHUNK_SIZE][curr_mask], return_counts=True)

                    if len(overlapping_ids) == 0:
                        print('no overlap')

                    # check if no overlap and add as lost value
                    #if len(overlapping_ids) == 0:
                    #    yield ((i, j), (layer_index, curr_id, count_curr), (layer_index + 1, None, 0))

                    for next_id, count in zip(overlapping_ids, counts):  
                        next_layer_index = layer_index + 1                 
                        yield (YEARS[layer_index], curr_id, count_curr), (YEARS[next_layer_index], next_id, count)
                        

def create_and_link_nodes(tx, parent_id, parent_count, child_id, child_count, current_year, next_year):

    # query = (
    #     "MERGE (p:Parent {id: $parent_id}) "
    #     "ON CREATE SET p.count = $parent_count, p.year = $current_year "
    #     #"ON MATCH SET p.year = $current_year "  
    #     "MERGE (c:Child {id: $child_id}) "
    #     "ON CREATE SET c.count = $child_count, c.year = $next_year "
    #     #"ON MATCH SET c.year = $next_year "
    #     "MERGE (p)-[:RELATES_TO]->(c)"
    # )


    query = (
        "MERGE (p:Patch {id: $parent_id, year: $current_year}) "
        "ON CREATE SET p.count = $parent_count "
        "MERGE (c:Patch {id: $child_id, year: $next_year}) "
        "ON CREATE SET c.count = $child_count "
        "MERGE (p)-[:RELATES_TO]->(c)"
    )

    tx.run(
        query, 
        parent_id=parent_id, 
        parent_count=parent_count, 
        child_id=child_id, 
        child_count=child_count,
        current_year=current_year,
        next_year=next_year
    )


'''
    
    Running

'''

combined = defaultdict(lambda: [0, 0])

for i in generate_heritage(YEARS):
    parent, child = i
    key = (parent[0], parent[1], child[0], child[1])
    combined[key][0] += parent[2]
    combined[key][1] += child[2]

driver = GraphDatabase.driver(URI, auth=(USERNAME, PSW))

with driver.session() as session:
    for key, (total_curr, total_next) in combined.items():
        parent_id, parent_count = key[1], total_curr
        child_id, child_count = key[3], total_next

        current_year = key[0]
        next_year = key[2]
        
        session.execute_write(
            create_and_link_nodes, 
            parent_id, 
            parent_count, 
            child_id, 
            child_count,
            current_year,
            next_year
        )


'''
MATCH pt=(p:Patch)-[:RELATES_TO]->(c:Patch) 
WHERE c.id <> 0
RETURN pt;
'''