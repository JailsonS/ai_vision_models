import numpy as np
import rasterio


from pprint import pprint
from skimage.measure import label
from scipy.ndimage import find_objects
from collections import defaultdict
from neo4j import GraphDatabase
from glob import glob

from utils.Fragmentation import ClassifyPatches

# tempo de vida dos fragmentos

# perda de área dos fragmentos parent

# ganho de área dos fragmentos parent

# número de fragmentos gerados a partir de conexões

'''
    Config Info
'''

PATH_IMAGES = '02_patches/data'
PATH_OUTPUT = '02_patches/data'

URI = "neo4j+s://18dd280e.databases.neo4j.io"
USERNAME = "neo4j"
PSW = "KZ9mxC2yA8abxd5-5HsT_92tAPBodSKD1P1ZzrWETI0"  # Substitua por sua senha

YEARS = [
    1985, 1986, 1987, 1988, 1989,
    1990, 1991, 1992, 1993, 1994,
    1995, 1996, 1997, 1998, 1999,
    2000, 2001, 2002, 2003, 2004,
    2005, 2006, 2007, 2008, 2009,
    2010, 2011, 2012, 2013, 2014,
    2015, 2016, 2017, 2018, 2019,
    2020, 2021, 2022
]

'''

    Input Data

'''


arrays = [rasterio.open(x) for x in glob(f'{PATH_IMAGES}/forest*')]


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
            
            # find the parent regions in the current layer
            overlapping_ids, counts = np.unique(current_layer[next_mask], return_counts=True)
            for curr_id, count in zip(overlapping_ids, counts):
                
                if curr_id == 0: continue
                
                # check if this tuple already exists in result
                if not any(entry == (i, (curr_id, np.sum(current_layer == curr_id)), (next_id, count)) for entry in result):
                    result.append((i, (curr_id, np.sum(current_layer == curr_id)), (next_id, count)))
                
            # find gained values if no overlap
            if len(overlapping_ids) == 0:
                result.append((i, (None, 0), (next_id, count_next)))
    
    return result

# (0, (1, 132), (1, 130))
# (0, (2, 37879), (2, 8961))
# (0, (2, 37879), (1274, 13))

class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def insert_nodes_and_relationship(self, node_data):
        with self.driver.session() as session:
            for data in node_data:
                position_layer, parent_info, child_info = data
                parent_id, parent_count = parent_info
                child_id, child_count = child_info

                idx = position_layer
                idx_next = idx + 1 if position_layer > 0 else 0

                current_year = YEARS[idx]
                next_year = YEARS[idx_next] if idx_next > 0 else YEARS[idx]

                session.write_transaction(
                    self._create_and_link_nodes, 
                    position_layer, 
                    parent_id, 
                    parent_count, 
                    child_id, 
                    child_count,
                    current_year,
                    next_year
                )

    @staticmethod
    def _create_and_link_nodes(tx, position_layer, parent_id, parent_count, child_id, child_count, current_year, next_year):
        query = (
            "MERGE (p:Node {id: $parent_id}) "
            "ON CREATE SET p.count = $parent_count, p.position_layer = $position_layer, p.year = $current_year"
            "ON MATCH SET p.position_layer = $position_layer "
            "MERGE (c:Node {id: $child_id}) "
            "ON CREATE SET c.count = $child_count, c.position_layer = $position_layer, p.year = $next_year "
            "ON MATCH SET c.position_layer = $position_layer "
            "MERGE (p)-[:RELATES_TO]->(c)"
        )
        tx.run(
            query, 
            position_layer=position_layer, 
            parent_id=parent_id, 
            parent_count=parent_count, 
            child_id=child_id, 
            child_count=child_count,
            current_year=current_year,
            next_year=next_year
        )


'''
    Main Running
'''


frag = ClassifyPatches(arrays)
classified_arrays = frag.classify()


result = track_heritage(classified_arrays)

for i in result[:3]:
    print(i)

'''
for t, array in enumerate(classified_arrays):

    data = np.expand_dims(array[0], axis=0)
    proj = array[1]

    print(f"Classified array at time {t}:\n{array[0]}")
    
    data = np.expand_dims(array[0], axis=0)
    proj = array[1]

    print(f"Classified array at time {t}:\n{array}")
    
    name = f'{PATH_OUTPUT}/{t}_output.tif'

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

neo4j_handler = Neo4jHandler(URI, USERNAME, PSW)
neo4j_handler.insert_nodes_and_relationship(result)
neo4j_handler.close()




