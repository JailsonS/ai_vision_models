from neo4j import GraphDatabase

class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    '''
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
        '''

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

