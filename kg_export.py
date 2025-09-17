from neo4j import GraphDatabase
import json

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "palatiou"))

def export_to_json():
    with driver.session() as session:
        query = """
        MATCH (n)-[r]->(m)
        RETURN id(n) AS source_id, labels(n) AS source_labels, properties(n) AS source_props,
               type(r) AS rel_type, properties(r) AS rel_props,
               id(m) AS target_id, labels(m) AS target_labels, properties(m) AS target_props
        """
        result = session.run(query)
        data = [dict(record) for record in result]
        with open("graph_export.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

export_to_json()