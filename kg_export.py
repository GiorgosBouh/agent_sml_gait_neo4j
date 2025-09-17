# export_graph.py
from neo4j import GraphDatabase
import json, gzip

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "palatiou")
BATCH = 50000

driver = GraphDatabase.driver(URI, auth=AUTH)

def export_nodes(ndjson_path="nodes.ndjson.gz"):
    q = "MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props SKIP $skip LIMIT $limit"
    with driver.session() as sess, gzip.open(ndjson_path, "wt", encoding="utf-8") as out:
        skip = 0
        total = 0
        while True:
            recs = list(sess.run(q, skip=skip, limit=BATCH))
            if not recs: break
            for r in recs:
                out.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
            total += len(recs)
            skip += BATCH
            print(f"nodes: {total}")

def export_rels(ndjson_path="rels.ndjson.gz"):
    q = """
    MATCH (a)-[r]->(b)
    RETURN id(r) AS id, type(r) AS type, properties(r) AS props,
           id(a) AS source, id(b) AS target
    SKIP $skip LIMIT $limit
    """
    with driver.session() as sess, gzip.open(ndjson_path, "wt", encoding="utf-8") as out:
        skip = 0
        total = 0
        while True:
            recs = list(sess.run(q, skip=skip, limit=BATCH))
            if not recs: break
            for r in recs:
                out.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
            total += len(recs)
            skip += BATCH
            print(f"rels: {total}")

if __name__ == "__main__":
    export_nodes()
    export_rels()