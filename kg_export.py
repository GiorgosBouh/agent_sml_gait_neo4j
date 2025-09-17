# export_graph_elementid.py
from neo4j import GraphDatabase
import json, gzip

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "palatiou")
BATCH = 50_000

driver = GraphDatabase.driver(URI, auth=AUTH)

def export_nodes(ndjson_path="nodes.ndjson.gz"):
    """
    Γράφει ανά γραμμή:
    {
      "eid": "<elementId(n)>",
      "labels": ["Label1","Label2",...],
      "props": { ... }
    }
    """
    q = """
    MATCH (n)
    RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props
    SKIP $skip LIMIT $limit
    """
    total = 0
    with driver.session() as sess, gzip.open(ndjson_path, "wt", encoding="utf-8") as out:
        skip = 0
        while True:
            recs = list(sess.run(q, skip=skip, limit=BATCH))
            if not recs:
                break
            for r in recs:
                out.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
            total += len(recs)
            skip += BATCH
            print(f"nodes: {total}")
    return total

def export_rels(ndjson_path="rels.ndjson.gz"):
    """
    Γράφει ανά γραμμή:
    {
      "eid": "<elementId(r)>",
      "type": "REL_TYPE",
      "props": { ... },
      "source": "<elementId(a)>",
      "target": "<elementId(b)>"
    }
    """
    q = """
    MATCH (a)-[r]->(b)
    RETURN elementId(r) AS eid, type(r) AS type, properties(r) AS props,
           elementId(a) AS source, elementId(b) AS target
    SKIP $skip LIMIT $limit
    """
    total = 0
    with driver.session() as sess, gzip.open(ndjson_path, "wt", encoding="utf-8") as out:
        skip = 0
        while True:
            recs = list(sess.run(q, skip=skip, limit=BATCH))
            if not recs:
                break
            for r in recs:
                out.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
            total += len(recs)
            skip += BATCH
            print(f"rels: {total}")
    return total

if __name__ == "__main__":
    n = export_nodes()
    r = export_rels()
    print(f"Done. Nodes: {n}, Rels: {r}")