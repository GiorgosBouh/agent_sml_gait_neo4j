#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroGait ASD — Knowledge Graph Ingest (CSV -> Neo4j) v3.1 (NO APOC)
- Γρηγορότερο ingest με BULK UNWIND
- Επιλογή value mode:
    * node  : (Sample)-[:HAS_VALUE]->(FeatureValue {value})-[:OF_FEATURE]->(Feature)
    * rel   : (Sample)-[:HAS_VALUE {value}]->(Feature)    <-- ΠΡΟΤΕΙΝΕΤΑΙ (πιο γρήγορο/ελαφρύ)
Usage:
  python kg_ingest.py \
    --uri bolt://localhost:7687 \
    --user neo4j --password palatiou \
    --csv "Final dataset.csv" --delim ";" \
    --value-mode rel --batch-size 200
"""

import argparse
import logging
import math
from tenacity import retry, stop_after_attempt, wait_fixed
from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm

CONSTRAINT_STMTS = [
    "CREATE CONSTRAINT cond_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT feature_code IF NOT EXISTS FOR (f:Feature) REQUIRE f.code IS UNIQUE",
    "CREATE CONSTRAINT subject_pid IF NOT EXISTS FOR (p:Subject) REQUIRE p.pid IS UNIQUE",
    "CREATE CONSTRAINT sample_id IF NOT EXISTS FOR (s:Sample) REQUIRE s.sample_id IS UNIQUE",
    "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
]

SEED_CONDITIONS = "UNWIND ['ASD','TD'] AS cname MERGE (:Condition {name:cname})"
SEED_DATASET = "MERGE (:Dataset {name:$dataset_name, url:$csv})"

CREATE_SUBJECT = """
MERGE (sub:Subject {pid:$pid})
WITH sub
MATCH (c:Condition {name:$cname})
MERGE (sub)-[:HAS_CONDITION]->(c)
"""

CREATE_SAMPLE = """
MATCH (sub:Subject {pid:$pid})
MERGE (s:Sample {sample_id:$sample_id})
SET s.row=$row, s.class=$class, s.trial_idx=$trial_idx
MERGE (sub)-[:HAS_SAMPLE]->(s)
WITH s
MERGE (d:Dataset {name:$dataset_name})
MERGE (s)-[:FROM_DATASET]->(d)
MERGE (prov:Provenance {repo:$repo, file:$file, lineNo:$row})
MERGE (s)-[:FROM_SOURCE]->(prov)
"""

MERGE_FEATURE = "MERGE (f:Feature {code:$code})"

# bulk create relationships with property (fast path)
BULK_VALUES_REL = """
UNWIND $rows AS r
MATCH (s:Sample {sample_id:r.sid})
MATCH (f:Feature {code:r.code})
MERGE (s)-[hv:HAS_VALUE]->(f)
SET hv.value = r.val
"""

# bulk create via FeatureValue node (classic)
BULK_VALUES_NODE = """
UNWIND $rows AS r
MATCH (s:Sample {sample_id:r.sid})
MATCH (f:Feature {code:r.code})
CREATE (s)-[:HAS_VALUE]->(fv:FeatureValue {value:r.val})
CREATE (fv)-[:OF_FEATURE]->(f)
"""

# enrichment
FEATURE_META_1 = """
MATCH (f:Feature)
SET f.stat = CASE
  WHEN f.code =~ '^(?i)mean[-\\s].*' THEN 'mean'
  WHEN f.code =~ '^(?i)variance[-\\s].*' THEN 'variance'
  WHEN f.code =~ '^(?i)std[-\\s].*' THEN 'std'
  ELSE f.stat END
"""
FEATURE_META_2 = """
MATCH (f:Feature)
SET f.side = CASE
  WHEN f.code =~ '(?i).*Left$'  THEN 'L'
  WHEN f.code =~ '(?i).*Right$' THEN 'R'
  WHEN f.code =~ '.*[A-Z]{3,}L$' THEN 'L'
  WHEN f.code =~ '.*[A-Z]{3,}R$' THEN 'R'
  ELSE coalesce(f.side,'NA') END,
    f.axis = CASE
      WHEN f.code CONTAINS '-x-' THEN 'x'
      WHEN f.code CONTAINS '-y-' THEN 'y'
      WHEN f.code CONTAINS '-z-' THEN 'z'
      ELSE coalesce(f.axis,'NA') END
"""
FEATURE_META_3 = """
MATCH (f:Feature)
SET f.family = CASE
  WHEN f.code =~ '(?i)^(mean|variance|std)[-\\s].*' AND f.code =~ '(?i).*(Knee|Ankle|Hip|Spine|Pelvis|Midspain|Midspine|Thigh|Shank|Foot)(Left|Right)?$' THEN 'coord'
  WHEN f.code IN ['StrLe','MaxStLe','MaxStWi','GaCT','StaT','SwiT','Velocity'] THEN 'spatiotemporal'
  WHEN f.code IN ['HaTiLPos','HaTiRPos'] THEN 'binary'
  WHEN f.code STARTS WITH 'Rom' THEN 'rom'
  ELSE coalesce(f.family,'other') END
"""
FEATURE_META_4 = """
MATCH (f:Feature)
SET f.unit = CASE
  WHEN f.family='coord' THEN CASE WHEN f.stat='variance' THEN 'm2' ELSE 'm' END
  WHEN f.family='spatiotemporal' AND f.code='Velocity' THEN 'm/s'
  WHEN f.family='spatiotemporal' AND f.code<>'Velocity' THEN 'ms'
  WHEN f.family='binary' THEN 'binary'
  WHEN f.family='rom' THEN 'deg'
  ELSE f.unit END
"""
FEATURE_META_5 = """
MATCH (f:Feature)
WITH f,
CASE
  WHEN f.code =~ '(?i).*Knee.*'  THEN 'Knee'
  WHEN f.code =~ '(?i).*Ankle.*' THEN 'Ankle'
  WHEN f.code =~ '(?i).*Hip.*'   THEN 'Hip'
  ELSE null END AS jname
FOREACH (_ IN CASE WHEN jname IS NULL THEN [] ELSE [1] END |
  MERGE (j:Joint {name:jname})
  MERGE (f)-[:ABOUT_JOINT]->(j)
  SET f.joint_guess=jname)
"""

CHECK_PARTICIPANTS = "MATCH (p:Subject) RETURN count(*) AS participants"
CHECK_BY_COND = "MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition) RETURN c.name AS condition, count(*) AS n ORDER BY condition"
CHECK_VALUES_NODE = "MATCH (:Sample)-[:HAS_VALUE]->(:FeatureValue) RETURN count(*) AS feature_values"
CHECK_VALUES_REL  = "MATCH (:Sample)-[hv:HAS_VALUE]->(:Feature) RETURN count(hv) AS feature_values"

def get_args():
    p = argparse.ArgumentParser(description="NeuroGait ASD — CSV to Neo4j KG (8 rows = 1 participant) v3.1 (NO APOC)")
    p.add_argument("--uri", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--delim", default=";")
    p.add_argument("--repo", default="GiorgosBouh/agent_sml_gait_neo4j")
    p.add_argument("--file", default="Final dataset.csv")
    p.add_argument("--dataset-name", default="NeuroGait-ASD-CSV")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--value-mode", choices=["node","rel"], default="rel")
    return p.parse_args()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def run(session, query, params=None):
    return session.run(query, params or {})

def coerce_float(v):
    """Ασφαλής μετατροπή, υποστηρίζει δεκαδικό κόμμα."""
    import math as _m
    import pandas as _pd
    if _pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        # αποφύγετε NaN
        return float(v) if not _m.isnan(float(v)) else None
    s = str(v).strip()
    if not s or s.lower() in ("nan","none","null"):
        return None
    s = s.replace(",", ".")
    try:
        x = float(s)
        return x if not _m.isnan(x) else None
    except Exception:
        return None

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger("kg_ingest_v31")

    # Load CSV
    log.info("Loading CSV via pandas: %s", args.csv)
    df = pd.read_csv(args.csv, delimiter=args.delim)
    df = df.reset_index(drop=True)
    df["lineNo"] = df.index + 2
    df["pid"] = df["lineNo"].apply(lambda ln: math.ceil((ln-1)/8))
    df["trial_idx"] = ((df["lineNo"] - 2) % 8) + 1
    df["cname"] = df["class"].map(lambda c: "ASD" if c=="A" else ("TD" if c=="T" else "UNKNOWN"))
    feature_cols = [c for c in df.columns if c not in ["class","lineNo","pid","trial_idx","cname"]]

    n_rows = df.shape[0]
    log.info("CSV shape: %s rows × %s cols", n_rows, df.shape[1])

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as sess:
            for stmt in CONSTRAINT_STMTS: run(sess, stmt).consume()
            run(sess, SEED_CONDITIONS).consume()
            run(sess, SEED_DATASET, {"dataset_name": args.dataset_name, "csv": args.csv}).consume()

            # Features
            log.info("Creating %d Feature nodes (distinct codes) ...", len(feature_cols))
            for code in tqdm(feature_cols, desc="Features"):
                run(sess, MERGE_FEATURE, {"code": code}).consume()

            # Subjects
            log.info("Creating Subject nodes ...")
            first_by_pid = df.groupby("pid").first(numeric_only=False)
            for pid, row in tqdm(first_by_pid.iterrows(), total=len(first_by_pid), desc="Subjects"):
                run(sess, CREATE_SUBJECT, {"pid": int(pid), "cname": row["cname"]}).consume()

            # Samples
            log.info("Creating Samples ...")
            for start in tqdm(range(0, n_rows, args.batch_size), desc="Samples"):
                chunk = df.iloc[start:start+args.batch_size]
                with sess.begin_transaction() as tx:
                    for _, r in chunk.iterrows():
                        tx.run(CREATE_SAMPLE, {
                            "pid": int(r["pid"]),
                            "sample_id": f"{r['pid']}-{int(r['trial_idx'])}-{int(r['lineNo'])}",
                            "row": int(r["lineNo"]),
                            "class": r["class"],
                            "trial_idx": int(r["trial_idx"]),
                            "dataset_name": args.dataset_name,
                            "repo": args.repo,
                            "file": args.file,
                        })
                    tx.commit()

            # Values fan-out (bulk UNWIND)
            log.info("Creating Values (bulk UNWIND, mode=%s) ...", args.value_mode)
            BULK_QUERY = BULK_VALUES_REL if args.value_mode == "rel" else BULK_VALUES_NODE
            for start in tqdm(range(0, n_rows, args.batch_size), desc="Values"):
                chunk = df.iloc[start:start+args.batch_size]
                rows = []
                for _, r in chunk.iterrows():
                    sid = f"{r['pid']}-{int(r['trial_idx'])}-{int(r['lineNo'])}"
                    for code in feature_cols:
                        val = coerce_float(r[code])
                        if val is not None:
                            rows.append({"sid": sid, "code": code, "val": val})
                if rows:
                    # σπάσε σε μικρότερα υπο-πακέτα για σταθερότητα
                    step = 5000
                    with sess.begin_transaction() as tx:
                        for i in range(0, len(rows), step):
                            tx.run(BULK_QUERY, {"rows": rows[i:i+step]})
                        tx.commit()

            # Enrichment
            for stmt in (FEATURE_META_1, FEATURE_META_2, FEATURE_META_3, FEATURE_META_4, FEATURE_META_5):
                run(sess, stmt).consume()

            # Checks
            p = run(sess, CHECK_PARTICIPANTS).single()
            bc = run(sess, CHECK_BY_COND).data()
            if args.value_mode == "rel":
                fv = run(sess, CHECK_VALUES_REL).single()
            else:
                fv = run(sess, CHECK_VALUES_NODE).single()

            log.info("Participants: %s", p["participants"])
            log.info("By condition: %s", bc)
            log.info("FeatureValues: %s", fv["feature_values"])
            logging.info("Done ✅")
    finally:
        driver.close()

if __name__ == "__main__":
    main()