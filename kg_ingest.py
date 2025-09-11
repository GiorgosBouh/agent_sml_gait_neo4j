#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroGait ASD — Knowledge Graph Ingest (CSV -> Neo4j) v3 (NO APOC)
- Διαβάζει το CSV στην Python (pandas)
- 1 participant per 8 consecutive rows (pid = ceil(row_index/8))
- Δημιουργεί: Subject, Condition, Sample, Feature, FeatureValue
- Εμπλουτίζει Feature metadata (stat/side/axis/family/unit/plane/joint_guess)
- Batching για ταχύτερο ingest

Usage:
  python kg_ingest.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password palatiou \
    --csv "Final dataset.csv" \
    --delim ";"
    [--dataset-name NeuroGait-ASD-CSV]
    [--repo GiorgosBouh/agent_sml_gait_neo4j]
    [--file "Final dataset.csv"]
"""

import argparse
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm
import math

# ---- Cypher (single-statement friendly) --------------------------------------

CONSTRAINT_STMTS = [
    "CREATE CONSTRAINT cond_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT feature_code IF NOT EXISTS FOR (f:Feature) REQUIRE f.code IS UNIQUE",
    "CREATE CONSTRAINT subject_pid IF NOT EXISTS FOR (p:Subject) REQUIRE p.pid IS UNIQUE",
    "CREATE CONSTRAINT sample_id IF NOT EXISTS FOR (s:Sample) REQUIRE s.sample_id IS UNIQUE",
    "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
    "CREATE INDEX feature_group IF NOT EXISTS FOR (f:Feature) ON (f.group)",
    "CREATE INDEX feature_family IF NOT EXISTS FOR (f:Feature) ON (f.family)",
    "CREATE INDEX feature_joint_guess IF NOT EXISTS FOR (f:Feature) ON (f.joint_guess)",
]

SEED_CONDITIONS = "UNWIND ['ASD','TD'] AS cname MERGE (:Condition {name:cname})"
SEED_DATASET = "MERGE (:Dataset {name:$dataset_name, url:$csv})"

CREATE_SUBJECT = """
MERGE (sub:Subject {pid:$pid})
WITH sub
MATCH (c:Condition {name:$cname})
MERGE (sub)-[:HAS_CONDITION]->(c)
RETURN sub
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
RETURN s
"""

MERGE_FEATURE = """
MERGE (f:Feature {code:$code})
RETURN f
"""

CREATE_VALUE_REL = """
MATCH (s:Sample {sample_id:$sample_id})
MATCH (f:Feature {code:$code})
MERGE (s)-[:HAS_VALUE]->(fv:FeatureValue {value:$value})
MERGE (fv)-[:OF_FEATURE]->(f)
"""

# Enrichment in 5 single statements (no semicolons)
FEATURE_META_1 = """
MATCH (f:Feature)
SET f.stat = CASE
  WHEN f.code STARTS WITH 'mean ' THEN 'mean'
  WHEN f.code STARTS WITH 'variance ' THEN 'variance'
  WHEN f.code STARTS WITH 'std ' THEN 'std'
  ELSE f.stat END
"""

FEATURE_META_2 = """
MATCH (f:Feature)
SET f.side = CASE
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
  WHEN f.code =~ '(?i)^(mean|variance|std)\\s+(HES|SPE|SHW|ELH|THH|SPK|HIA|KNF).*' THEN 'angle'
  WHEN f.code =~ '(?i)^(mean|variance|std)-[xyz]-.*' THEN 'coord'
  WHEN f.code IN ['StrLe','MaxStLe','MaxStWi','GaCT','StaT','SwiT','Velocity'] THEN 'spatiotemporal'
  WHEN f.code IN ['HaTiLPos','HaTiRPos'] THEN 'binary'
  WHEN f.code STARTS WITH 'Rom' THEN 'rom'
  ELSE coalesce(f.family,'other') END
"""

FEATURE_META_4 = """
MATCH (f:Feature)
SET f.unit = CASE
  WHEN f.family='angle' THEN 'deg'
  WHEN f.family='coord' THEN CASE WHEN f.stat='variance' THEN 'm2' ELSE 'm' END
  WHEN f.family='spatiotemporal' AND f.code='Velocity' THEN 'm/s'
  WHEN f.family='spatiotemporal' AND f.code<>'Velocity' THEN 'ms'
  WHEN f.family='binary' THEN 'binary'
  WHEN f.family='rom' THEN 'deg'
  ELSE f.unit END
"""

FEATURE_META_5 = """
MATCH (f:Feature) WHERE f.family='angle'
SET f.plane = coalesce(f.plane,'Sagittal')
WITH f,
CASE
  WHEN f.code =~ '(?i).*HIAN(L|R).*' THEN 'Knee'
  WHEN f.code =~ '(?i).*KNFO(L|R).*' THEN 'Ankle'
  WHEN f.code =~ '(?i).*SPKN(L|R).*' THEN 'Hip'
  ELSE null END AS jname
FOREACH (_ IN CASE WHEN jname IS NULL THEN [] ELSE [1] END |
  MERGE (j:Joint {name:jname})
  MERGE (f)-[:ABOUT_JOINT]->(j)
  SET f.joint_guess=jname)
"""

CHECK_PARTICIPANTS = "MATCH (p:Subject) RETURN count(*) AS participants"
CHECK_BY_COND = "MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition) RETURN c.name AS condition, count(*) AS n ORDER BY condition"
CHECK_VALUES = "MATCH (:Sample)-[:HAS_VALUE]->(:FeatureValue) RETURN count(*) AS feature_values"

# ---- Helpers -----------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="NeuroGait ASD — CSV to Neo4j KG (8 rows = 1 participant) v3 (NO APOC)")
    p.add_argument("--uri", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--csv", required=True, help="CSV local path OR URL (pandas can read both)")
    p.add_argument("--delim", default=";", help="CSV delimiter (default ';')")
    p.add_argument("--repo", default="GiorgosBouh/agent_sml_gait_neo4j")
    p.add_argument("--file", default="Final dataset.csv")
    p.add_argument("--dataset-name", default="NeuroGait-ASD-CSV")
    p.add_argument("--batch-size", type=int, default=2000, help="rows per transaction for values fan-out")
    return p.parse_args()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def run(session, query, params=None):
    return session.run(query, params or {})

# ---- Main --------------------------------------------------------------------

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger("kg_ingest_v3")

    # Load CSV via pandas (semicolon-delimited)
    log.info("Loading CSV via pandas: %s", args.csv)
    df = pd.read_csv(args.csv, delimiter=args.delim)
    n_rows = df.shape[0]
    log.info("CSV shape: %s rows × %s cols", n_rows, df.shape[1])

    # Compute participant id (pid) & trial_idx per row
    # lineNo (1-based) = dataframe index + 2 (header row counts as 1 in earlier convention)
    df = df.reset_index(drop=True)
    df["lineNo"] = df.index + 2
    df["pid"] = (df["lineNo"].apply(lambda ln: math.ceil((ln-1)/8)))  # rows 2-9 => pid=1, 10-17 => 2, ...
    df["trial_idx"] = ((df["lineNo"] - 2) % 8) + 1

    # Map class to Condition name
    def map_class(c):
        return "ASD" if c == "A" else "TD" if c == "T" else "UNKNOWN"
    df["cname"] = df["class"].map(map_class)

    # Prepare
    feature_cols = [c for c in df.columns if c not in ["class", "lineNo", "pid", "trial_idx", "cname"]]

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as sess:
            # Constraints / Indexes
            for stmt in CONSTRAINT_STMTS:
                run(sess, stmt).consume()

            # Seed Condition & Dataset
            run(sess, SEED_CONDITIONS).consume()
            run(sess, SEED_DATASET, {"dataset_name": args.dataset_name, "csv": args.csv}).consume()

            # Create all Feature nodes once
            log.info("Creating %d Feature nodes (distinct codes) ...", len(feature_cols))
            for code in tqdm(feature_cols, desc="Features"):
                run(sess, MERGE_FEATURE, {"code": code}).consume()

            # Create Subjects once (distinct pid) and link to Condition (per pid majority or first row)
            log.info("Creating Subject nodes ...")
            # Take first class per pid (dataset grouped by 8 makes it consistent)
            first_by_pid = df.groupby("pid").first(numeric_only=False)
            for pid, row in tqdm(first_by_pid.iterrows(), total=len(first_by_pid), desc="Subjects"):
                run(sess, CREATE_SUBJECT, {"pid": int(pid), "cname": row["cname"]}).consume()

            # Create Samples & Values in batches
            log.info("Creating Samples & Values (batched) ...")
            B = args.batch_size
            for start in tqdm(range(0, n_rows, B), desc="Batches"):
                chunk = df.iloc[start:start+B]
                # Samples
                for _, r in chunk.iterrows():
                    params = {
                        "pid": int(r["pid"]),
                        "sample_id": f"{r['pid']}-{int(r['trial_idx'])}-{int(r['lineNo'])}",
                        "row": int(r["lineNo"]),
                        "class": r["class"],
                        "trial_idx": int(r["trial_idx"]),
                        "dataset_name": args.dataset_name,
                        "repo": args.repo,
                        "file": args.file,
                    }
                    run(sess, CREATE_SAMPLE, params).consume()

                # Values fan-out
                # Build small batches of (sample_id, code, value)
                value_params = []
                for _, r in chunk.iterrows():
                    sample_id = f"{r['pid']}-{int(r['trial_idx'])}-{int(r['lineNo'])}"
                    for code in feature_cols:
                        v = r[code]
                        if pd.isna(v) or v == "":
                            continue
                        try:
                            val = float(v)
                        except Exception:
                            continue
                        value_params.append({"sample_id": sample_id, "code": code, "value": val})

                # Commit values in sub-batches to avoid huge transactions
                step = 5000
                for i in range(0, len(value_params), step):
                    sub = value_params[i:i+step]
                    # Use explicit transaction for speed
                    with sess.begin_transaction() as tx:
                        for vp in sub:
                            tx.run(CREATE_VALUE_REL, vp)
                        tx.commit()

            # Enrich features (5 single statements)
            for stmt in [FEATURE_META_1, FEATURE_META_2, FEATURE_META_3, FEATURE_META_4, FEATURE_META_5]:
                run(sess, stmt).consume()

            # Checks
            p = run(sess, CHECK_PARTICIPANTS).single()
            bc = run(sess, CHECK_BY_COND).data()
            fv = run(sess, CHECK_VALUES).single()
            log.info("Participants: %s", p["participants"])
            log.info("By condition: %s", bc)
            log.info("FeatureValues: %s", fv["feature_values"])

        logging.info("Done ✅")
    finally:
        driver.close()


if __name__ == "__main__":
    main()