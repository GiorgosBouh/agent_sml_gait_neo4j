#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroGait ASD — Knowledge Graph Ingest (CSV -> Neo4j) v2
- 1 participant per 8 consecutive rows (pid = ceil(lineNo/8))
- Subject(pid) -> HAS_SAMPLE -> Sample(row, class, trial_idx)
- Sample -> HAS_VALUE -> FeatureValue(value) -> OF_FEATURE -> Feature(code, rich metadata)
- Feature -> ABOUT_JOINT -> Joint (when detectable)
- Sample -> FROM_DATASET -> Dataset(name, url)
- Sample -> FROM_SOURCE  -> Provenance(repo, file, lineNo)

Usage:
  python kg_ingest.py \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password palatiou \
    --csv "https://raw.githubusercontent.com/GiorgosBouh/agent_sml_gait_neo4j/main/Final%20dataset.csv" \
    --delim ";" \
    --repo "GiorgosBouh/agent_sml_gait_neo4j" \
    --file "Final dataset.csv"

Requires:
  - Neo4j 5.x
  - APOC (Core) enabled
"""

import argparse
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from neo4j import GraphDatabase


CONSTRAINTS = """
CREATE CONSTRAINT cond_name IF NOT EXISTS
FOR (c:Condition) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT feature_code IF NOT EXISTS
FOR (f:Feature) REQUIRE f.code IS UNIQUE;

CREATE CONSTRAINT subject_pid IF NOT EXISTS
FOR (p:Subject) REQUIRE p.pid IS UNIQUE;

CREATE CONSTRAINT sample_id IF NOT EXISTS
FOR (s:Sample) REQUIRE s.sample_id IS UNIQUE;

CREATE CONSTRAINT dataset_name IF NOT EXISTS
FOR (d:Dataset) REQUIRE d.name IS UNIQUE;

CREATE INDEX feature_group IF NOT EXISTS
FOR (f:Feature) ON (f.group);

CREATE INDEX feature_family IF NOT EXISTS
FOR (f:Feature) ON (f.family);

CREATE INDEX feature_joint_guess IF NOT EXISTS
FOR (f:Feature) ON (f.joint_guess);
"""

SEED_NODES = """
UNWIND ["ASD","TD"] AS cname
MERGE (:Condition {name:cname});

MERGE (:Dataset {name:$dataset_name, url:$csv});
"""

# Ingest with grouping by 8 rows using apoc.load.csv to access 'lineNo'
INGEST = """
CALL apoc.load.csv($csv, {sep:$delim, header:true, skip:1}) YIELD map, lineNo
WITH map, lineNo, toInteger(ceil(lineNo / 8.0)) AS pid
// Subject per block-of-8
MERGE (sub:Subject {pid: pid})
WITH sub, map, lineNo, pid,
CASE map.class WHEN 'A' THEN 'ASD' WHEN 'T' THEN 'TD' ELSE 'UNKNOWN' END AS cname
MERGE (c:Condition {name:cname})
MERGE (sub)-[:HAS_CONDITION]->(c)

// Sample per row
MERGE (s:Sample {sample_id: apoc.create.uuid()})
SET s.row = lineNo,
    s.class = map.class,
    s.trial_idx = ((lineNo - 1) % 8) + 1
MERGE (sub)-[:HAS_SAMPLE]->(s)

// Dataset + Provenance
MERGE (d:Dataset {name:$dataset_name})
WITH s, map, lineNo, d
MERGE (s)-[:FROM_DATASET]->(d)
MERGE (prov:Provenance {repo:$repo, file:$file, lineNo: lineNo})
MERGE (s)-[:FROM_SOURCE]->(prov)

// Fan-out features
WITH s, map
UNWIND keys(map) AS k
WITH s, k, map[k] AS v
WHERE k <> 'class' AND v IS NOT NULL AND v <> ""
MERGE (f:Feature {code: k})
MERGE (s)-[:HAS_VALUE]->(fv:FeatureValue {value: toFloat(v)})
MERGE (fv)-[:OF_FEATURE]->(f);
"""

# Rich metadata on Feature: stat/side/axis/family/plane/joint_guess
FEATURE_METADATA = """
// stat: mean / variance / std
MATCH (f:Feature)
SET f.stat = CASE
  WHEN f.code STARTS WITH "mean " THEN "mean"
  WHEN f.code STARTS WITH "variance " THEN "variance"
  WHEN f.code STARTS WITH "std " THEN "std"
  ELSE f.stat
END;

// side: L/R from suffix (…L / …R), when present
MATCH (f:Feature)
SET f.side = CASE
  WHEN f.code =~ '.*[A-Z]{3,}L$' THEN 'L'
  WHEN f.code =~ '.*[A-Z]{3,}R$' THEN 'R'
  ELSE coalesce(f.side, 'NA')
END;

// axis from -x- / -y- / -z-
MATCH (f:Feature)
SET f.axis = CASE
  WHEN f.code CONTAINS '-x-' THEN 'x'
  WHEN f.code CONTAINS '-y-' THEN 'y'
  WHEN f.code CONTAINS '-z-' THEN 'z'
  ELSE coalesce(f.axis, 'NA')
END;

// family + unit
MATCH (f:Feature)
SET f.family = CASE
  WHEN f.code =~ '(?i)^(mean|variance|std)\\s+(HES|SPE|SHW|ELH|THH|SPK|HIA|KNF).*' THEN 'angle'
  WHEN f.code =~ '(?i)^(mean|variance|std)-[xyz]-.*' THEN 'coord'
  WHEN f.code IN ['StrLe','MaxStLe','MaxStWi','GaCT','StaT','SwiT','Velocity'] THEN 'spatiotemporal'
  WHEN f.code IN ['HaTiLPos','HaTiRPos'] THEN 'binary'
  WHEN f.code STARTS WITH 'Rom' THEN 'rom'
  ELSE coalesce(f.family,'other')
END,
f.unit = CASE
  WHEN f.family = 'angle' THEN 'deg'
  WHEN f.family = 'coord' THEN CASE WHEN f.stat='variance' THEN 'm2' ELSE 'm' END
  WHEN f.family = 'spatiotemporal' AND f.code='Velocity' THEN 'm/s'
  WHEN f.family = 'spatiotemporal' AND f.code<>'Velocity' THEN 'ms'
  WHEN f.family = 'binary' THEN 'binary'
  WHEN f.family = 'rom' THEN 'deg'
  ELSE f.unit
END;

// plane (heuristic: main angle families are sagittal here)
MATCH (f:Feature)
WHERE f.family='angle'
SET f.plane = coalesce(f.plane,'Sagittal');

// joint_guess from code patterns
MATCH (f:Feature)
WITH f,
CASE
  WHEN f.code =~ '(?i).*HIAN(L|R).*' THEN 'Knee'
  WHEN f.code =~ '(?i).*KNFO(L|R).*' THEN 'Ankle'
  WHEN f.code =~ '(?i).*SPKN(L|R).*' THEN 'Hip'
  ELSE null
END AS jname
FOREACH (_ IN CASE WHEN jname IS NULL THEN [] ELSE [1] END |
  MERGE (j:Joint {name:jname})
  MERGE (f)-[:ABOUT_JOINT]->(j)
  SET f.joint_guess = jname
);
"""

CHECKS = {
    "participants": """
        MATCH (p:Subject) RETURN count(*) AS participants;
    """,
    "by_condition": """
        MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
        RETURN c.name AS condition, count(*) AS n ORDER BY condition;
    """,
    "sample_rows": """
        MATCH (p:Subject)-[:HAS_SAMPLE]->(s:Sample)
        WITH p.pid AS pid, min(s.row) AS start, max(s.row) AS finish
        RETURN pid, start, finish ORDER BY pid LIMIT 10;
    """,
    "values_count": """
        MATCH (:Sample)-[:HAS_VALUE]->(:FeatureValue) RETURN count(*) AS feature_values;
    """,
    "features_overview": """
        MATCH (f:Feature)
        WITH f.family AS fam, f.stat AS stat, f.side AS side, count(*) AS n
        RETURN fam, stat, side, n ORDER BY n DESC LIMIT 20;
    """
}

def get_args():
    p = argparse.ArgumentParser(description="NeuroGait ASD — CSV to Neo4j KG (8 rows = 1 participant) v2")
    p.add_argument("--uri", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--csv", required=True, help="Raw GitHub URL or http(s) path")
    p.add_argument("--delim", default=";", help="CSV delimiter (default ';')")
    p.add_argument("--repo", default="GiorgosBouh/agent_sml_gait_neo4j")
    p.add_argument("--file", default="Final dataset.csv")
    p.add_argument("--dataset-name", default="NeuroGait-ASD-CSV")
    p.add_argument("--skip-ingest", action="store_true")
    return p.parse_args()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def run(session, query, params=None):
    return session.run(query, params or {}).data()

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    log = logging.getLogger("kg_ingest_v2")

    log.info("Connecting to Neo4j: %s", args.uri)
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as sess:
            log.info("Creating constraints / indexes ...")
            run(sess, CONSTRAINTS)

            log.info("Seeding base nodes (Condition, Dataset) ...")
            run(sess, SEED_NODES, {"dataset_name": args.dataset_name, "csv": args.csv})

            if not args.skip_ingest:
                log.info("Ingesting CSV -> Graph (APOC) ...")
                run(sess, INGEST, {
                    "csv": args.csv,
                    "delim": args.delim,
                    "dataset_name": args.dataset_name,
                    "repo": args.repo,
                    "file": args.file
                })

                log.info("Enriching Feature metadata (stat/side/axis/family/plane/joint_guess + Joint links) ...")
                run(sess, FEATURE_METADATA)

            log.info("Running quick checks ...")
            for name, q in CHECKS.items():
                res = run(sess, q)
                log.info("%s: %s", name, res)

        log.info("Done ✅")
    finally:
        driver.close()

if __name__ == "__main__":
    main()