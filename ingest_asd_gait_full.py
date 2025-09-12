"""
ASD & Typical Gait Knowledge Graph Ingestion (FULL)
---------------------------------------------------
Builds a rich Neo4j knowledge graph and a Feature-Feature correlation layer.

Graph (per participant/trial):
  (:Subject {pid, group})
      └─[:HAS_TRIAL]→ (:Trial {trial_id, path, uid})
            ├─[:HAS_FILE {kind}]→ (:File {path, kind, uid})
            └─[:HAS_FEATURE]→ (:FeatureValue {value})
                                └─[:OF_FEATURE]→ (:Feature {code, stat})

Extra (global, across trials):
  (:Feature)-[:CORRELATED_WITH {r, n, method, min_abs, min_pairs}]->(:Feature)

Groups ingested:
  - ASD: Dataset/Autism/children with ASD/[1..50]
  - TD : Dataset/Typical/[1..50]
Excluded:
  - Dataset/Autism/Severe level of ASD

Env (.env):
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password_here
  DATASET_ROOT=/home/ilab/agent_sml_gait_neo4j/Dataset
  # Optional correlation settings:
  CORR_METHOD=spearman           # 'pearson' or 'spearman' (default: spearman)
  CORR_MIN_ABS_R=0.7            # absolute r threshold (default: 0.7)
  CORR_MIN_PAIRS=20             # minimum paired observations (default: 20)
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

# --------------------------
# Load configuration
# --------------------------
load_dotenv()

ROOT = Path(os.getenv("DATASET_ROOT", "./Dataset")).expanduser()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

CORR_METHOD = os.getenv("CORR_METHOD", "spearman").strip().lower()
if CORR_METHOD not in {"pearson", "spearman"}:
    CORR_METHOD = "spearman"
CORR_MIN_ABS_R = float(os.getenv("CORR_MIN_ABS_R", "0.7"))
CORR_MIN_PAIRS = int(os.getenv("CORR_MIN_PAIRS", "20"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --------------------------
# Helpers
# --------------------------
def uid(*parts) -> str:
    """Stable SHA1 UID from multiple parts."""
    m = hashlib.sha1()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"|")
    return m.hexdigest()

def detect_group_from_path(path: Path) -> str:
    """Detect group from folder path."""
    p = path.as_posix().lower()
    if "children with asd" in p:
        return "ASD"
    if "/typical/" in p or p.endswith("/typical") or p.split("/")[-1] == "Typical".lower():
        return "TD"
    return "unknown"

def read_features_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """Read features.xlsx (robustly). Returns empty DF on failure."""
    try:
        df = pd.read_excel(xlsx_path)
        # Normalize columns
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"[WARN] Failed to read {xlsx_path}: {e}")
        return pd.DataFrame()

def is_trial_dir(p: Path) -> bool:
    """Heuristic: a trial dir contains at least one of the expected files."""
    try:
        names = [n.lower() for n in os.listdir(p)]
    except Exception:
        return False
    return any(n.endswith("_2d.xlsx") for n in names) or \
           any(n.endswith("features.xlsx") for n in names) or \
           any(n.endswith(".avi") for n in names)

# --------------------------
# Neo4j write transactions
# --------------------------
def upsert_subject_tx(tx, pid: str, group: str):
    tx.run("""
        MERGE (s:Subject {pid:$pid})
        ON CREATE SET s.group=$group, s.createdAt=timestamp()
        ON MATCH  SET s.group=coalesce(s.group,$group)
    """, pid=pid, group=group)

def create_trial_tx(tx, pid: str, trial_id: str, trial_path: str, t_uid: str):
    tx.run("""
        MATCH (s:Subject {pid:$pid})
        MERGE (t:Trial {uid:$t_uid})
          ON CREATE SET t.trial_id=$trial_id, t.path=$trial_path, t.createdAt=timestamp()
        MERGE (s)-[:HAS_TRIAL]->(t)
    """, pid=pid, t_uid=t_uid, trial_id=trial_id, trial_path=str(trial_path))

def attach_file_tx(tx, pid: str, trial_id: str, fpath: str, kind: str, f_uid: str):
    tx.run("""
        MATCH (s:Subject {pid:$pid})-[:HAS_TRIAL]->(t:Trial {trial_id:$trial_id})
        MERGE (f:File {uid:$f_uid})
          ON CREATE SET f.kind=$kind, f.path=$fpath, f.createdAt=timestamp()
        MERGE (t)-[:HAS_FILE {kind:$kind}]->(f)
    """, pid=pid, trial_id=trial_id, f_uid=f_uid, kind=kind, fpath=fpath)

def attach_features_tx(tx, pid: str, trial_id: str, rows: List[Dict]):
    tx.run("""
        UNWIND $rows AS row
        MATCH (s:Subject {pid:$pid})-[:HAS_TRIAL]->(t:Trial {trial_id:$trial_id})
        MERGE (f:Feature {code:row.code})
          ON CREATE SET f.stat=row.stat
        CREATE (fv:FeatureValue {value:row.value})
        MERGE (t)-[:HAS_FEATURE]->(fv)
        MERGE (fv)-[:OF_FEATURE]->(f)
    """, pid=pid, trial_id=trial_id, rows=rows)

def upsert_feature_correlation_tx(tx, f_code_a: str, f_code_b: str,
                                  r: float, n: int, method: str,
                                  min_abs: float, min_pairs: int):
    # Ensure directional uniqueness by ordering codes (a < b lexicographically)
    a, b = sorted([f_code_a, f_code_b])
    tx.run("""
        MERGE (fa:Feature {code:$a})
        MERGE (fb:Feature {code:$b})
        MERGE (fa)-[rel:CORRELATED_WITH]->(fb)
        ON CREATE SET rel.createdAt=timestamp()
        SET rel.r=$r, rel.n=$n, rel.method=$method,
            rel.min_abs=$min_abs, rel.min_pairs=$min_pairs,
            rel.updatedAt=timestamp()
    """, a=a, b=b, r=float(r), n=int(n), method=method,
         min_abs=float(min_abs), min_pairs=int(min_pairs))

# --------------------------
# Scan & Ingest
# --------------------------
def scan_and_ingest(root: Path) -> List[Tuple[str, str, float]]:
    """
    Scans dataset folders, ingests nodes/relationships,
    and returns a list of observations: (trial_uid, feature_code, value)
    """
    observations: List[Tuple[str, str, float]] = []

    # Level-1 group dirs: Autism, Typical
    for group_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        # Skip the "Severe level of ASD" branch entirely
        if "Severe level of ASD" in group_dir.name:
            print(f"[SKIP] {group_dir}")
            continue

        # Under Autism/children with ASD/* and Typical/* we expect participant folders
        for participant_dir in sorted(group_dir.rglob("*")):
            if not participant_dir.is_dir():
                continue

            # Participant folders typically named "1", "2", ..., "50"
            pid = participant_dir.name
            group = detect_group_from_path(participant_dir)

            # Upsert subject
            with driver.session() as session:
                session.execute_write(upsert_subject_tx, pid, group)

            # Trial directories nested under participant
            for trial_dir in participant_dir.rglob("*"):
                if not (trial_dir.is_dir() and is_trial_dir(trial_dir)):
                    continue

                trial_id = trial_dir.name
                t_uid = uid(pid, trial_id, trial_dir)

                # Create trial node and relation to subject
                with driver.session() as session:
                    session.execute_write(create_trial_tx, pid, trial_id, str(trial_dir), t_uid)

                # Attach files and parse features
                for f in trial_dir.iterdir():
                    if not f.is_file():
                        continue
                    fn = f.name.lower()
                    kind = None
                    if fn.endswith("_2d.xlsx"):
                        kind = "2d"
                    elif fn.endswith("features.xlsx"):
                        kind = "features"
                    elif fn.endswith(".avi"):
                        kind = "video"
                    elif fn.endswith(".xlsx"):
                        # beware: this also matches *_2d.xlsx / *features.xlsx but those already caught
                        kind = "raw"

                    if kind:
                        f_uid = uid(pid, trial_id, str(f), kind)
                        with driver.session() as session:
                            session.execute_write(attach_file_tx, pid, trial_id, str(f), kind, f_uid)

                    # Parse features to build FeatureValue + collect observations
                    if fn.endswith("features.xlsx"):
                        df = read_features_xlsx(f)
                        if df.empty:
                            continue

                        # Case A: tidy (code, value[, stat])
                        if "code" in df.columns and "value" in df.columns:
                            rows = []
                            for _, r in df.iterrows():
                                code = str(r.get("code"))
                                if not code or pd.isna(code):
                                    continue
                                try:
                                    val = float(r.get("value"))
                                except (TypeError, ValueError):
                                    continue
                                stat = str(r.get("stat", "value"))
                                rows.append({"code": code, "value": val, "stat": stat})
                                observations.append((t_uid, code, val))
                        else:
                            # Case B: wide (columns are features, row 0 has values)
                            rows = []
                            for c in df.columns:
                                try:
                                    val = float(df[c].iloc[0])
                                except Exception:
                                    continue
                                code = str(c)
                                rows.append({"code": code, "value": val, "stat": "value"})
                                observations.append((t_uid, code, val))

                        if rows:
                            with driver.session() as session:
                                session.execute_write(attach_features_tx, pid, trial_id, rows)

    return observations

# --------------------------
# Correlation Layer
# --------------------------
def build_feature_correlations(observations: List[Tuple[str, str, float]]):
    """
    observations: list of (trial_uid, feature_code, value)
    Builds a wide trial x feature matrix; computes pairwise correlations;
    writes CORRELATED_WITH edges between Feature nodes for |r| >= threshold & n >= min_pairs.
    """
    if not observations:
        print("[INFO] No observations collected; skipping correlation layer.")
        return

    df = pd.DataFrame(observations, columns=["trial_uid", "feature_code", "value"])
    # pivot to wide: rows=trial, cols=feature, values=value
    mat = df.pivot_table(index="trial_uid", columns="feature_code", values="value", aggfunc="mean")
    # Drop features with too few non-NaN values to allow decent correlation
    valid_counts = mat.notna().sum(axis=0)

    # Compute correlation matrix
    corr = mat.corr(method=CORR_METHOD, min_periods=CORR_MIN_PAIRS)

    # For each pair (i,j), compute n (overlap count) and r; create edge if passes thresholds
    features = list(mat.columns)
    n_features = len(features)
    print(f"[INFO] Correlation method={CORR_METHOD}, min_abs={CORR_MIN_ABS_R}, min_pairs={CORR_MIN_PAIRS}")
    print(f"[INFO] Candidate features for correlation: {n_features}")

    # Precompute pairwise counts (number of trials where both features present)
    # We can use boolean matrices to count co-valid entries
    present = mat.notna().astype(np.uint8)
    # pairwise co-occurrence using dot-product trick
    co_counts = present.T @ present  # DataFrame with indices = features

    created = 0
    with driver.session() as session:
        for i in range(n_features):
            fi = features[i]
            for j in range(i+1, n_features):
                fj = features[j]
                n = int(co_counts.loc[fi, fj]) if (fi in co_counts.index and fj in co_counts.columns) else 0
                if n < CORR_MIN_PAIRS:
                    continue
                r = corr.loc[fi, fj]
                if pd.isna(r) or abs(r) < CORR_MIN_ABS_R:
                    continue
                session.execute_write(
                    upsert_feature_correlation_tx,
                    fi, fj, float(r), n, CORR_METHOD, CORR_MIN_ABS_R, CORR_MIN_PAIRS
                )
                created += 1

    print(f"[INFO] Feature correlation edges created/updated: {created}")

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    if not ROOT.exists():
        raise SystemExit(f"Dataset root not found: {ROOT}")

    print(f"Scanning dataset root: {ROOT}")
    observations = scan_and_ingest(ROOT)
    print(f"[INFO] Observations collected: {len(observations)}")

    print("Building feature-feature correlation layer...")
    build_feature_correlations(observations)

    print("✅ Done. Knowledge Graph + Correlation layer built.")