#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ASD & Typical Gait Knowledge Graph Ingestion (STRICT + SID + PROGRESS + pid_num)
---------------------------------------------------------------------------------
Graph:
  (:Subject {sid, pid, pid_num, group})
      └─[:HAS_TRIAL]→ (:Trial {uid, trial_id, path})
            ├─[:HAS_FILE {kind}]→ (:File {uid, kind, path})
            └─[:HAS_FEATURE]→ (:FeatureValue {value})
                                └─[:OF_FEATURE]→ (:Feature {code, stat})

Κύριες αρχές:
- Subjects γίνονται MERGE με `sid = "<group>:<pid>"`.
- Για συμβατότητα με υπάρχον unique constraint στο `Subject.pid`, το `pid` γράφεται = `sid`.
- Προστίθεται `pid_num` (int) για τον αριθμητικό κωδικό συμμετέχοντα.
- Σκανάρονται ΜΟΝΟ:
    ASD:    Dataset/Autism/children with ASD/<digit>
    Typical:Dataset/Typical/<digit>
- Trials μόνο μέσα σε: augmentation/* και video/*
- Correlations (Spearman/Pearson) με thresholds από env.

Env (.env δίπλα στο script ή OS env):
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=palatiou
  DATASET_ROOT=/home/ilab/agent_sml_gait_neo4j/Dataset
  CORR_METHOD=spearman
  CORR_MIN_ABS_R=0.7
  CORR_MIN_PAIRS=20
  VERBOSE=1
  SKIP_FEATURES=0
  ONLY_CORR=0
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ---------- load .env explicitly (script dir) and also allow OS env ----------
HERE = Path(__file__).resolve().parent
load_dotenv(dotenv_path=HERE / ".env", override=False)
load_dotenv(override=False)  # also pick up OS env vars

# ---------- config ----------
ROOT = Path(os.getenv("DATASET_ROOT", "./Dataset")).expanduser()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

CORR_METHOD = os.getenv("CORR_METHOD", "spearman").strip().lower()
if CORR_METHOD not in {"pearson", "spearman"}:
    CORR_METHOD = "spearman"
CORR_MIN_ABS_R = float(os.getenv("CORR_MIN_ABS_R", "0.7"))
CORR_MIN_PAIRS = int(os.getenv("CORR_MIN_PAIRS", "20"))

VERBOSE = os.getenv("VERBOSE", "1") == "1"         # default verbose on
SKIP_FEATURES = os.getenv("SKIP_FEATURES", "0") == "1"
ONLY_CORR = os.getenv("ONLY_CORR", "0") == "1"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)

def uid(*parts) -> str:
    """Stable SHA1 UID from multiple parts."""
    m = hashlib.sha1()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"|")
    return m.hexdigest()

def is_trial_dir(p: Path) -> bool:
    """A directory is a trial if it contains at least one expected file type."""
    try:
        names = [n.lower() for n in os.listdir(p)]
    except Exception:
        return False
    return any(n.endswith("_2d.xlsx") for n in names) or \
           any(n.endswith("features.xlsx") for n in names) or \
           any(n.endswith(".avi") for n in names)

def read_features_xlsx(x: Path) -> pd.DataFrame:
    """Read features.xlsx robustly; return empty DataFrame on error."""
    try:
        df = pd.read_excel(x)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        log(f"[WARN] Failed to read {x}: {e}")
        return pd.DataFrame()

# ---------------- Constraints helper (idempotent) ----------------
def ensure_constraints():
    cypher = """
    CREATE CONSTRAINT subject_sid IF NOT EXISTS
    FOR (s:Subject) REQUIRE s.sid IS UNIQUE;

    CREATE CONSTRAINT trial_uid IF NOT EXISTS
    FOR (t:Trial) REQUIRE t.uid IS UNIQUE;

    CREATE CONSTRAINT file_uid IF NOT EXISTS
    FOR (f:File) REQUIRE f.uid IS UNIQUE;

    CREATE INDEX feature_code IF NOT EXISTS
    FOR (f:Feature) ON (f.code);
    """
    with driver.session() as s:
        for stmt in [x.strip() for x in cypher.strip().split(";") if x.strip()]:
            s.run(stmt)

# ---------------- Neo4j transactions ----------------
def upsert_subject_tx(tx, pid: str, group: str):
    # Make sid unique and pid compatible with old unique constraint on pid
    sid = f"{group}:{pid}"
    # Try to derive numeric pid (pid_num); if not int, leave None
    try:
        pid_num = int(pid)
    except Exception:
        pid_num = None

    tx.run("""
        MERGE (s:Subject {sid:$sid})
        ON CREATE SET
            s.pid       = $sid,     // keep legacy unique(pid) happy
            s.pid_num   = $pid_num, // numeric id kept separately
            s.group     = $group,
            s.createdAt = timestamp()
        ON MATCH SET
            s.group     = coalesce(s.group,$group),
            s.pid_num   = coalesce(s.pid_num,$pid_num)
    """, sid=sid, pid_num=pid_num, group=group)

def create_trial_tx(tx, pid: str, group: str, trial_id: str, trial_path: str, t_uid: str):
    sid = f"{group}:{pid}"
    tx.run("""
        MATCH (s:Subject {sid:$sid})
        MERGE (t:Trial {uid:$t_uid})
          ON CREATE SET t.trial_id=$trial_id, t.path=$trial_path, t.createdAt=timestamp()
        MERGE (s)-[:HAS_TRIAL]->(t)
    """, sid=sid, t_uid=t_uid, trial_id=trial_id, trial_path=str(trial_path))

def attach_files_batch_tx(tx, pid: str, group: str, trial_id: str, rows: List[Dict]):
    sid = f"{group}:{pid}"
    tx.run("""
        UNWIND $rows AS r
        MATCH (s:Subject {sid:$sid})-[:HAS_TRIAL]->(t:Trial {trial_id:$trial_id})
        MERGE (f:File {uid:r.f_uid})
          ON CREATE SET f.kind=r.kind, f.path=r.path, f.createdAt=timestamp()
        MERGE (t)-[:HAS_FILE {kind:r.kind}]->(f)
    """, sid=sid, trial_id=trial_id, rows=rows)

def attach_features_tx(tx, pid: str, group: str, trial_id: str, rows: List[Dict]):
    sid = f"{group}:{pid}"
    tx.run("""
        UNWIND $rows AS row
        MATCH (s:Subject {sid:$sid})-[:HAS_TRIAL]->(t:Trial {trial_id:$trial_id})
        MERGE (f:Feature {code:row.code})
          ON CREATE SET f.stat=row.stat
        CREATE (fv:FeatureValue {value:row.value})
        MERGE (t)-[:HAS_FEATURE]->(fv)
        MERGE (fv)-[:OF_FEATURE]->(f)
    """, sid=sid, trial_id=trial_id, rows=rows)

def upsert_feature_correlation_tx(tx, a: str, b: str, r: float, n: int,
                                  method: str, min_abs: float, min_pairs: int):
    a, b = sorted([a, b])
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

# ---------------- Scanner ----------------
def iter_participants(root: Path):
    """Yield tuples (group, pid, participant_dir) for strict folders only."""
    # ASD
    asd_root = root / "Autism" / "children with ASD"
    if asd_root.exists():
        for d in sorted(asd_root.iterdir(), key=lambda p: (not p.name.isdigit(), p.name)):
            if d.is_dir() and d.name.isdigit():
                yield ("ASD", d.name, d)
    # TD
    td_root = root / "Typical"
    if td_root.exists():
        for d in sorted(td_root.iterdir(), key=lambda p: (not p.name.isdigit(), p.name)):
            if d.is_dir() and d.name.isdigit():
                yield ("TD", d.name, d)

def scan_and_ingest(root: Path) -> List[Tuple[str, str, float]]:
    """Scan dataset, ingest graph, and return (trial_uid, feature_code, value) observations."""
    observations: List[Tuple[str, str, float]] = []
    parts = list(iter_participants(root))
    total = len(parts)
    log(f"[INFO] Participants discovered: {total}")

    for k, (group, pid, pdir) in enumerate(parts, 1):
        log(f"[{k}/{total}] Subject pid={pid} group={group}")

        # 0) Ensure Subject exists
        with driver.session() as sess:
            sess.execute_write(upsert_subject_tx, pid, group)

        # 1) Collect trial directories only under augmentation/* and video/*
        cand_dirs: List[Path] = []
        for sub in ("augmentation", "video"):
            base = pdir / sub
            if base.exists():
                cand_dirs += [d for d in base.rglob("*") if d.is_dir()]

        trials = [d for d in cand_dirs if is_trial_dir(d)]
        log(f"  └─ trials found: {len(trials)}")

        for tdir in trials:
            trial_id = tdir.name
            t_uid = uid(pid, group, trial_id, tdir)

            with driver.session() as sess:
                sess.execute_write(create_trial_tx, pid, group, trial_id, str(tdir), t_uid)

            files_rows: List[Dict] = []
            feature_rows: List[Dict] = []

            for f in tdir.iterdir():
                if not f.is_file():
                    continue
                fn = f.name.lower()
                kind = None
                if fn.endswith("_2d.xlsx"): kind = "2d"
                elif fn.endswith("features.xlsx"): kind = "features"
                elif fn.endswith(".avi"): kind = "video"
                elif fn.endswith(".xlsx"): kind = "raw"

                if kind:
                    files_rows.append({
                        "f_uid": uid(pid, group, trial_id, str(f), kind),
                        "kind": kind,
                        "path": str(f)
                    })

                if (not SKIP_FEATURES) and fn.endswith("features.xlsx"):
                    df = read_features_xlsx(f)
                    if df.empty:
                        continue
                    if "code" in df.columns and "value" in df.columns:
                        for _, r in df.iterrows():
                            code = str(r.get("code") or "").strip()
                            if not code:
                                continue
                            try:
                                val = float(r.get("value"))
                            except Exception:
                                continue
                            stat = str(r.get("stat", "value"))
                            feature_rows.append({"code": code, "value": val, "stat": stat})
                            observations.append((t_uid, code, val))
                    else:
                        # wide format: first row has values
                        for c in df.columns:
                            try:
                                val = float(df[c].iloc[0])
                            except Exception:
                                continue
                            code = str(c)
                            feature_rows.append({"code": code, "value": val, "stat": "value"})
                            observations.append((t_uid, code, val))

            # write in batches
            if files_rows:
                with driver.session() as sess:
                    sess.execute_write(attach_files_batch_tx, pid, group, trial_id, files_rows)
            if feature_rows:
                with driver.session() as sess:
                    sess.execute_write(attach_features_tx, pid, group, trial_id, feature_rows)

    return observations

# ---------------- Correlations ----------------
def build_feature_correlations(observations: List[Tuple[str, str, float]]):
    if not observations:
        log("[INFO] No observations; skip correlations.")
        return

    df = pd.DataFrame(observations, columns=["trial_uid", "feature_code", "value"])
    # wide matrix: trial x feature
    mat = df.pivot_table(index="trial_uid", columns="feature_code", values="value", aggfunc="mean")

    # pairwise correlation with min observations
    corr = mat.corr(method=CORR_METHOD, min_periods=CORR_MIN_PAIRS)

    # co-presence counts
    present = mat.notna().astype(np.uint8)
    co = present.T @ present

    feats = list(mat.columns)
    created = 0
    log(f"[INFO] Correlation method={CORR_METHOD} min_abs={CORR_MIN_ABS_R} min_pairs={CORR_MIN_PAIRS}")

    with driver.session() as sess:
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                fi, fj = feats[i], feats[j]
                n = int(co.loc[fi, fj]) if fi in co.index and fj in co.columns else 0
                if n < CORR_MIN_PAIRS:
                    continue
                r = corr.loc[fi, fj]
                if pd.isna(r) or abs(r) < CORR_MIN_ABS_R:
                    continue
                sess.execute_write(
                    upsert_feature_correlation_tx,
                    fi, fj, float(r), n, CORR_METHOD, CORR_MIN_ABS_R, CORR_MIN_PAIRS
                )
                created += 1

    log(f"[INFO] Correlation edges created/updated: {created}")

# ---------------- Main ----------------
if __name__ == "__main__":
    print(f"Scanning dataset root: {ROOT}", flush=True)

    if ONLY_CORR:
        print("[WARN] ONLY_CORR=1 not implemented to pull from DB; run full ingest.", flush=True)
        sys.exit(0)

    # Ensure constraints we depend on (idempotent)
    ensure_constraints()

    observations = scan_and_ingest(ROOT)
    print(f"[INFO] Observations collected: {len(observations)}", flush=True)

    if CORR_MIN_ABS_R >= 2 or CORR_MIN_PAIRS >= 9999:
        print("[INFO] Correlations disabled by thresholds.", flush=True)
    else:
        build_feature_correlations(observations)

    print("✅ Done.", flush=True)