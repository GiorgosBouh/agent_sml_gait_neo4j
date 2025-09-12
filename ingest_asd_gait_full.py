#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ASD & Typical Gait Knowledge Graph Ingestion (STRICT + PROGRESS)
- ΜΟΝΟ participants:
    ASD:    Dataset/Autism/children with ASD/<digit>
    Typical:Dataset/Typical/<digit>
- Trials ΜΟΝΟ μέσα σε: augmentation/* και video/*
- Progress prints για να βλέπεις εξέλιξη
- Correlation layer προαιρετικό μέσω thresholds στο .env

Env (.env ή inline):
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=palatiou
  DATASET_ROOT=/home/ilab/agent_sml_gait_neo4j/Dataset
  CORR_METHOD=spearman
  CORR_MIN_ABS_R=0.7
  CORR_MIN_PAIRS=20
"""

import os, sys, hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

# -------- load env explicitly next to script (fallback: cwd) --------
HERE = Path(__file__).resolve().parent
load_dotenv(dotenv_path=HERE / ".env", override=False)
load_dotenv(override=False)  # also allow inline env / cwd

ROOT = Path(os.getenv("DATASET_ROOT", "./Dataset")).expanduser()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

CORR_METHOD = os.getenv("CORR_METHOD", "spearman").strip().lower()
if CORR_METHOD not in {"pearson", "spearman"}: CORR_METHOD = "spearman"
CORR_MIN_ABS_R = float(os.getenv("CORR_MIN_ABS_R", "0.7"))
CORR_MIN_PAIRS = int(os.getenv("CORR_MIN_PAIRS", "20"))

# Optional controls
VERBOSE = os.getenv("VERBOSE", "1") == "1"  # default on (prints progress)
SKIP_FEATURES = os.getenv("SKIP_FEATURES", "0") == "1"
ONLY_CORR = os.getenv("ONLY_CORR", "0") == "1"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)

def uid(*parts) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update(str(p).encode("utf-8"))
        m.update(b"|")
    return m.hexdigest()

def is_trial_dir(p: Path) -> bool:
    try:
        names = [n.lower() for n in os.listdir(p)]
    except Exception:
        return False
    return any(n.endswith("_2d.xlsx") for n in names) or \
           any(n.endswith("features.xlsx") for n in names) or \
           any(n.endswith(".avi") for n in names)

def read_features_xlsx(x: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(x)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        log(f"[WARN] Failed to read {x}: {e}")
        return pd.DataFrame()

# --------------- Neo4j TX ---------------
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

def attach_files_batch_tx(tx, pid: str, trial_id: str, rows: List[Dict]):
    tx.run("""
        UNWIND $rows AS r
        MATCH (s:Subject {pid:$pid})-[:HAS_TRIAL]->(t:Trial {trial_id:$trial_id})
        MERGE (f:File {uid:r.f_uid})
          ON CREATE SET f.kind=r.kind, f.path=r.path, f.createdAt=timestamp()
        MERGE (t)-[:HAS_FILE {kind:r.kind}]->(f)
    """, pid=pid, trial_id=trial_id, rows=rows)

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

# --------------- Scanner ---------------
def iter_participants(root: Path):
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
    observations: List[Tuple[str, str, float]] = []
    parts = list(iter_participants(root))
    total = len(parts)
    log(f"[INFO] Participants discovered: {total}")

    for k, (group, pid, pdir) in enumerate(parts, 1):
        log(f"[{k}/{total}] Subject pid={pid} group={group}")

        # Subject
        with driver.session() as sess:
            sess.execute_write(upsert_subject_tx, pid, group)

        # Candidate trial dirs only under augmentation/* and video/*
        cand_dirs: List[Path] = []
        for sub in ("augmentation", "video"):
            base = pdir / sub
            if base.exists():
                cand_dirs += [d for d in base.rglob("*") if d.is_dir()]

        trials = [d for d in cand_dirs if is_trial_dir(d)]
        log(f"  └─ trials found: {len(trials)}")

        for tdir in trials:
            trial_id = tdir.name
            t_uid = uid(pid, trial_id, tdir)

            with driver.session() as sess:
                sess.execute_write(create_trial_tx, pid, trial_id, str(tdir), t_uid)

            # Batch files
            files_rows = []
            feature_rows = []

            for f in tdir.iterdir():
                if not f.is_file(): continue
                fn = f.name.lower()
                kind = None
                if fn.endswith("_2d.xlsx"): kind = "2d"
                elif fn.endswith("features.xlsx"): kind = "features"
                elif fn.endswith(".avi"): kind = "video"
                elif fn.endswith(".xlsx"): kind = "raw"

                if kind:
                    files_rows.append({"f_uid": uid(pid, trial_id, str(f), kind),
                                       "kind": kind, "path": str(f)})

                if (not SKIP_FEATURES) and fn.endswith("features.xlsx"):
                    df = read_features_xlsx(f)
                    if df.empty: continue
                    if "code" in df.columns and "value" in df.columns:
                        for _, r in df.iterrows():
                            code = str(r.get("code") or "").strip()
                            if not code: continue
                            try: val = float(r.get("value"))
                            except: continue
                            stat = str(r.get("stat", "value"))
                            feature_rows.append({"code": code, "value": val, "stat": stat})
                            observations.append((t_uid, code, val))
                    else:
                        for c in df.columns:
                            try: val = float(df[c].iloc[0])
                            except: continue
                            code = str(c)
                            feature_rows.append({"code": code, "value": val, "stat": "value"})
                            observations.append((t_uid, code, val))

            # write batches
            if files_rows:
                with driver.session() as sess:
                    sess.execute_write(attach_files_batch_tx, pid, trial_id, files_rows)
            if feature_rows:
                with driver.session() as sess:
                    sess.execute_write(attach_features_tx, pid, trial_id, feature_rows)

    return observations

# --------------- Correlations ---------------
def build_feature_correlations(observations: List[Tuple[str, str, float]]):
    if not observations:
        log("[INFO] No observations; skip correlations.")
        return
    df = pd.DataFrame(observations, columns=["trial_uid", "feature_code", "value"])
    mat = df.pivot_table(index="trial_uid", columns="feature_code", values="value", aggfunc="mean")
    corr = mat.corr(method=CORR_METHOD, min_periods=CORR_MIN_PAIRS)

    present = mat.notna().astype(np.uint8)
    co = present.T @ present
    feats = list(mat.columns)
    created = 0
    log(f"[INFO] Correlation method={CORR_METHOD} min_abs={CORR_MIN_ABS_R} min_pairs={CORR_MIN_PAIRS}")

    with driver.session() as sess:
        for i in range(len(feats)):
            for j in range(i+1, len(feats)):
                fi, fj = feats[i], feats[j]
                n = int(co.loc[fi, fj]) if fi in co.index and fj in co.columns else 0
                if n < CORR_MIN_PAIRS: continue
                r = corr.loc[fi, fj]
                if pd.isna(r) or abs(r) < CORR_MIN_ABS_R: continue
                sess.execute_write(upsert_feature_correlation_tx,
                                   fi, fj, float(r), n, CORR_METHOD,
                                   CORR_MIN_ABS_R, CORR_MIN_PAIRS)
                created += 1
    log(f"[INFO] Correlation edges created/updated: {created}")

# --------------- Main ---------------
if __name__ == "__main__":
    print(f"Scanning dataset root: {ROOT}", flush=True)
    if ONLY_CORR:
        # (προαιρετικό) αν έχεις ήδη φορτώσει παρατηρήσεις και θες μόνο correlations,
        # θα έπρεπε να τις ξαναδιαβάσουμε από DB. Εδώ κρατάμε το απλό path: κάνε full run.
        print("[WARN] ONLY_CORR=1 not implemented to pull from DB; run full ingest.", flush=True)
        sys.exit(0)

    observations = scan_and_ingest(ROOT)
    print(f"[INFO] Observations collected: {len(observations)}", flush=True)

    if CORR_MIN_ABS_R >= 2 or CORR_MIN_PAIRS >= 9999:
        print("[INFO] Correlations disabled by thresholds.", flush=True)
    else:
        build_feature_correlations(observations)

    print("✅ Done.", flush=True)