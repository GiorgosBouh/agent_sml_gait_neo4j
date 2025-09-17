#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit UI for ASD Gait — Neo4j Explorer (NL ↔ Cypher) with JSON-backed Catalog

Schema:
  (:Subject {pid,sid})-[:HAS_TRIAL]->(:Trial {uid})
  (:Trial)-[:HAS_FILE]->(:File {uid,kind})
  (:Trial)-[:HAS_FEATURE]->(:FeatureValue {value})-[:OF_FEATURE]->(:Feature {code,stat,joint_guess})

Run:
  streamlit run agent_app.py --server.port 8504 --server.address 0.0.0.0
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
from neo4j import GraphDatabase, basic_auth, Driver

from nl2cypher import NL2Cypher, GraphCatalog

# ----------------------- Streamlit Config -----------------------
st.set_page_config(page_title="ASD Gait — Neo4j Explorer", page_icon="🦶", layout="wide")
st.title("🧠 ASD Gait — NL ↔ Cypher")

SECRETS = st.secrets if hasattr(st, "secrets") else {}
DEFAULT_URI = SECRETS.get("NEO4J_URI", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
DEFAULT_USER = SECRETS.get("NEO4J_USER", os.getenv("NEO4J_USER", "neo4j"))
DEFAULT_PASS = SECRETS.get("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "palatiou"))
DEFAULT_ALLOW_WRITE = str(SECRETS.get("ALLOW_WRITE", os.getenv("ALLOW_WRITE", "false"))).lower() in ("1","true","yes")

WRITE_BLOCK_RE = re.compile(
    r"(?i)\b(create|merge|delete|detach\s+delete|set\s+|remove\s+|"
    r"load\s+csv|apoc\.(periodic|refactor|trigger|schema)|call\s+dbms)\b"
)
def is_write_query(cypher: str) -> bool:
    return WRITE_BLOCK_RE.search(cypher or "") is not None

# ----------------------- Neo4j Client -----------------------
@st.cache_resource(show_spinner=False)
def get_driver(uri: str, user: str, password: str) -> Driver:
    return GraphDatabase.driver(uri, auth=basic_auth(user, password))

def run_query(driver: Driver, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if not cypher or not cypher.strip():
        raise ValueError("Empty Cypher")
    with driver.session() as session:
        result = session.run(cypher, **(params or {}))
        return [dict(r) for r in result]

def to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    norm = []
    for r in rows:
        norm.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    return pd.DataFrame(norm)

# ----------------------- Sidebar: Connection -----------------------
st.sidebar.header("⚙️ Connection")
uri = st.sidebar.text_input("Neo4j URI", value=DEFAULT_URI)
user = st.sidebar.text_input("User", value=DEFAULT_USER)
password = st.sidebar.text_input("Password", value=DEFAULT_PASS, type="password")

st.sidebar.divider()
allow_write = st.sidebar.toggle("Allow write queries (dangerous)", value=DEFAULT_ALLOW_WRITE)
show_raw = st.sidebar.toggle("Show raw JSON under tables", value=False)

# ----------------------- Sidebar: Catalog from JSON -----------------------
st.sidebar.divider()
st.sidebar.header("📦 Graph JSON Catalog")
default_data_dir = os.getenv("GRAPH_JSON_DIR", os.getcwd())
data_dir = st.sidebar.text_input("Data folder (must contain nodes.ndjson.gz)", value=default_data_dir)
reload_catalog = st.sidebar.button("Reload catalog")

@st.cache_resource(show_spinner=False)
def load_catalog_cached(path: str) -> GraphCatalog:
    cat = GraphCatalog(path)
    cat.load()
    return cat

catalog: Optional[GraphCatalog] = None
try:
    if reload_catalog:
        load_catalog_cached.clear()
    catalog = load_catalog_cached(data_dir)
    found_nodes = os.path.exists(os.path.join(data_dir, "nodes.ndjson.gz"))
    badge = "✅ Found" if found_nodes else "❌ Missing"
    st.sidebar.success(f"Catalog {badge} nodes.ndjson.gz — joints: {len(catalog.joints)}")
    st.sidebar.caption(f"Subjects — ASD: {catalog.subject_counts['ASD']} | TD: {catalog.subject_counts['TD']}")
except Exception as e:
    st.sidebar.error("Catalog failed to load")
    st.sidebar.caption(str(e))

# ----------------------- Connect to DB -----------------------
st.sidebar.divider()
driver: Optional[Driver] = None
try:
    driver = get_driver(uri, user, password)
    ok = run_query(driver, "RETURN 1 AS ok")
    if ok and ok[0].get("ok") == 1:
        st.sidebar.success("Neo4j connected ✅")
    else:
        st.sidebar.error("Neo4j not reachable ❌")
except Exception as e:
    st.sidebar.error("Neo4j not reachable ❌")
    st.sidebar.caption(str(e))

# ----------------------- NL Engine -----------------------
@st.cache_resource(show_spinner=False)
def get_engine(path: str) -> NL2Cypher:
    return NL2Cypher(data_dir=path)

engine = get_engine(data_dir)

# ----------------------- Main UI -----------------------
tabs = st.tabs(["💬 Ask (NL → Cypher)", "🧪 Query (Cypher)", "📊 Reports", "ℹ️ Help"])

# Tab 1 — Ask
with tabs[0]:
    st.subheader("💬 Natural Language → Cypher")
    q = st.text_area("Question (EN/GR)", height=120, placeholder="e.g., 'mean right hip angle in ASD', 'compare ASD vs TD for knee', 'completeness'")
    colA, colB = st.columns([1,1])
    with colA:
        gen = st.button("🧠 Generate Cypher", type="primary")
    with colB:
        run = st.button("▶️ Generate & Execute", type="secondary")

    if gen or run:
        try:
            cypher = engine.to_cypher(q)
            st.code(cypher, language="cypher")
            if run:
                if not driver:
                    st.error("Not connected to Neo4j.")
                elif not allow_write and is_write_query(cypher):
                    st.error("Blocked write query (read-only mode).")
                else:
                    rows = run_query(driver, cypher)
                    df = to_df(rows)
                    st.success(f"Returned {len(df)} rows.")
                    st.dataframe(df, width='stretch')
                    if show_raw: st.write(rows)
        except Exception as e:
            st.exception(e)

# Tab 2 — Raw query
with tabs[1]:
    st.subheader("🧪 Run raw Cypher")
    default_cy = "MATCH (t:Trial) RETURN count(t) AS trials;"
    cy = st.text_area("Cypher", value=default_cy, height=150)
    c1, c2 = st.columns([1,1])
    if c1.button("Run", type="primary"):
        try:
            if not driver:
                st.error("Not connected to Neo4j.")
            elif not allow_write and is_write_query(cy):
                st.error("Blocked write query (read-only mode).")
            else:
                rows = run_query(driver, cy)
                df = to_df(rows)
                st.success(f"Returned {len(df)} rows.")
                st.dataframe(df, width='stretch')
                if show_raw: st.write(rows)
        except Exception as e:
            st.exception(e)
    if c2.button("Clear"):
        st.rerun()

# Tab 3 — Reports
with tabs[2]:
    st.subheader("📊 Quick Reports")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Trials completeness**")
        cy = """
MATCH (t:Trial)
OPTIONAL MATCH (t)-[:HAS_FILE]->(f:File)
WITH t, collect(DISTINCT f.kind) AS kinds
OPTIONAL MATCH (t)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(feat:Feature)
WITH t, kinds, count(DISTINCT fv) AS nvals, count(DISTINCT feat) AS nfeats
RETURN count(*) AS trials,
       sum(CASE WHEN size(kinds)=4 THEN 1 ELSE 0 END) AS trials_all_files,
       sum(CASE WHEN nvals=463 AND nfeats=463 THEN 1 ELSE 0 END) AS trials_complete;
""".strip()
        st.code(cy, language="cypher")
        if st.button("Run completeness"):
            try:
                rows = run_query(driver, cy) if driver else []
                df = to_df(rows)
                st.dataframe(df, width='stretch')
                if show_raw: st.write(rows)
            except Exception as e:
                st.exception(e)

    with c2:
        st.markdown("**Top correlated features**")
        limit = int(st.number_input("Limit", min_value=1, max_value=200, value=20, step=1))
        cy = f"""
MATCH (a:Feature)-[r:CORRELATED_WITH]->(b:Feature)
RETURN a.code AS A, b.code AS B, r.r AS r, r.n AS n
ORDER BY abs(r) DESC, n DESC
LIMIT {limit};
""".strip()
        st.code(cy, language="cypher")
        if st.button("Run correlations"):
            try:
                rows = run_query(driver, cy) if driver else []
                df = to_df(rows)
                st.dataframe(df, width='stretch')
                if show_raw: st.write(rows)
            except Exception as e:
                st.exception(e)

# Tab 4 — Help
with tabs[3]:
    st.subheader("ℹ️ How to use")
    st.markdown(
        "- **Connect** to Neo4j from the sidebar (URI, user, password). A green badge confirms connection.\n"
        "- **Catalog**: set folder containing `nodes.ndjson.gz` and click **Reload catalog**. "
        "NL→Cypher uses this catalog (joint_guess/sides/stats and stem fallback) so queries match your actual graph.\n"
        "- **Ask (NL → Cypher):** Type a question and press *Generate* or *Generate & Execute*. "
        "The app produces Cypher aligned with your schema using `joint_guess` OR `code` stem + side suffix.\n"
        "- **Query (Cypher):** Run read-only queries. Write operations (CREATE/MERGE/DELETE/SET/LOAD CSV…) are blocked unless you enable **Allow write**.\n"
        "- **Reports:** One-click completeness & correlations summaries.\n"
        "- **Remote use:** If running on a remote server, use SSH port-forwarding and open http://localhost:8504 on your laptop."
    )