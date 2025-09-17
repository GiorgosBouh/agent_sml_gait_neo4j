#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit UI for ASD Gait — Neo4j Explorer (NL ↔ Cypher)
Schema (current):
  (:Subject {pid,sid})-[:HAS_TRIAL]->(:Trial {uid})
  (:Trial)-[:HAS_FILE]->(:File {uid,kind})
  (:Trial)-[:HAS_FEATURE]->(:FeatureValue {value})-[:OF_FEATURE]->(:Feature {code,stat})

Run:
  streamlit run agent_app.py --server.port 8504 --server.address 0.0.0.0
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
from neo4j import GraphDatabase, basic_auth, exceptions as neo4j_ex, Driver

# Optional: import the rule-based NL→Cypher engine you created earlier.
# If it's missing, we keep a local lightweight generator below.
try:
    from nl2cypher import NL2Cypher  # type: ignore
except Exception:
    NL2Cypher = None  # fallback to local rules

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

st.sidebar.divider()
driver: Optional[Driver] = None
driver_err = None
try:
    driver = get_driver(uri, user, password)
    ok = run_query(driver, "RETURN 1 AS ok")
    if ok and ok[0].get("ok") == 1:
        st.sidebar.success("Connected to Neo4j ✅", icon="🟢")
    else:
        st.sidebar.error("Neo4j not reachable ❌", icon="🔴")
except Exception as e:
    driver_err = str(e)
    st.sidebar.error("Neo4j not reachable ❌", icon="🔴")
    st.sidebar.caption(driver_err or "")

# ----------------------- NL helpers (local) -----------------------
JOINT_STEMS: Dict[str, Tuple[str,str]] = {
    # keyword → (feature code stem, nice name)
    "knee": ("HIAN", "Knee"),
    "γόνα": ("HIAN", "Knee"),
    "gonato": ("HIAN", "Knee"),
    "ankle": ("KNFO", "Ankle"),
    "ποδοκν": ("KNFO", "Ankle"),
    "hip": ("SPKN", "Hip"),
    "ισχ": ("SPKN", "Hip"),
    "trunk": ("THHTI","TrunkTilt"),
    "κορμ": ("THHTI","TrunkTilt"),
    "spine": ("SPINE","Spine"),
    "pelvis": ("SPEL","Pelvis"),
}

SPATIOTEMPORAL: Dict[str, Tuple[str,str]] = {
    # label → (Feature.code, unit)
    "velocity": ("Velocity", "m/s"),
    "stance": ("StaT", "ms"),
    "swing": ("SwiT", "ms"),
    "gait cycle": ("GaCT", "ms"),
    "stride length": ("StrLe", "m"),
    "step length": ("MaxStLe", "m"),
    "step width": ("MaxStWi", "m"),
}

def detect_condition(q: str) -> str:
    uq = q.lower()
    if re.search(r"\basd\b.*(vs|and|&|,)\s*\btd\b", uq) or re.search(r"\btd\b.*(vs|and|&|,)\s*\basd\b", uq):
        return "BOTH"
    if "asd" in uq or "αυτισ" in uq:
        return "ASD"
    if re.search(r"\btd\b", uq) or "τυπικ" in uq or "control" in uq:
        return "TD"
    return "BOTH"

def detect_side(q: str) -> str:
    uq = q.lower()
    if re.search(r"\b(left|αριστερ(?:ά|η)|αρ\.)\b", uq): return "L"
    if re.search(r"\b(right|δεξι(?:ά|ή)|δε\.)\b", uq): return "R"
    return "BOTH"

def detect_joint(q: str) -> Optional[Tuple[str,str]]:
    uq = q.lower()
    for key, (stem, name) in JOINT_STEMS.items():
        if key in uq:
            return stem, name
    return None

def spatiotemporal_key(q: str) -> Optional[str]:
    uq = q.lower()
    for label in SPATIOTEMPORAL.keys():
        if label in uq or label.replace(" ", "") in uq:
            return label
    return None

def feature_regex_from_text(q: str) -> Optional[str]:
    m = re.search(r"(features?|codes?|χαρακτηριστικ\w*|κωδικ\w*)\s*(?:like|όπως|=|regex)\s*([A-Za-z0-9\-\_\.\\\*]+)", q, flags=re.IGNORECASE)
    if m:
        return m.group(2).replace("*", ".*")
    return None

def asks_coupling(q: str) -> Optional[Tuple[str,str]]:
    uq = q.lower()
    if "coupling" in uq or "συσχέτι" in uq or "correlat" in uq or "regress" in uq:
        # best-effort: find two joints
        keys = [k for k in JOINT_STEMS.keys() if k in uq]
        if len(keys) >= 2:
            a = JOINT_STEMS[keys[0]][0]
            b = JOINT_STEMS[keys[1]][0]
            return a, b
        # default Knee→Ankle
        return "HIAN", "KNFO"
    return None

# ----------------------- Cypher templates (new schema) -----------------------
def cy_unwind_sides(side: str) -> str:
    return "WITH CASE WHEN '{s}'='BOTH' THEN ['L','R'] ELSE ['{s}'] END AS sides\nUNWIND sides AS side".format(s=side)

def cy_mean_per_subject(cond: str, side: str, joint_stem: str, stat: str="mean") -> str:
    if side == "BOTH":
        side_filter = f"(f.code =~ '(?i).*{joint_stem}L\\s*$' OR f.code =~ '(?i).*{joint_stem}R\\s*$')"
        side_expr   = "CASE WHEN f.code =~ '(?i).*L\\s*$' THEN 'L' ELSE 'R' END"
    else:
        side_filter = f"f.code =~ '(?i).*{joint_stem}{side}\\s*$'"
        side_expr   = f"'{side}'"
    cond_case = "CASE WHEN s.pid STARTS WITH 'ASD:' THEN 'ASD' WHEN s.pid STARTS WITH 'TD:' THEN 'TD' ELSE 'UNK' END"
    cond_filter = "" if cond == "BOTH" else f"WHERE s.pid STARTS WITH '{cond}:'"
    return f"""
MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
{cond_filter}
WHERE f.stat='{stat}' AND {side_filter}
WITH {cond_case} AS condition, {side_expr} AS side, s.pid AS pid, avg(fv.value) AS subj_mean
RETURN condition, side, round(avg(subj_mean),2) AS mean_value, count(*) AS n
ORDER BY condition, side;
""".strip()

def cy_spatiotemporal_mean(cond: str, feature_code: str) -> str:
    cond_case = "CASE WHEN s.pid STARTS WITH 'ASD:' THEN 'ASD' WHEN s.pid STARTS WITH 'TD:' THEN 'TD' ELSE 'UNK' END"
    cond_filter = "" if cond == "BOTH" else f"WHERE s.pid STARTS WITH '{cond}:'"
    return f"""
MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature {{code:'{feature_code}'}})
{cond_filter}
WITH {cond_case} AS condition, s.pid AS pid, avg(fv.value) AS subj_mean
RETURN condition, round(avg(subj_mean),3) AS mean_value, count(*) AS n
ORDER BY condition;
""".strip()

def cy_compare_groups(code_pattern: str, stat: str="mean") -> str:
    return f"""
UNWIND ['ASD','TD'] AS grp
MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
WHERE s.pid STARTS WITH grp + ':' AND f.stat='{stat}' AND f.code =~ '(?i).*{code_pattern}\\s*$'
WITH grp AS condition, s.pid AS pid, avg(fv.value) AS subj_mean
RETURN condition, round(avg(subj_mean),3) AS mean_value, count(*) AS n
ORDER BY condition;
""".strip()

def cy_coupling_ols(cond: str, side: str, a_stem: str, b_stem: str) -> str:
    cond_case = "CASE WHEN s.pid STARTS WITH 'ASD:' THEN 'ASD' WHEN s.pid STARTS WITH 'TD:' THEN 'TD' ELSE 'UNK' END"
    cond_filter = "" if cond == "BOTH" else f"WHERE s.pid STARTS WITH '{cond}:'"
    return f"""
{cy_unwind_sides(side)}
WITH side, '{a_stem}' AS a_stem, '{b_stem}' AS b_stem
MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(av:FeatureValue)-[:OF_FEATURE]->(af:Feature)
{cond_filter}
WHERE af.stat='mean' AND af.code =~ ('(?i).*' + a_stem + side + '\\s*$')
MATCH (s)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(bv:FeatureValue)-[:OF_FEATURE]->(bf:Feature)
WHERE bf.stat='mean' AND bf.code =~ ('(?i).*' + b_stem + side + '\\s*$')
WITH {cond_case} AS condition, side, s.pid AS pid, avg(av.value) AS A, avg(bv.value) AS B
WITH condition, side, collect({{a:A, b:B}}) AS pairs
UNWIND pairs AS p
WITH condition, side,
     count(*) AS n,
     sum(p.a) AS sa, sum(p.b) AS sb,
     sum(p.a*p.b) AS sab,
     sum(p.a*p.a) AS saa, sum(p.b*p.b) AS sbb
WITH condition, side, n, sa, sb, sab, saa, sbb,
     (n*saa - sa*sa) AS denom,
     (n*sab - sa*sb) AS num,
     (n*sbb - sb*sb) AS Syy
RETURN condition, side, n,
       round(CASE WHEN denom<>0 THEN num/denom ELSE null END,4) AS beta,
       round(CASE WHEN denom>0 AND Syy>0 THEN (num*num)/(denom*Syy) ELSE null END,4) AS R2
ORDER BY condition, side;
""".strip()

def cy_list_features(regex_like: str) -> str:
    return f"""
MATCH (f:Feature)
WHERE f.code =~ '(?i).*{regex_like}.*'
RETURN f.code AS code, f.stat AS stat
ORDER BY code
LIMIT 200;
""".strip()

def cy_count_subjects() -> str:
    return "MATCH (s:Subject) RETURN sum(CASE WHEN s.pid STARTS WITH 'ASD:' THEN 1 ELSE 0 END) AS asd_cases, sum(CASE WHEN s.pid STARTS WITH 'TD:' THEN 1 ELSE 0 END) AS td_cases;"

# ----------------------- NL→Cypher generator (local fallback) -----------------------
def generate_cypher_local(q: str) -> Tuple[str,str]:
    uq = (q or "").strip()
    if not uq:
        return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;", "noop"

    # counts
    if re.search(r"\b(count|how many|πόσα)\b.*\b(asd|td|subjects|participants|cases)\b", uq, flags=re.IGNORECASE):
        return cy_count_subjects(), "count"

    # feature listing
    rgx = feature_regex_from_text(uq)
    if rgx:
        return cy_list_features(rgx), "features"

    # spatiotemporal
    sp = spatiotemporal_key(uq)
    if sp:
        cond = detect_condition(uq)
        code, _unit = SPATIOTEMPORAL[sp]
        return cy_spatiotemporal_mean(cond, code), "spatio"

    # coupling / regression
    cp = asks_coupling(uq)
    if cp:
        cond = detect_condition(uq)
        side = detect_side(uq)
        a_stem, b_stem = cp
        return cy_coupling_ols(cond, side, a_stem, b_stem), "coupling"

    # compare ASD vs TD
    if re.search(r"(compare|vs|diff|σύγκρι|διαφορ)", uq, flags=re.IGNORECASE):
        side = detect_side(uq)
        js = detect_joint(uq) or ("HIAN","Knee")
        code_pat = f"{js[0]}{'' if side=='BOTH' else side}"
        return cy_compare_groups(code_pat), "compare"

    # mean of a feature by side/cond
    if re.search(r"(mean|avg|average|μ\.?ο\.?|μέση)", uq, flags=re.IGNORECASE) or detect_joint(uq):
        cond = detect_condition(uq)
        side = detect_side(uq)
        js = detect_joint(uq) or ("HIAN","Knee")
        return cy_mean_per_subject(cond, side, js[0], "mean"), "mean"

    # fallback
    return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;", "noop"

# ----------------------- Main UI -----------------------
tabs = st.tabs(["💬 Ask (NL → Cypher)", "🧪 Query (Cypher)", "📊 Reports", "ℹ️ Help"])

# Tab 1 — Ask
with tabs[0]:
    st.subheader("💬 Natural Language → Cypher")
    q = st.text_area("Question (EN/GR)", height=120, placeholder="e.g., 'mean knee right in ASD', 'compare ASD vs TD for knee', 'list features like HIAN'")
    colA, colB = st.columns([1,1])
    with colA:
        gen = st.button("🧠 Generate Cypher", type="primary")
    with colB:
        run = st.button("▶️ Generate & Execute", type="secondary")

    if gen or run:
        try:
            if NL2Cypher is not None:
                engine = NL2Cypher()
                cypher = engine.to_cypher(q)
            else:
                cypher, _ = generate_cypher_local(q)

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
                    st.dataframe(df, use_container_width=True)
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
                st.dataframe(df, use_container_width=True)
                if show_raw: st.write(rows)
        except Exception as e:
            st.exception(e)
    if c2.button("Clear"):
        st.experimental_rerun()

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
                st.dataframe(df, use_container_width=True)
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
                st.dataframe(df, use_container_width=True)
                if show_raw: st.write(rows)
            except Exception as e:
                st.exception(e)

# Tab 4 — Help
with tabs[3]:
    st.subheader("ℹ️ How to use")
    st.markdown("""
- **Connect** to Neo4j from the sidebar (URI, user, password). You should see a green “Connected”.
- **Ask (NL → Cypher):** Type a question and press *Generate* or *Generate & Execute*.
  The app produces Cypher aligned with your schema: