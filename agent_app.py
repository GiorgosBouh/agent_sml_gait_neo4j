#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import math
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase, exceptions as neo4j_ex

from nl2cypher import NL2Cypher, load_json

# ---------- UI config ----------
st.set_page_config(page_title="NeuroGait NL↔Cypher Agent", page_icon="🧠", layout="wide")
st.title("🧠 NeuroGait Agent — NL ↔ Cypher (ASD/TD)")

# ---------- Sidebar ----------
with st.sidebar:
    # προτίμηση από secrets, αλλιώς env, αλλιώς defaults
    def sget(k, default):
        return st.secrets.get(k, os.getenv(k, default))

    uri = st.text_input("URI", sget("NEO4J_URI", "bolt://localhost:7687"))
    user = st.text_input("User", sget("NEO4J_USER", "neo4j"))
    password = st.text_input("Password", type="password", value=sget("NEO4J_PASSWORD", "palatiou"))

    st.divider()
    st.header("Model")
    st.caption("Χρησιμοποιείται δωρεάν small model (HuggingFace FLAN-T5-Small ή αντίστοιχο).")
    show_cypher = st.checkbox("Προβολή Cypher", value=True)
    enable_ml = st.checkbox("Ενεργοποίηση ML panel (XGBoost/Linear)", value=True)

# ---------- Helpers ----------
MAX_QUESTION_CHARS = 500  # ασφαλές truncation για SLM

def sanitize_question(q: str) -> str:
    if not q:
        return ""
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    if len(q) > MAX_QUESTION_CHARS:
        q = q[:MAX_QUESTION_CHARS]
    return q

# ---------- Load resources ----------
@st.cache_resource
def get_generator():
    # NL2Cypher εσωτερικά κάνει HF pipeline· εδώ δεν αλλάζουμε το lib, απλά το “προστατεύουμε” απ' τα inputs
    return NL2Cypher()

@st.cache_resource
def get_synonyms():
    return load_json("synonyms.json")

@st.cache_resource
def get_fewshots():
    return load_json("fewshots.json")

generator = get_generator()
synonyms = get_synonyms()
fewshots = get_fewshots()

# ---------- Neo4j client ----------
class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as sess:
            res = sess.run(cypher, params or {})
            return [dict(r) for r in res.data()]

    def ping(self) -> bool:
        try:
            self.run("RETURN 1 AS ok")
            return True
        except Exception:
            return False

# ---------- Session state ----------
if "db" not in st.session_state:
    st.session_state["db"] = None
if "last_cypher" not in st.session_state:
    st.session_state["last_cypher"] = ""
if "last_rows" not in st.session_state:
    st.session_state["last_rows"] = []

# ---------- Connect ----------
col_conn, col_ping = st.columns([1,1])
with col_conn:
    if st.button("🔌 Connect"):
        try:
            db = Neo4jClient(uri, user, password)
            if not db.ping():
                st.error("Απέτυχε το health check (RETURN 1). Έλεγξε URI/credentials/DB.")
            else:
                st.session_state["db"] = db
                st.success("Connected to Neo4j ✔")
        except neo4j_ex.ServiceUnavailable as e:
            st.error(f"ServiceUnavailable: {e}")
        except neo4j_ex.AuthError as e:
            st.error(f"AuthError: {e}")
        except Exception as e:
            st.error(f"Connect error: {e}")

with col_ping:
    if st.session_state["db"] and st.button("🩺 Health check"):
        ok = st.session_state["db"].ping()
        st.success("DB OK") if ok else st.error("DB not reachable")

db: Optional[Neo4jClient] = st.session_state.get("db")

# ---------- NL → Cypher → Exec ----------
st.subheader("❓ Ρώτησε σε φυσική γλώσσα")
q = st.text_area(
    "Ερώτηση (GR/EN):",
    height=100,
    placeholder="π.χ. 'πόση είναι η μέση γωνία του γόνατος (δεξί) σε ASD;' ή 'ASD vs TD knee→ankle coupling'"
)

def rule_based_fallback(user_q: str) -> str:
    """Απλό fallback όταν το SLM αποτυγχάνει: αναγνώριση knee/ankle/hip + ASD/TD + L/R."""
    uq = user_q.lower()
    cond = "ASD" if "asd" in uq else ("TD" if "td" in uq or "typical" in uq else "ASD")
    side = "R" if "right" in uq or "δεξ" in uq else ("L" if "left" in uq or "αρ" in uq else "R")
    # knee mean
    knee_code = "HIANR" if side == "R" else "HIANL"
    # ankle mean
    ankle_code = "KNFOR" if side == "R" else "KNFOL"

    if "coupling" in uq or "συσχ" in uq or "αύξηση" in uq:
        # per-subject OLS coupling knee→ankle
        return f"""
CALL () {{
  WITH *
  UNWIND [ ['L','HIANL','KNFOL'], ['R','HIANR','KNFOR'] ] AS cfg
  WITH cfg[0] AS side, cfg[1] AS knee_code, cfg[2] AS ankle_code
  MATCH (p:Subject)-[:HAS_CONDITION]->(:Condition {{name:'{cond}'}})
  MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
  MATCH (s)-[:HAS_VALUE]->(kfv:FeatureValue)-[:OF_FEATURE]->(kf:Feature)
  WHERE kf.stat='mean' AND kf.code =~ ('(?i).*' + knee_code + '\\s*$')
  MATCH (s)-[:HAS_VALUE]->(afv:FeatureValue)-[:OF_FEATURE]->(af:Feature)
  WHERE af.stat='mean' AND af.code =~ ('(?i).*' + ankle_code + '\\s*$')
  WITH side, p.pid AS pid, avg(kfv.value) AS knee_subj_mean, avg(afv.value) AS ankle_subj_mean
  WITH side,
       collect(knee_subj_mean) AS X, collect(ankle_subj_mean) AS Y,
       size(collect(knee_subj_mean)) AS n,
       reduce(a=0.0, v IN collect(knee_subj_mean) | a+v) AS sx,
       reduce(a=0.0, v IN collect(ankle_subj_mean) | a+v) AS sy,
       reduce(a=0.0, i IN range(0,n-1) | a + X[i]*Y[i]) AS sxy,
       reduce(a=0.0, i IN range(0,n-1) | a + X[i]*X[i]) AS sxx,
       reduce(a=0.0, i IN range(0,n-1) | a + Y[i]*Y[i]) AS syy
  WITH side, n, sx, sy, sxy, sxx, syy,
       (n*sxx - sx*sx) AS denom,
       (n*sxy - sx*sy) AS num,
       (n*syy - sy*sy) AS Syy
  RETURN '{cond}' AS condition, side, n,
         round(CASE WHEN denom<>0 THEN num*1.0/denom ELSE null END,4) AS beta,
         round(CASE WHEN n>0 THEN (sy - (CASE WHEN denom<>0 THEN num*1.0/denom ELSE 0 END)*sx)*1.0/n ELSE null END,4) AS alpha,
         round(CASE WHEN denom>0 AND Syy>0 THEN (1.0*num*num)/(denom*Syy) ELSE null END,4) AS R2,
         round(CASE WHEN denom<>0 THEN (num*1.0/denom)*5.0 ELSE null END,4) AS delta_for_plus5
}}
RETURN condition, side, n, beta, alpha, R2, delta_for_plus5
ORDER BY side;
        """.strip()

    # αλλιώς απλό “mean by joint”
    target_code = knee_code if ("knee" in uq or "γόνα" in uq) else ankle_code
    joint = "Knee" if target_code.startswith("HIAN") else "Ankle"
    return f"""
MATCH (p:Subject)-[:HAS_CONDITION]->(:Condition {{name:'{cond}'}})
MATCH (p)-[:HAS_SAMPLE]->(:Sample)-[:HAS_VALUE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
WHERE f.stat='mean' AND f.code =~ '(?i).*{target_code}\\s*$'
RETURN '{joint}' AS joint, '{side}' AS side, '{cond}' AS condition,
       round(avg(fv.value),2) AS mean_deg, count(*) AS n;
    """.strip()

def generate_cypher(user_q: str) -> str:
    """Προσπάθησε με SLM· αν αποτύχει ή επιστρέψει άδειο, κάνε fallback."""
    cleaned = sanitize_question(user_q)
    if not cleaned:
        return ""
    try:
        cy = generator(cleaned, synonyms, fewshots)  # η NL2Cypher σου
        if not isinstance(cy, str) or not cy.strip():
            raise ValueError("Empty Cypher from model")
        return cy
    except Exception as e:
        st.warning(f"Model fallback (rule-based): {e}")
        return rule_based_fallback(cleaned)

# --- UI controls for NL2Cypher/Exec ---
colA, colB = st.columns([1,1])
with colA:
    if st.button("🧠 Generate Cypher"):
        cy = generate_cypher(q)
        st.session_state["last_cypher"] = cy

with colB:
    exec_disabled = st.session_state["db"] is None
    if st.button("▶️ Execute", disabled=exec_disabled):
        cy = st.session_state.get("last_cypher", "")
        if not cy.strip():
            st.warning("Δεν υπάρχει Cypher για εκτέλεση.")
        else:
            try:
                rows = st.session_state["db"].run(cy)
                st.session_state["last_rows"] = rows
                st.success(f"OK — {len(rows)} γραμμές.")
            except neo4j_ex.Neo4jError as e:
                st.error(f"Neo4j error: {e}")
            except Exception as e:
                st.error(f"Exec error: {e}")

# Show cypher
cy = st.session_state.get("last_cypher", "")
if show_cypher and cy:
    st.code(cy, language="cypher")

# Show results
rows = st.session_state.get("last_rows", [])
if rows:
    st.write("**Αποτελέσματα**")
    st.dataframe(pd.DataFrame(rows))

    cols = set(rows[0].keys())
    if {"n", "beta"}.issubset(cols):
        st.info(
            "How to read: condition = group (ASD/TD); side = limb (L/R); n = participants (~50/group); "
            "beta = ankle change (deg) per 1 deg knee; alpha = baseline ankle when knee=0; "
            "R² = coupling strength (0–1); delta_for_plus5 = expected ankle change (deg) for +5 deg knee."
        )

st.divider()
st.subheader("⚙️ Quick actions")

qa1, qa2, qa3 = st.columns(3)
with qa1:
    disabled = st.session_state["db"] is None
    if st.button("📊 Knee→Ankle (per-subject OLS, ASD/TD, L/R)", disabled=disabled):
        # Χρησιμοποιώ το fewshot (ενημέρωσε το template σου σε CALL () { WITH * ... } αν θες να φύγει το deprecation warning)
        fs = [fs for fs in fewshots if "coupling per subject" in fs.get("q","").lower()]
        if fs:
            st.session_state["last_cypher"] = fs[0]["cypher"]
            st.session_state["last_rows"] = st.session_state["db"].run(fs[0]["cypher"])

with qa2:
    if st.button("🧩 Clinician explanation row"):
        st.info(
            "How to read: condition = group (ASD/TD); side = limb (L/R); n = participants (~50/group); "
            "beta = ankle change (deg) per 1 deg knee; alpha = baseline ankle when knee=0; "
            "R² = coupling strength (0–1); delta_for_plus5 = expected ankle change (deg) for +5 deg knee."
        )

with qa3:
    if enable_ml and st.button("🤖 ML demo (XGB/Linear) ASD Right: predict Ankle from Knee/Hip/StaT/Velocity", disabled=disabled):
        df = pd.DataFrame(st.session_state["db"].run("""
MATCH (p:Subject)-[:HAS_CONDITION]->(:Condition {name:'ASD'})
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
OPTIONAL MATCH (s)-[:HAS_VALUE]->(k:FeatureValue)-[:OF_FEATURE]->(fk:Feature)
WHERE fk.stat='mean' AND fk.code =~ '(?i).*HIANR\\s*$'
OPTIONAL MATCH (s)-[:HAS_VALUE]->(a:FeatureValue)-[:OF_FEATURE]->(fa:Feature)
WHERE fa.stat='mean' AND fa.code =~ '(?i).*KNFOR\\s*$'
OPTIONAL MATCH (s)-[:HAS_VALUE]->(h:FeatureValue)-[:OF_FEATURE]->(fh:Feature)
WHERE fh.stat='mean' AND fh.code =~ '(?i).*SPKNR\\s*$'
OPTIONAL MATCH (s)-[:HAS_VALUE]->(st:FeatureValue)-[:OF_FEATURE]->(:Feature {code:'StaT'})
OPTIONAL MATCH (s)-[:HAS_VALUE]->(v:FeatureValue)-[:OF_FEATURE]->(:Feature {code:'Velocity'})
RETURN coalesce(k.value, null) AS knee,
       coalesce(h.value, null) AS hip,
       coalesce(st.value, null) AS stat_ms,
       coalesce(v.value, null) AS vel,
       coalesce(a.value, null) AS ankle
"""))
        df = df.dropna()
        if df.empty:
            st.warning("No data for ML.")
        else:
            X = df[["knee","hip","stat_ms","vel"]].values
            y = df["ankle"].values
            try:
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=300, max_depth=3, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9,
                    objective="reg:squarederror", n_jobs=0, random_state=42
                )
                model.fit(X, y)
                r2 = float(model.score(X, y))
                st.success(f"XGBoost R²={r2:.4f} (in-sample). Feature importances:")
                st.json(dict(zip(["knee","hip","StaT","Velocity"], model.feature_importances_.round(4))))
            except Exception:
                from sklearn.linear_model import LinearRegression
                m = LinearRegression().fit(X, y)
                r2 = float(m.score(X, y))
                st.success(f"Linear R²={r2:.4f} (in-sample).")