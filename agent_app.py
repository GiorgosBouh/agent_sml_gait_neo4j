#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import json
import pandas as pd
from typing import Dict, Any, List, Optional
import streamlit as st
from neo4j import GraphDatabase

from nl2cypher import NL2Cypher, load_json

# ---------- UI config ----------
st.set_page_config(page_title="NeuroGait NL↔Cypher Agent", page_icon="🧠", layout="wide")
st.title("🧠 NeuroGait Agent — NL ↔ Cypher (ASD/TD)")

# ---------- Sidebar ----------
with st.sidebar:
    uri = st.text_input("URI", os.getenv("NEO4J_URI","bolt://localhost:7687"))
    user = st.text_input("User", os.getenv("NEO4J_USER","neo4j"))
    password = st.text_input("Password", type="password",
                             value=os.getenv("NEO4J_PASSWORD","palatiou"))
    st.divider()
    st.header("Model")
    st.caption("Χρησιμοποιείται δωρεάν small model (HuggingFace FLAN-T5-Small).")
    show_cypher = st.checkbox("Προβολή Cypher", value=True)
    enable_ml = st.checkbox("Ενεργοποίηση ML panel (XGBoost/Linear)", value=True)

# ---------- Load resources ----------
@st.cache_resource
def get_generator():
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

if st.button("🔌 Connect"):
    st.session_state["db"] = Neo4jClient(uri, user, password)
    st.success("Connected to Neo4j.")

db: Optional[Neo4jClient] = st.session_state.get("db")

# ---------- NL → Cypher → Exec ----------
st.subheader("❓ Ρώτησε σε φυσική γλώσσα")
q = st.text_area("Ερώτηση (GR/EN):", height=100,
                 placeholder="π.χ. 'πόση είναι η μέση γωνία του γόνατος (δεξί) σε ASD;' ή 'ASD vs TD knee→ankle coupling'")

colA, colB = st.columns([1,1])
with colA:
    if st.button("🧠 Generate Cypher"):
        try:
            cy = generator(q, synonyms, fewshots)
        except Exception as e:
            st.error(f"Model error: {e}")
            cy = ""
        st.session_state["last_cypher"] = cy
with colB:
    if st.button("▶️ Execute"):
        cy = st.session_state.get("last_cypher","")
        if not db:
            st.error("Δεν υπάρχει σύνδεση Neo4j.")
        elif not cy.strip():
            st.warning("Δεν υπάρχει Cypher για εκτέλεση.")
        else:
            try:
                rows = db.run(cy)
                st.session_state["last_rows"] = rows
                st.success(f"OK — {len(rows)} γραμμές.")
            except Exception as e:
                st.error(str(e))

# Show cypher
cy = st.session_state.get("last_cypher","")
if show_cypher and cy:
    st.code(cy, language="cypher")

# Show results
rows = st.session_state.get("last_rows", [])
if rows:
    st.write("**Αποτελέσματα**")
    st.dataframe(pd.DataFrame(rows))

    # Λιτό clinical explanation αν ταιριάζουν στήλες
    cols = set(rows[0].keys())
    if {"n","beta"}.issubset(cols):
        st.info("How to read: condition = group (ASD/TD); side = limb (L/R); n = participants (~50/group); "
                "beta = ankle change (deg) per 1 deg knee; alpha = baseline ankle when knee=0; "
                "R2 = coupling strength (0–1); delta_for_plus5 = expected ankle change (deg) for +5 deg knee.")

st.divider()
st.subheader("⚙️ Quick actions")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("📊 Knee→Ankle (per-subject OLS, ASD/TD, L/R)"):
        cy = [fs for fs in fewshots if "coupling per subject" in fs["q"]][0]["cypher"]
        st.session_state["last_cypher"] = cy
        if db:
            st.session_state["last_rows"] = db.run(cy)
with c2:
    if st.button("🧩 Clinician explanation row"):
        st.info("How to read: condition = group (ASD/TD); side = limb (L/R); n = participants (~50/group); "
                "beta = ankle change (deg) per 1 deg knee; alpha = baseline ankle when knee=0; "
                "R2 = coupling strength (0–1); delta_for_plus5 = expected ankle change (deg) for +5 deg knee.")
with c3:
    if enable_ml and st.button("🤖 ML demo (XGB/Linear) ASD Right: predict Ankle from Knee/Hip/StaT/Velocity"):
        if not db:
            st.error("Connect first.")
        else:
            df = pd.DataFrame(db.run(f"""
MATCH (p:Subject)-[:HAS_CONDITION]->(:Condition {{name:'ASD'}})
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
OPTIONAL MATCH (s)-[:HAS_VALUE]->(k:FeatureValue)-[:OF_FEATURE]->(fk:Feature)
WHERE fk.stat='mean' AND fk.code =~ '(?i).*HIANR\\s*$'
OPTIONAL MATCH (s)-[:HAS_VALUE]->(a:FeatureValue)-[:OF_FEATURE]->(fa:Feature)
WHERE fa.stat='mean' AND fa.code =~ '(?i).*KNFOR\\s*$'
OPTIONAL MATCH (s)-[:HAS_VALUE]->(h:FeatureValue)-[:OF_FEATURE]->(fh:Feature)
WHERE fh.stat='mean' AND fh.code =~ '(?i).*SPKNR\\s*$'
OPTIONAL MATCH (s)-[:HAS_VALUE]->(st:FeatureValue)-[:OF_FEATURE]->(:Feature {{code:'StaT'}})
OPTIONAL MATCH (s)-[:HAS_VALUE]->(v:FeatureValue)-[:OF_FEATURE]->(:Feature {{code:'Velocity'}})
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