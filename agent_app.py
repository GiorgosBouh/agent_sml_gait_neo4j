#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase, exceptions as neo4j_ex

from nl2cypher import NL2Cypher, load_json

# ---------- UI config ----------
st.set_page_config(page_title="NeuroGait NL↔Cypher Agent", page_icon="🧠", layout="wide")
st.title("🧠 NeuroGait Agent — NL ↔ Cypher (ASD/TD)")

# ---------- Sidebar ----------
with st.sidebar:
    def sget(k, default):
        return st.secrets.get(k, os.getenv(k, default))

    uri = st.text_input("URI", sget("NEO4J_URI", "bolt://127.0.0.1:7687"))
    user = st.text_input("User", sget("NEO4J_USER", "neo4j"))
    password = st.text_input("Password", type="password", value=sget("NEO4J_PASSWORD", "palatiou"))

    st.divider()
    st.header("Model")
    st.caption("Small free SLM (HF FLAN-T5-Small) — χρησιμοποιείται μόνο αν δεν αναγνωρίσει το intent ο κανόνας.")
    show_cypher = st.checkbox("Προβολή Cypher", value=True)
    enable_ml = st.checkbox("Ενεργοποίηση ML panel (XGBoost/Linear)", value=False)

# ---------- Caches ----------
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

    def ping(self) -> bool:
        try:
            self.run("RETURN 1 AS ok")
            return True
        except Exception:
            return False

# ---------- Session ----------
if "db" not in st.session_state: st.session_state["db"] = None
if "last_cypher" not in st.session_state: st.session_state["last_cypher"] = ""
if "last_rows" not in st.session_state: st.session_state["last_rows"] = []
if "last_tag" not in st.session_state: st.session_state["last_tag"] = ""

# ---------- Connect controls ----------
col_conn, col_ping = st.columns([1,1])
with col_conn:
    if st.button("🔌 Connect"):
        try:
            db = Neo4jClient(uri, user, password)
            if not db.ping():
                st.error("Health check failed (RETURN 1). Check URI/credentials/DB.")
            else:
                st.session_state["db"] = db
                st.success("Connected ✔")
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

# ---------- Helpers: parsing & dictionaries ----------

JOINT_MAP = {
    # joint keyword -> (code stem, nice name)
    # angles
    "knee":   ("HIAN", "Knee"),
    "γονατ":  ("HIAN", "Knee"),
    "ankle":  ("KNFO", "Ankle"),
    "ποδοκν": ("KNFO", "Ankle"),
    "hip":    ("SPKN", "Hip"),
    "ισχυ":   ("SPKN", "Hip"),
    "trunk":  ("THHTI","TrunkTilt"),
    "κορμ":   ("THHTI","TrunkTilt"),
    "spine":  ("THH",  "Spine"),
    "pelvis": ("SPEL", "Pelvis"),
    "ώμος":   ("SHWR", "Shoulder"),
    "shoulder": ("SHWR","Shoulder"),
    "elbow":  ("ELHA", "Elbow"),
    "αγκων":  ("ELHA", "Elbow"),
}

SPATIOTEMPORAL = {
    # plain feature codes in your KG
    "stride length": ("StrLe", "m"),
    "step length":   ("MaxStLe", "m"),
    "step width":    ("MaxStWi", "m"),
    "gait cycle":    ("GaCT", "ms"),
    "stance":        ("StaT", "ms"),
    "swing":         ("SwiT", "ms"),
    "velocity":      ("Velocity", "m/s"),
}

def detect_condition(uq: str) -> str:
    if re.search(r"\b(asd|αυτισ)\b", uq): return "ASD"
    if re.search(r"\b(td|typical|τυπικ)\b", uq): return "TD"
    if re.search(r"\b(asd\s*vs\s*td|td\s*vs\s*asd|compare|diff|διαφορ)\b", uq): return "BOTH"
    return "ASD"  # default

def detect_side(uq: str) -> str:
    if re.search(r"\b(right|δεξ)\b", uq): return "R"
    if re.search(r"\b(left|αρισ|αρ\.)\b", uq): return "L"
    if re.search(r"\b(both|and both|και τα δυο|bilateral)\b", uq): return "BOTH"
    return "R"  # default

def find_joint(uq: str) -> Optional[Tuple[str,str]]:
    for k,(stem,nice) in JOINT_MAP.items():
        if k in uq:
            return stem, nice
    return None

def asks_mean(uq: str) -> bool:
    return any(w in uq for w in ["mean", "average", "μέση", "μεση", "avg"])

def asks_variance(uq: str) -> bool:
    return any(w in uq for w in ["variance", "var", "διασπορ"])

def asks_std(uq: str) -> bool:
    return any(w in uq for w in ["std", "stdev", "τυπικ", "διακ"])

def asks_coupling(uq: str) -> Optional[Tuple[str,str]]:
    """
    Returns pair (source_joint_stem, target_joint_stem) for OLS if found.
    Examples:
      knee->ankle, hip->knee, knee->trunk …
    """
    pairs = [
        (["knee","γονατ"], ["ankle","ποδοκν"]),
        (["hip","ισχυ"], ["knee","γονατ"]),
        (["knee","γονατ"], ["trunk","κορμ"]),
    ]
    for src_list, tgt_list in pairs:
        if any(k in uq for k in src_list) and any(k in uq for k in tgt_list):
            # map to stems
            src = next(JOINT_MAP[k][0] for k in JOINT_MAP if k in uq and k in src_list)
            tgt = next(JOINT_MAP[k][0] for k in JOINT_MAP if k in uq and k in tgt_list)
            return src, tgt
    if any(w in uq for w in ["coupling","συσχ","συσχετ","regression","ols"]):
        # default knee->ankle
        return "HIAN","KNFO"
    return None

def spatiotemporal_key(uq: str) -> Optional[Tuple[str,str]]:
    for k,(code,unit) in SPATIOTEMPORAL.items():
        if any(w in uq for w in [k, k.replace(" ", ""), k.split()[0]]):
            return code, unit
    return None

def feature_regex_from_text(uq: str) -> Optional[str]:
    m = re.search(r"(features?|χαρακτηριστικ|codes?|κωδικ\w+)\s*(like|όπως|regex|=)\s*([A-Za-z0-9\.\-\_\*]+)", uq)
    if m:
        pat = m.group(3)
        # turn glob to regex-ish (very simple)
        pat = pat.replace("*", ".*")
        return pat
    return None

# ---------- Intent → Cypher (Rule-based) ----------

def cy_mean_per_subject(cond: str, side: str, joint_stem: str, joint_name: str, stat="mean") -> str:
    # Build regex suffix based on side
    code = f"{joint_stem}{'R' if side=='R' else 'L'}" if side in ("L","R") else ""
    side_filter = (
        f"f.code =~ '(?i).*{joint_stem}L\\s*$' OR f.code =~ '(?i).*{joint_stem}R\\s*$'"
        if side == "BOTH" else f"f.code =~ '(?i).*{code}\\s*$'"
    )
    cond_filter = "" if cond == "BOTH" else f"WHERE c.name='{cond}'"
    return f"""
MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
{cond_filter}
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
MATCH (s)-[:HAS_VALUE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
WHERE f.stat='{stat}' AND ({side_filter})
WITH c.name AS condition,
     CASE WHEN f.code =~ '(?i).*L\\s*$' THEN 'L' ELSE 'R' END AS side,
     p.pid AS pid, avg(fv.value) AS subj_mean
RETURN condition, side, round(avg(subj_mean),2) AS mean_deg, count(*) AS n
ORDER BY condition, side
""".strip()

def cy_spatiotemporal_mean(cond: str, code: str) -> str:
    cond_filter = "" if cond == "BOTH" else f"WHERE c.name='{cond}'"
    return f"""
MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
{cond_filter}
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
MATCH (s)-[:HAS_VALUE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature {{code:'{code}'}})
WITH c.name AS condition, p.pid AS pid, avg(fv.value) AS subj_mean
RETURN condition, round(avg(subj_mean),3) AS mean_value, count(*) AS n
ORDER BY condition
""".strip()

def cy_coupling_ols(cond: str, side: str, src_stem: str, tgt_stem: str) -> str:
    def pattern(stem,side):
        if side == "BOTH":
            return f"f.code =~ '(?i).*{stem}L\\s*$' OR f.code =~ '(?i).*{stem}R\\s*$'"
        return f"f.code =~ '(?i).*{stem}{side}\\s*$'"

    cond_filter = "" if cond == "BOTH" else f"WHERE c.name='{cond}'"
    return f"""
UNWIND [{ "'L','R'" if side=='BOTH' else f"'{side}'" }] AS side
MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
{cond_filter}
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)

// source X
MATCH (s)-[:HAS_VALUE]->(xv:FeatureValue)-[:OF_FEATURE]->(xf:Feature)
WHERE xf.stat='mean' AND ({pattern(src_stem, 'L' if side=='BOTH' else side)})

// target Y
MATCH (s)-[:HAS_VALUE]->(yv:FeatureValue)-[:OF_FEATURE]->(yf:Feature)
WHERE yf.stat='mean' AND ({pattern(tgt_stem,'L' if side=='BOTH' else side)})

WITH c.name AS condition, side, p.pid AS pid,
     avg(xv.value) AS X, avg(yv.value) AS Y
WITH condition, side, collect(X) AS Xs, collect(Y) AS Ys
WITH condition, side, Xs, Ys, size(Xs) AS n,
     reduce(a=0.0, v IN Xs | a+v) AS sx,
     reduce(a=0.0, v IN Ys | a+v) AS sy,
     reduce(a=0.0, i IN range(0,n-1) | a + Xs[i]*Ys[i]) AS sxy,
     reduce(a=0.0, i IN range(0,n-1) | a + Xs[i]*Xs[i]) AS sxx,
     reduce(a=0.0, i IN range(0,n-1) | a + Ys[i]*Ys[i]) AS syy
WITH condition, side, n, sx, sy, sxy, sxx, syy,
     (n*sxx - sx*sx) AS denom, (n*sxy - sx*sy) AS num, (n*syy - sy*sy) AS Syy
RETURN condition, side, n,
       round(CASE WHEN denom<>0 THEN num/denom ELSE null END,4) AS beta,
       round(CASE WHEN denom>0 AND Syy>0 THEN (num*num)/(denom*Syy) ELSE null END,4) AS R2,
       round(CASE WHEN denom<>0 THEN (num/denom)*5.0 ELSE null END,4) AS delta_for_plus5
ORDER BY condition, side
""".strip()

def cy_compare_groups(code_pattern: str, stat: str = "mean") -> str:
    return f"""
UNWIND ['ASD','TD'] AS grp
MATCH (p:Subject)-[:HAS_CONDITION]->(:Condition {{name:grp}})
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
MATCH (s)-[:HAS_VALUE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
WHERE f.stat='{stat}' AND f.code =~ '(?i).*{code_pattern}\\s*$'
WITH grp AS condition, p.pid AS pid, avg(fv.value) AS subj_mean
WITH condition, avg(subj_mean) AS mean_val
RETURN condition, round(mean_val,3) AS mean_value
ORDER BY condition
""".strip()

def cy_list_features(regex_like: str) -> str:
    # user pattern already simple ".*"
    return f"""
MATCH (f:Feature)
WHERE f.code =~ '(?i).*{regex_like}.*'
RETURN f.code AS code, f.stat AS stat
ORDER BY code LIMIT 200
""".strip()

def cy_count_participants() -> str:
    return """
MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
RETURN c.name AS condition, count(*) AS participants
ORDER BY condition
""".strip()

def cy_correlation(cond: str, side: str, a_stem: str, b_stem: str) -> str:
    def pattern(stem,side):
        if side == "BOTH":
            return f"f.code =~ '(?i).*{stem}L\\s*$' OR f.code =~ '(?i).*{stem}R\\s*$'"
        return f"f.code =~ '(?i).*{stem}{side}\\s*$'"
    cond_filter = "" if cond == "BOTH" else f"WHERE c.name='{cond}'"
    return f"""
UNWIND [{ "'L','R'" if side=='BOTH' else f"'{side}'" }] AS side
MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
{cond_filter}
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)

// A
MATCH (s)-[:HAS_VALUE]->(av:FeatureValue)-[:OF_FEATURE]->(af:Feature)
WHERE af.stat='mean' AND ({pattern(a_stem, 'L' if side=='BOTH' else side)})

// B
MATCH (s)-[:HAS_VALUE]->(bv:FeatureValue)-[:OF_FEATURE]->(bf:Feature)
WHERE bf.stat='mean' AND ({pattern(b_stem, 'L' if side=='BOTH' else side)})

WITH c.name AS condition, side, p.pid AS pid, avg(av.value) AS A, avg(bv.value) AS B
WITH condition, side, collect(A) AS As, collect(B) AS Bs
WITH condition, side, As, Bs, size(As) AS n,
     reduce(a=0.0, v IN As | a+v) AS sA,
     reduce(a=0.0, v IN Bs | a+v) AS sB,
     reduce(a=0.0, i IN range(0,n-1) | a + As[i]*Bs[i]) AS sAB,
     reduce(a=0.0, i IN range(0,n-1) | a + As[i]*As[i]) AS sAA,
     reduce(a=0.0, i IN range(0,n-1) | a + Bs[i]*Bs[i]) AS sBB
WITH condition, side, n, sA, sB, sAB, sAA, sBB,
     (n*sAB - sA*sB) AS cov_num,
     sqrt( (n*sAA - sA*sA) * (n*sBB - sB*sB) ) AS cov_den
RETURN condition, side, n,
       round(CASE WHEN cov_den<>0 THEN cov_num/cov_den ELSE null END,4) AS r
ORDER BY condition, side
""".strip()

def intent_router(user_q: str) -> Optional[Tuple[str,str]]:
    """
    Tries to understand the query and returns (cypher, tag).
    tag ∈ {"mean","variance","std","spatio","coupling","compare","features","count","corr"}
    """
    uq = user_q.lower()

    # 1) participants count
    if any(w in uq for w in ["how many participants","πόσοι συμμετέχ","count subjects","πόσα άτομα"]):
        return cy_count_participants(), "count"

    # 2) list features
    rgx = feature_regex_from_text(uq)
    if rgx:
        return cy_list_features(rgx), "features"

    # 3) spatiotemporal intent
    sp = spatiotemporal_key(uq)
    if sp:
        cond = detect_condition(uq)
        code, _unit = sp
        return cy_spatiotemporal_mean(cond, code), "spatio"

    # 4) coupling / regression
    cp = asks_coupling(uq)
    if cp:
        cond = detect_condition(uq)
        side = detect_side(uq)
        src, tgt = cp  # stems
        return cy_coupling_ols(cond, side, src, tgt), "coupling"

    # 5) correlation (pearson) between two joints if mentioned
    if "correl" in uq or "συσχε" in uq:
        # try to pick any two joint mentions
        mentioned = [JOINT_MAP[k][0] for k in JOINT_MAP if k in uq]
        if len(mentioned) >= 2:
            cond = detect_condition(uq)
            side = detect_side(uq)
            return cy_correlation(cond, side, mentioned[0], mentioned[1]), "corr"

    # 6) mean / variance / std for a joint
    j = find_joint(uq)
    if j and (asks_mean(uq) or asks_variance(uq) or asks_std(uq) or "angle" in uq or "γων" in uq):
        stem, nice = j
        cond = detect_condition(uq)
        side = detect_side(uq)
        if asks_variance(uq):
            return cy_mean_per_subject(cond, side, stem, nice, stat="variance"), "variance"
        if asks_std(uq):
            return cy_mean_per_subject(cond, side, stem, nice, stat="std"), "std"
        # default → mean per subject
        return cy_mean_per_subject(cond, side, stem, nice, stat="mean"), "mean"

    # 7) group comparison (ASD vs TD) if code-like explicit mention
    m_code = re.search(r"(hian|knfo|spkn|thhti|thh|spel|shwr|elha)[lr]?", uq)
    if m_code and ("vs" in uq or "compare" in uq or "diff" in uq or "διαφορ" in uq):
        code = m_code.group(0).upper()
        return cy_compare_groups(code_pattern=code), "compare"

    # 8) fallback: try SLM (NL2Cypher will also do its own fallback for simple knee/ankle queries)
    return None

# ---------- Clinical explanation (plain prose) ----------

def clinical_explanation(rows: List[Dict[str,Any]], tag: str) -> Optional[str]:
    if not rows: return None
    # try to summarise briefly depending on tag + available columns
    cols = set(rows[0].keys())

    if tag in {"mean","variance","std"} and {"condition","side","mean_deg","n"}.issubset(cols):
        # one or two rows per side/condition
        parts = []
        grp = {}
        for r in rows:
            grp.setdefault((r.get("condition"), r.get("side")), []).append(r)
        for (cond,side), arr in grp.items():
            m = arr[0]["mean_deg"]
            n = arr[0]["n"]
            stat_name = {"mean":"mean angle","variance":"angle variance","std":"angle std"}[tag]
            parts.append(f"For {cond}, {('left' if side=='L' else 'right')} limb: "
                         f"{stat_name} ≈ {m}°, based on ~{n} participants (per-subject means).")
        return " ".join(parts)

    if tag == "spatio" and {"condition","mean_value","n"}.issubset(cols):
        # assume ordered ASD,TD two rows
        text = []
        for r in rows:
            text.append(f"{r['condition']}: mean ≈ {r['mean_value']} (n≈{r['n']}).")
        return "Spatiotemporal summary — " + " ".join(text)

    if tag == "coupling" and {"condition","side","beta","R2","n"}.issubset(cols):
        pieces = []
        for r in rows:
            beta = r.get("beta"); r2 = r.get("R2"); n = r.get("n"); cond = r.get("condition"); side = r.get("side")
            limb = "right" if side=="R" else "left"
            pieces.append(f"{cond}, {limb}: slope β≈{beta} (Δtarget per 1° source), R²≈{r2}, n≈{n}.")
        return ("Coupling (OLS across per-subject means). "
                "Interpretation: β>1 ⇒ the target joint changes more than the source; "
                "low R² (<0.2) suggests weak coupling or high between-subject variability. "
                + " ".join(pieces))

    if tag == "compare" and {"condition","mean_value"}.issubset(cols):
        # expect 2 rows ASD/TD
        d = {r["condition"]: r["mean_value"] for r in rows}
        if "ASD" in d and "TD" in d:
            diff = round(d["ASD"] - d["TD"], 3)
            pct = round(100.0*diff/ d["TD"], 1) if d["TD"] else None
            base = f"Group comparison (per-subject means). ASD ≈ {d['ASD']}, TD ≈ {d['TD']}, Δ≈{diff}"
            if pct is not None:
                base += f" ({pct}% vs TD)."
            return base
        return "Group comparison (per-subject means)."

    if tag == "corr" and {"condition","side","r","n"}.issubset(cols):
        text = []
        for r in rows:
            side = "right" if r["side"]=="R" else "left"
            text.append(f"{r['condition']}, {side}: Pearson r≈{r['r']} (n≈{r['n']}).")
        return "Correlation across per-subject means. " + " ".join(text)

    if tag == "count" and {"condition","participants"}.issubset(cols):
        return "Participants per group: " + ", ".join(f"{r['condition']}={r['participants']}" for r in rows)

    # generic fallback
    if "n" in cols and "beta" in cols and "R2" in cols:
        return ("Regression summary: β is the slope (target change per 1° source). "
                "R² quantifies coupling strength (0–1). Higher β ⇒ stronger following; "
                "very low R² suggests noise/variability.")
    return None

# ---------- NL → Cypher → Exec ----------

st.subheader("❓ Ρώτησε σε φυσική γλώσσα")
q = st.text_area(
    "Ερώτηση (GR/EN):",
    height=110,
    placeholder="Examples: 'mean knee angle right in ASD', 'ASD vs TD velocity', 'knee->ankle coupling left', 'list features like HIAN', 'participants count'"
)

def generate_cypher(user_q: str) -> Tuple[str,str]:
    # 1) Try rich rule-based router
    routed = intent_router(user_q)
    if routed:
        cy, tag = routed
        return cy, tag

    # 2) Else use SLM (which internally also falls back for simple knee/ankle)
    cy = generator(user_q, synonyms, fewshots)
    return cy, "slm"

colA, colB = st.columns([1,1])
with colA:
    if st.button("🧠 Generate Cypher"):
        cy, tag = generate_cypher(q)
        st.session_state["last_cypher"] = cy
        st.session_state["last_tag"] = tag

with colB:
    exec_disabled = st.session_state["db"] is None
    if st.button("▶️ Execute", disabled=exec_disabled):
        cy = st.session_state.get("last_cypher","")
        if not cy.strip():
            st.warning("Δεν υπάρχει Cypher για εκτέλεση.")
        else:
            try:
                rows = st.session_state["db"].run(cy)
                st.session_state["last_rows"] = rows
                st.success(f"OK — {len(rows)} rows.")
            except neo4j_ex.Neo4jError as e:
                st.error(f"Neo4j error: {e}")
            except Exception as e:
                st.error(f"Exec error: {e}")

# Show cypher
cy = st.session_state.get("last_cypher","")
if show_cypher and cy:
    st.code(cy, language="cypher")

# Show results + clinical explanation (flow text)
rows = st.session_state.get("last_rows", [])
tag = st.session_state.get("last_tag","")
if rows:
    st.write("**Αποτελέσματα**")
    st.dataframe(pd.DataFrame(rows))
    expl = clinical_explanation(rows, tag)
    if expl:
        st.write(expl)

# ---------- Quick actions ----------
st.divider()
st.subheader("⚙️ Quick actions")

qa1, qa2, qa3 = st.columns(3)
with qa1:
    if st.button("📊 Knee→Ankle coupling (ASD & TD, both sides)"):
        cy = cy_coupling_ols("BOTH", "BOTH", "HIAN", "KNFO")
        st.session_state["last_cypher"] = cy
        st.session_state["last_tag"] = "coupling"
        if db:
            st.session_state["last_rows"] = db.run(cy)

with qa2:
    if st.button("🦵 Mean Knee angle per subject (ASD/TD, L/R)"):
        st.session_state["last_cypher"] = cy_mean_per_subject("BOTH", "BOTH", "HIAN", "Knee")
        st.session_state["last_tag"] = "mean"
        if db:
            st.session_state["last_rows"] = db.run(st.session_state["last_cypher"])

with qa3:
    if st.button("🏃 Velocity (ASD vs TD)"):
        st.session_state["last_cypher"] = cy_spatiotemporal_mean("BOTH", "Velocity")
        st.session_state["last_tag"] = "spatio"
        if db:
            st.session_state["last_rows"] = db.run(st.session_state["last_cypher"])

# (Optional) tiny ML demo kept as-is; toggle via sidebar
if enable_ml:
    st.divider()
    st.subheader("🤖 Quick ML demo (optional)")
    if st.button("ASD Right: predict Ankle from Knee/Hip/StaT/Velocity"):
        if not db:
            st.error("Connect first.")
        else:
            df = pd.DataFrame(db.run("""
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
                    st.success(f"XGB R²={r2:.4f} (in-sample).")
                    st.json(dict(zip(["knee","hip","StaT","Velocity"], model.feature_importances_.round(4))))
                except Exception:
                    from sklearn.linear_model import LinearRegression
                    m = LinearRegression().fit(X, y)
                    r2 = float(m.score(X, y))
                    st.success(f"Linear R²={r2:.4f} (in-sample).")               
                