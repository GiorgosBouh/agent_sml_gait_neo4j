#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
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
    st.caption("Small free SLM (HF FLAN-T5-Small) — χρησιμοποιείται μόνο όταν δεν αναγνωριστεί το intent από τους κανόνες.")
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

# ---------- Dictionaries / parsing helpers ----------
JOINT_MAP = {
    # joint keyword -> (code stem, nice name)
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
    "thorax":  ("THHTI","Thorax"),
    "θωρακ":   ("THHTI","Thorax"),
    "θωρακα":  ("THHTI","Thorax"),
    "στήθος":  ("THHTI","Thorax"),
    "chest":   ("THHTI","Thorax"),
}

SPATIOTEMPORAL = {
    "stride length": ("StrLe", "m"),
    "step length":   ("MaxStLe", "m"),
    "step width":    ("MaxStWi", "m"),
    "gait cycle":    ("GaCT", "ms"),
    "stance":        ("StaT", "ms"),
    "swing":         ("SwiT", "ms"),
    "velocity":      ("Velocity", "m/s"),
}

def detect_condition(uq: str) -> str:
    # explicit two-group intents (and/&/, in any order)
    if re.search(r'\basd\b.*(?:and|&|,)\s*(?:td|typical)\b', uq) or \
       re.search(r'\b(?:td|typical)\b.*(?:and|&|,)\s*asd\b', uq):
        return "BOTH"
    # vs/compare
    if re.search(r'(asd\s*vs\s*td|td\s*vs\s*asd|compare|diff|διαφορ)', uq):
        return "BOTH"
    # single-group fallbacks
    if re.search(r'\basd\b|αυτισ', uq): return "ASD"
    if re.search(r'\btd\b|typical|τυπικ', uq): return "TD"
    return "ASD"

def detect_side(uq: str) -> str:
    if re.search(r"(right|δεξ)", uq): return "R"
    if re.search(r"(left|αρισ|αρ\.)", uq): return "L"
    if re.search(r"(both|και τα δυο|bilateral)", uq): return "BOTH"
    return "R"

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
    pairs = [
        (["knee","γονατ"], ["ankle","ποδοκν"]),
        (["hip","ισχυ"],   ["knee","γονατ"]),
        (["knee","γονατ"], ["trunk","κορμ"]),
    ]
    for src_list, tgt_list in pairs:
        if any(k in uq for k in src_list) and any(k in uq for k in tgt_list):
            src = next(JOINT_MAP[k][0] for k in JOINT_MAP if k in uq and k in src_list)
            tgt = next(JOINT_MAP[k][0] for k in JOINT_MAP if k in uq and k in tgt_list)
            return src, tgt
    if any(w in uq for w in ["coupling","συσχ","συσχετ","regression","ols"]):
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
        pat = m.group(3).replace("*", ".*")
        return pat
    return None

# ---------- Intent → Cypher (Rule-based) ----------

def cy_mean_per_subject(cond: str, side: str, joint_stem: str, joint_name: str, stat="mean") -> str:
    if side == "BOTH":
        side_filter = f"(f.code =~ '(?i).*{joint_stem}L\\s*$' OR f.code =~ '(?i).*{joint_stem}R\\s*$')"
        side_expr   = "CASE WHEN f.code =~ '(?i).*L\\s*$' THEN 'L' ELSE 'R' END"
    else:
        side_filter = f"f.code =~ '(?i).*{joint_stem}{side}\\s*$'"
        side_expr   = f"'{side}'"

    cond_filter = "" if cond == "BOTH" else f"WHERE c.name='{cond}'"

    return f"""
MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
{cond_filter}
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
MATCH (s)-[:HAS_VALUE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
WHERE f.stat='{stat}' AND {side_filter}
WITH c.name AS condition,
     {side_expr} AS side,
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

def cy_unwind_sides(side: str) -> str:
    # αξιόπιστο UNWIND για μία ή δύο πλευρές
    return f"WITH CASE WHEN '{side}'='BOTH' THEN ['L','R'] ELSE ['{side}'] END AS sides\nUNWIND sides AS side"

def cy_coupling_ols(cond: str, side: str, src_stem: str, tgt_stem: str) -> str:
    cond_filter = "" if cond == "BOTH" else f"WHERE c.name='{cond}'"
    return f"""
{cy_unwind_sides(side)}
WITH side, '{src_stem}' AS src_stem, '{tgt_stem}' AS tgt_stem
MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
{cond_filter}
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
/* source X */
MATCH (s)-[:HAS_VALUE]->(xv:FeatureValue)-[:OF_FEATURE]->(xf:Feature)
WHERE xf.stat='mean' AND xf.code =~ ('(?i).*' + src_stem + side + '\\s*$')
/* target Y */
MATCH (s)-[:HAS_VALUE]->(yv:FeatureValue)-[:OF_FEATURE]->(yf:Feature)
WHERE yf.stat='mean' AND yf.code =~ ('(?i).*' + tgt_stem + side + '\\s*$')
WITH c.name AS condition, side, p.pid AS pid, avg(xv.value) AS X, avg(yv.value) AS Y, src_stem, tgt_stem
WITH condition, side, src_stem, tgt_stem, collect({{x:X, y:Y}}) AS pairs
UNWIND pairs AS p
WITH condition, side, src_stem, tgt_stem,
     count(*) AS n,
     sum(p.x) AS sx,
     sum(p.y) AS sy,
     sum(p.x*p.y) AS sxy,
     sum(p.x*p.x) AS sxx,
     sum(p.y*p.y) AS syy
WITH condition, side, src_stem, tgt_stem, n, sx, sy, sxy, sxx, syy,
     (n*sxx - sx*sx) AS denom,
     (n*sxy - sx*sy) AS num,
     (n*syy - sy*sy) AS Syy
RETURN condition, side, src_stem, tgt_stem, n,
       round(CASE WHEN denom<>0 THEN num/denom ELSE null END,4) AS beta,
       round(CASE WHEN denom>0 AND Syy>0 THEN (num*num)/(denom*Syy) ELSE null END,4) AS R2,
       round(CASE WHEN denom<>0 THEN (num/denom)*5.0 ELSE null END,4) AS delta_for_plus5,
       round(CASE WHEN denom<>0 THEN (num/denom)*10.0 ELSE null END,4) AS delta_for_plus10
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
    cond_filter = "" if cond == "BOTH" else f"WHERE c.name='{cond}'"
    return f"""
{cy_unwind_sides(side)}
WITH side, '{a_stem}' AS a_stem, '{b_stem}' AS b_stem
MATCH (p:Subject)-[:HAS_CONDITION]->(c:Condition)
{cond_filter}
MATCH (p)-[:HAS_SAMPLE]->(s:Sample)
/* A */
MATCH (s)-[:HAS_VALUE]->(av:FeatureValue)-[:OF_FEATURE]->(af:Feature)
WHERE af.stat='mean' AND af.code =~ ('(?i).*' + a_stem + side + '\\s*$')
/* B */
MATCH (s)-[:HAS_VALUE]->(bv:FeatureValue)-[:OF_FEATURE]->(bf:Feature)
WHERE bf.stat='mean' AND bf.code =~ ('(?i).*' + b_stem + side + '\\s*$')
WITH c.name AS condition, side, p.pid AS pid, avg(av.value) AS A, avg(bv.value) AS B
WITH condition, side, collect({{a:A, b:B}}) AS pairs
UNWIND pairs AS p
WITH condition, side,
     count(*) AS n,
     sum(p.a) AS sA,
     sum(p.b) AS sB,
     sum(p.a*p.b) AS sAB,
     sum(p.a*p.a) AS sAA,
     sum(p.b*p.b) AS sBB
WITH condition, side, n, sA, sB, sAB, sAA, sBB,
     (n*sAB - sA*sB) AS cov_num,
     sqrt( (n*sAA - sA*sA) * (n*sBB - sB*sB) ) AS cov_den
RETURN condition, side, n,
       round(CASE WHEN cov_den<>0 THEN cov_num/cov_den ELSE null END,4) AS r
ORDER BY condition, side
""".strip()

def intent_router(user_q: str) -> Optional[Tuple[str,str]]:
    """
    Returns (cypher, tag).
    tag ∈ {"mean","variance","std","spatio","coupling","compare","features","count","corr"}
    """
    uq = (user_q or "").lower().strip()
    if not uq:
        return None

    # 1) participants count
    if any(w in uq for w in ["how many participants","πόσοι συμμετέχ","count subjects","πόσα άτομα","how many asd","how many td"]):
        return cy_count_participants(), "count"

    # 2) list features
    rgx = feature_regex_from_text(uq)
    if rgx:
        return cy_list_features(rgx), "features"

    # 3) spatiotemporal
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
        src, tgt = cp
        return cy_coupling_ols(cond, side, src, tgt), "coupling"

    # 5) correlation
    if "correl" in uq or "συσχε" in uq:
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
        return cy_mean_per_subject(cond, side, stem, nice, stat="mean"), "mean"

    # 7) group comparison by explicit code pattern
    m_code = re.search(r"(hian|knfo|spkn|thhti|thh|spel|shwr|elha)[lr]?", uq)
    if m_code and ("vs" in uq or "compare" in uq or "diff" in uq or "διαφορ" in uq):
        code = m_code.group(0).upper()
        return cy_compare_groups(code_pattern=code), "compare"

    # 8) fallback → SLM
    return None

# ---------- Clinical explanation (plain prose) ----------

def clinical_explanation(rows: List[Dict[str,Any]], tag: str) -> Optional[str]:
    if not rows:
        return None

    def cond_text(cond: str) -> str:
        return {"ASD":"children with autism (ASD)",
                "TD":"typically developing children (TD)"}\
               .get(cond or "", cond or "the group")

    def side_text(side: Optional[str]) -> str:
        return {"L":"on the left side", "R":"on the right side"}.get(side or "", "")

    # helper για όνομα άρθρωσης από stem
    STEM2NAME = {
        "HIAN":"knee", "KNFO":"ankle", "SPKN":"hip", "THHTI":"trunk tilt",
        "SPEL":"pelvis", "THH":"spine", "SHWR":"shoulder", "ELHA":"elbow"
    }

    cols = set(rows[0].keys())

    # --- Coupling/Regression με ονόματα και +10° ---
    if tag == "coupling" and {"condition","side","beta","R2","n"}.issubset(cols):
        parts = []
        for r in rows:
            group = cond_text(r.get("condition"))
            side  = side_text(r.get("side"))
            beta  = r.get("beta")
            r2    = r.get("R2")
            n     = r.get("n")
            d5    = r.get("delta_for_plus5")
            d10   = r.get("delta_for_plus10")
            src   = STEM2NAME.get(str(r.get("src_stem","")).upper(), "source joint")
            tgt   = STEM2NAME.get(str(r.get("tgt_stem","")).upper(), "target joint")

            # verbal trend
            if beta is None:
                trend = f"we cannot estimate how the {tgt} changes when the {src} changes"
            else:
                b = float(beta)
                if b < -0.2:
                    trend = f"the {tgt} tends to move in the opposite direction to the {src}"
                elif -0.2 <= b < 0.2:
                    trend = f"the {tgt} changes very little when the {src} changes"
                elif 0.2 <= b < 0.8:
                    trend = f"the {tgt} follows somewhat when the {src} changes"
                elif 0.8 <= b <= 1.2:
                    trend = f"the {tgt} follows the {src} almost one-to-one"
                else:
                    trend = f"the {tgt} tends to move even more than the {src}"

            nums = []
            if beta is not None:
                nums.append(f"β={float(beta):.3f} (≈ {tgt} change per +1° at the {src})")
            if r2 is not None:
                nums.append(f"R²={float(r2):.3f} (≈ {round(100*float(r2))}% of between-child variation explained)")
            if d5 is not None:
                nums.append(f"~{float(d5):+.2f}° at the {tgt} for +5° at the {src}")
            if d10 is not None:
                nums.append(f"~{float(d10):+.2f}° at the {tgt} for +10° at the {src}")

            parts.append(
                f"In {group}{(', ' + side) if side else ''}: "
                f"when the {src} increases, {trend}. "
                f"Numbers: {'; '.join(nums)}. (about {n} children)."
            )
        return " ".join(parts)

    # — τα υπόλοιπα branches (mean/spatio/compare/corr/count) άφησέ τα όπως τα έχεις —
    # Fallbacks…
    return None


# ---------- NL → Cypher → Exec ----------
st.subheader("❓ Ρώτησε σε φυσική γλώσσα")
q = st.text_area(
    "Ερώτηση (GR/EN):",
    height=110,
    placeholder="Examples: 'mean knee angle right in ASD', 'ASD vs TD velocity', 'knee->ankle coupling left', 'list features like HIAN', 'participants count'"
)

def generate_cypher(user_q: str) -> Tuple[str, str]:
    """
    Returns (cypher, tag).

    Order:
      1) Rule-based intent router (fast, deterministic).
      2) SLM (NL2Cypher) fallback.
         If SLM output is empty/invalid (not starting with a Cypher keyword),
         return a safe default query (mean right knee in ASD).
    """
    # 1) Rule-based
    routed = intent_router(user_q)
    if routed:
        cy, tag = routed
        return cy, tag

    # 2) SLM fallback
    cy = generator(user_q, synonyms, fewshots)

    # Validate SLM output looks like Cypher
    is_valid = isinstance(cy, str) and bool(re.search(r'(?is)^\s*(MATCH|WITH|CALL|UNWIND|RETURN|CREATE|MERGE)\b', cy))
    if not is_valid:
        # Safe default: per-subject mean, ASD, Right knee (HIANR)
        return cy_mean_per_subject("ASD", "R", "HIAN", "Knee", stat="mean"), "mean"

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

# Show results + clinical explanation
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

# (Optional) tiny ML demo; toggle via sidebar
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