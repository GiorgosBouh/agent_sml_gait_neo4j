#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NL → Cypher for ASD Gait graph with hybrid logic:
- Rule-based exact matches first (fast, deterministic).
- If no exact match, fallback to FREE local small language model (Ollama).
- If LLM not reachable, fallback to noop query.

Schema:
  (:Subject {pid,sid})-[:HAS_TRIAL]->(:Trial {uid})
  (:Trial)-[:HAS_FILE]->(:File {uid,kind})
  (:Trial)-[:HAS_FEATURE]->(:FeatureValue {value})-[:OF_FEATURE]->(:Feature {code,stat})
"""

import os, re, json, requests
from typing import Dict, Any, Optional

# ------------------- Config -------------------
USE_LLM = str(os.getenv("USE_LLM", "false")).lower() in ("1","true","yes")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "12.0"))

# ------------------- Templates -------------------
TEMPLATES = {
    "count": "MATCH (s:Subject) RETURN sum(CASE WHEN s.pid STARTS WITH 'ASD:' THEN 1 ELSE 0 END) AS asd_cases, sum(CASE WHEN s.pid STARTS WITH 'TD:' THEN 1 ELSE 0 END) AS td_cases;",
    "correlations": """
MATCH (a:Feature)-[r:CORRELATED_WITH]->(b:Feature)
RETURN a.code AS A, b.code AS B, r.r AS r, r.n AS n
ORDER BY abs(r) DESC, n DESC
LIMIT 20;""",
    "completeness": """
MATCH (t:Trial)
OPTIONAL MATCH (t)-[:HAS_FILE]->(f:File)
WITH t, collect(DISTINCT f.kind) AS kinds
OPTIONAL MATCH (t)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(feat:Feature)
WITH t, kinds, count(DISTINCT fv) AS nvals, count(DISTINCT feat) AS nfeats
RETURN count(*) AS trials,
       sum(CASE WHEN size(kinds)=4 THEN 1 ELSE 0 END) AS trials_all_files,
       sum(CASE WHEN nvals=463 AND nfeats=463 THEN 1 ELSE 0 END) AS trials_complete;""",
    "mean": """
MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
WHERE {where_clause}
WITH CASE WHEN s.pid STARTS WITH 'ASD:' THEN 'ASD' WHEN s.pid STARTS WITH 'TD:' THEN 'TD' ELSE 'UNK' END AS condition,
     {side_expr} AS side, s.pid AS pid, avg(fv.value) AS subj_mean
RETURN condition, side, round(avg(subj_mean),2) AS mean_value, count(*) AS n
ORDER BY condition, side;"""
}

# ------------------- Helpers (rules) -------------------
def _detect_condition(q: str) -> str:
    ql = q.lower()
    if "asd" in ql: return "ASD"
    if "td" in ql or "τυπ" in ql or "control" in ql: return "TD"
    return "BOTH"

def _detect_side(q: str) -> str:
    ql = q.lower()
    if "right" in ql or "δεξ" in ql: return "R"
    if "left" in ql or "αριστ" in ql: return "L"
    return "BOTH"

def _detect_joint(q: str) -> Optional[str]:
    ql = q.lower()
    if "knee" in ql or "γόνα" in ql: return "HIAN"
    if "hip" in ql or "ισχ" in ql: return "SPKN"
    if "ankle" in ql or "ποδο" in ql: return "KNFO"
    if "trunk" in ql or "κορμ" in ql: return "THHTI"
    if "pelvis" in ql or "λεκ" in ql or "πυελ" in ql: return "SPEL"
    return None

def _detect_intent(q: str) -> str:
    ql = q.lower()
    if "count" in ql or "πόσα" in ql: return "count"
    if "correl" in ql or "συσχ" in ql: return "correlations"
    if "complete" in ql or "πλήρη" in ql: return "completeness"
    if "compare" in ql or "σύγκρι" in ql or "vs" in ql: return "compare"
    if "mean" in ql or "average" in ql or "μέση" in ql or "μ.ο." in ql: return "mean"
    return "unknown"

# ------------------- Ollama -------------------
SYS_PROMPT = """You are an assistant that extracts slots from a clinical gait query.
Return ONLY strict JSON:
{"intent":"mean|compare|count|correlations|completeness",
 "joint":"knee|hip|ankle|trunk|pelvis",
 "side":"L|R|BOTH",
 "cond":"ASD|TD|BOTH",
 "stat":"mean|std|var"}"""

def _ollama_generate(prompt: str) -> Optional[str]:
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("response","").strip()
    except Exception:
        return None
    return None

def _llm_parse(q: str) -> Optional[Dict[str,str]]:
    if not USE_LLM: return None
    resp = _ollama_generate(SYS_PROMPT + "\nQuery: " + q)
    if not resp: return None
    try:
        js = re.search(r"\{.*\}", resp, re.DOTALL).group(0)
        return json.loads(js)
    except Exception:
        return None

# ------------------- Engine -------------------
class NL2Cypher:
    def __init__(self): pass

    def to_cypher(self, q: str) -> str:
        q = (q or "").strip()
        if not q:
            return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;"

        # 1. Rule-based
        intent = _detect_intent(q)
        cond   = _detect_condition(q)
        side   = _detect_side(q)
        joint  = _detect_joint(q)

        if intent in ("count","correlations","completeness"):
            return TEMPLATES[intent]

        if intent == "mean" and joint:
            side_expr = f"'{side}'" if side in ("L","R") else "CASE WHEN f.code =~ '(?i).*L$' THEN 'L' ELSE 'R' END"
            regex = f"(?i).*{joint}{side}\\s*$" if side in ("L","R") else f"(?i).*{joint}[LR]\\s*$"
            where_clause = f"s.pid STARTS WITH '{cond}:' AND f.stat='mean' AND f.code =~ '{regex}'" if cond!="BOTH" else f"f.stat='mean' AND f.code =~ '{regex}'"
            return TEMPLATES["mean"].format(where_clause=where_clause, side_expr=side_expr)

        # 2. If no exact match → LLM
        slots = _llm_parse(q)
        if slots:
            intent = slots.get("intent","mean").lower()
            cond   = slots.get("cond","BOTH").upper()
            side   = slots.get("side","BOTH").upper()
            joint  = slots.get("joint","knee").lower()
            stat   = slots.get("stat","mean").lower()
            if intent=="count": return TEMPLATES["count"]
            if intent=="correlations": return TEMPLATES["correlations"]
            if intent=="completeness": return TEMPLATES["completeness"]
            if intent=="mean":
                side_expr = f"'{side}'" if side in ("L","R") else "CASE WHEN f.code =~ '(?i).*L$' THEN 'L' ELSE 'R' END"
                regex = f"(?i).*{joint.upper()}{side}\\s*$" if side in ("L","R") else f"(?i).*{joint.upper()}[LR]\\s*$"
                where_clause = f"s.pid STARTS WITH '{cond}:' AND f.stat='{stat}' AND f.code =~ '{regex}'" if cond!="BOTH" else f"f.stat='{stat}' AND f.code =~ '{regex}'"
                return TEMPLATES["mean"].format(where_clause=where_clause, side_expr=side_expr)

        # 3. Fallback
        return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;"