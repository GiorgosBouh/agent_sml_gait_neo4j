#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NL → Cypher for ASD Gait graph with JSON-backed catalog.

Χρησιμοποιεί τα αρχεία export του γράφου:
  - nodes.ndjson.gz  (μία γραμμή/κόμβος: {"eid","labels","props"})

Ο GraphCatalog διαβάζει ΜΟΝΟ τους κόμβους Feature & Subject και συνάγει:
  - joints από f.props.joint_guess (π.χ. "hip","knee"...), ΑΛΛΑ έχουμε και fallback σε stem του code
  - διαθέσιμες πλευρές ανά joint (L/R) από κατάληξη f.props.code
  - διαθέσιμα stats ανά joint (mean/std/var) από f.props.stat
  - πλήθος Subjects ανά ομάδα (ASD/TD) από s.props.pid prefix

Το NL2Cypher παράγει Cypher με:
  - φίλτρο joint ως: (joint_guess==joint) **ή** code-stem (π.χ. HIAN για knee)
  - side φίλτρο με regex `…L$` ή `…R$` (κρατάμε την επιλογή χρήστη ακόμη κι αν ο κατάλογος δεν ξέρει πλευρές)
  - single WHERE (όχι διπλά WHERE)
  - προαιρετικό Ollama fallback (USE_LLM=true) μόνο αν δεν βρεθεί σαφής κανόνας.
"""

import os
import re
import json
import gzip
from typing import Dict, Any, Optional, Set, Tuple, List

# ---------- Optional LLM (Ollama) ----------
import requests
USE_LLM = str(os.getenv("USE_LLM", "false")).lower() in ("1", "true", "yes", "on")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "8.0"))

_SYS_PROMPT = """You extract structured slots from a short clinical gait query.
Return ONLY strict JSON with keys:
{"intent":"mean|compare|count|correlations|completeness",
 "joint":"knee|hip|ankle|trunk|spine|pelvis",
 "side":"L|R|BOTH",
 "cond":"ASD|TD|BOTH",
 "stat":"mean|std|var"}"""

def _ollama_generate(prompt: str) -> Optional[str]:
    if not USE_LLM:
        return None
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:
        pass
    return None

def _llm_parse_slots(question: str) -> Optional[Dict[str, str]]:
    resp = _ollama_generate(_SYS_PROMPT + "\nQuery: " + question)
    if not resp:
        return None
    try:
        m = re.search(r"\{.*\}", resp, flags=re.DOTALL)
        js = m.group(0) if m else resp
        obj = json.loads(js)
        # defaults
        obj.setdefault("intent", "mean")
        obj.setdefault("joint", "knee")
        obj.setdefault("side", "BOTH")
        obj.setdefault("cond", "BOTH")
        obj.setdefault("stat", "mean")
        return obj
    except Exception:
        return None

# ---------- Graph Catalog (from nodes.ndjson.gz) ----------

class GraphCatalog:
    """
    Διαβάζει nodes.ndjson.gz και κρατά metadata για NL→Cypher.
    Κάθε γραμμή: {"eid":..., "labels":[...], "props":{...}}
    Περιμένουμε:
      Subject: props.pid (π.χ. "ASD:26", "TD:8")
      Feature: props.code, props.stat, props.joint_guess
    """

    def __init__(self, data_dir: str = ".", nodes_file: Optional[str] = None):
        self.data_dir = data_dir
        self.nodes_path = nodes_file or os.path.join(data_dir, "nodes.ndjson.gz")

        # Derived metadata
        self.joints: Set[str] = set()                      # πχ {'hip','knee',...}
        self.joint_sides: Dict[str, Set[str]] = {}         # {'hip': {'L','R'}, ...}
        self.joint_stats: Dict[str, Set[str]] = {}         # {'hip': {'mean','std',...}}
        self.subject_counts: Dict[str, int] = {"ASD": 0, "TD": 0}

        self._loaded: bool = False

    def _side_from_code(self, code: str) -> Optional[str]:
        if not isinstance(code, str):
            return None
        code = code.strip()
        if re.search(r"(?i)\bL\s*$", code):
            return "L"
        if re.search(r"(?i)\bR\s*$", code):
            return "R"
        return None

    def load(self) -> None:
        if self._loaded:
            return
        if not os.path.exists(self.nodes_path):
            raise FileNotFoundError(f"nodes file not found: {self.nodes_path}")

        with gzip.open(self.nodes_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    labels = obj.get("labels", []) or []
                    props = obj.get("props", {}) or {}
                except Exception:
                    continue

                # Subjects → ASD/TD counters
                if "Subject" in labels:
                    pid = props.get("pid")
                    if isinstance(pid, str):
                        if pid.startswith("ASD:"):
                            self.subject_counts["ASD"] += 1
                        elif pid.startswith("TD:"):
                            self.subject_counts["TD"] += 1

                # Features → joints/sides/stats
                if "Feature" in labels:
                    stat = props.get("stat")
                    code = props.get("code")
                    jg = props.get("joint_guess")
                    if isinstance(jg, str) and jg.strip():
                        joint = jg.strip().lower()
                        self.joints.add(joint)
                        if isinstance(stat, str):
                            self.joint_stats.setdefault(joint, set()).add(stat.strip().lower())
                        side = self._side_from_code(code) if isinstance(code, str) else None
                        if side in ("L", "R"):
                            self.joint_sides.setdefault(joint, set()).add(side)

        self._loaded = True

    # ---- helpers ----
    def has_joint(self, joint: str) -> bool:
        return joint.lower() in self.joints

    def available_sides(self, joint: str) -> Set[str]:
        return self.joint_sides.get(joint.lower(), set())

    def has_stat(self, joint: str, stat: str) -> bool:
        return stat.lower() in self.joint_stats.get(joint.lower(), set())

    def guess_best_stat(self, joint: str, desired: str) -> str:
        js = self.joint_stats.get(joint.lower(), set())
        if desired in js:
            return desired
        if "mean" in js:
            return "mean"
        return next(iter(js), "mean")


# ---------- NL rules ----------

JOINT_ALIASES: Dict[str, List[str]] = {
    "knee":   ["knee", "γόνα", "gonato"],
    "hip":    ["hip", "ισχ", "ischio"],
    "ankle":  ["ankle", "ποδο", "podo", "ankl"],
    "trunk":  ["trunk", "κορμ", "trunk tilt"],
    "spine":  ["spine", "σπονδ"],
    "pelvis": ["pelvis", "λεκ", "πυελ"],
}

# stem ανά joint για fallback όταν λείπει/δεν ταιριάζει το joint_guess
JOINT_CODE_STEM = {
    "knee": "HIAN",
    "hip": "SPKN",
    "ankle": "KNFO",
    "trunk": "THHTI",
    # πρόσθεσε αν γνωρίζεις stems για spine/pelvis
}

def _detect_cond(q: str) -> str:
    t = q.lower()
    if "asd" in t or "αυτισ" in t: return "ASD"
    if "td" in t or "τυπικ" in t or "control" in t: return "TD"
    return "BOTH"

def _detect_side(q: str) -> str:
    t = q.lower()
    if re.search(r"\b(right|δεξ|dexi|\br\b)", t): return "R"
    if re.search(r"\b(left|αριστ|\bl\b)", t): return "L"
    return "BOTH"

def _detect_joint(q: str, catalog: GraphCatalog) -> Optional[str]:
    t = q.lower()
    for canon, aliases in JOINT_ALIASES.items():
        if any(a in t for a in aliases):
            return canon if (catalog.has_joint(canon) or canon in JOINT_CODE_STEM) else None
    for j in sorted(catalog.joints):
        if j in t:
            return j
    return None

def _detect_stat(q: str) -> str:
    t = q.lower()
    if "std" in t or "stdev" in t: return "std"
    if "var" in t or "variance" in t or "διακύ" in t: return "var"
    return "mean"

def _detect_intent(q: str) -> str:
    t = q.lower()
    if "count" in t or "πόσα" in t: return "count"
    if "correl" in t or "συσχετ" in t: return "correlations"
    if "complete" in t or "πληρό" in t or "kinds" in t: return "completeness"
    if "compare" in t or "σύγκρι" in t or " vs " in t: return "compare"
    if "mean" in t or "average" in t or "μ.ο" in t or "μέση" in t: return "mean"
    return "mean"

# ---------- Joint filter helper (joint_guess OR code stem) ----------

def _joint_filter(joint: str) -> str:
    stem = JOINT_CODE_STEM.get(joint.lower(), "")
    if stem:
        return (
            f"( (exists(f.joint_guess) AND toLower(f.joint_guess)='{joint.lower()}') "
            f"OR f.code =~ '(?i).*{stem}.*' )"
        )
    return f"(exists(f.joint_guess) AND toLower(f.joint_guess)='{joint.lower()}')"

# ---------- Cypher templates ----------

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
WITH CASE WHEN s.pid STARTS WITH 'ASD:' THEN 'ASD'
          WHEN s.pid STARTS WITH 'TD:'  THEN 'TD'  ELSE 'UNK' END AS condition,
     {side_expr} AS side,
     s.pid AS pid,
     avg(fv.value) AS subj_mean
RETURN condition, side, round(avg(subj_mean),2) AS mean_value, count(*) AS n
ORDER BY condition, side;""",
    "compare_asd_td": """
UNWIND {sides} AS side
MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
WHERE f.stat='{stat}' AND {joint_filter} AND (
  (side='L' AND f.code =~ '(?i).*L\\s*$') OR
  (side='R' AND f.code =~ '(?i).*R\\s*$')
)
  AND s.pid STARTS WITH 'ASD:'
WITH 'ASD' AS cond, side, avg(fv.value) AS avg_val
RETURN cond, side, avg_val
UNION ALL
UNWIND {sides} AS side
MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(f:Feature)
WHERE f.stat='{stat}' AND {joint_filter} AND (
  (side='L' AND f.code =~ '(?i).*L\\s*$') OR
  (side='R' AND f.code =~ '(?i).*R\\s*$')
)
  AND s.pid STARTS WITH 'TD:'
WITH 'TD' AS cond, side, avg(fv.value) AS avg_val
RETURN cond, side, avg_val
ORDER BY cond, side;"""
}

# ---------- Engine ----------

class NL2Cypher:
    """Φτιάχνει Cypher αξιοποιώντας GraphCatalog από JSON export του γράφου + stem fallback."""

    def __init__(self, data_dir: Optional[str] = None, nodes_file: Optional[str] = None, catalog: Optional[GraphCatalog] = None):
        if catalog is not None:
            self.catalog = catalog
        else:
            dd = data_dir or os.getenv("GRAPH_JSON_DIR", ".")
            self.catalog = GraphCatalog(dd, nodes_file=nodes_file)
            self.catalog.load()

    def to_cypher(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;"

        # Κανόνες πάνω στον κατάλογο (και με stem fallback)
        intent = _detect_intent(q)
        cond = _detect_cond(q)
        side = _detect_side(q)
        joint = _detect_joint(q, self.catalog) or self._fallback_joint()
        stat = _detect_stat(q)
        stat = self.catalog.guess_best_stat(joint, stat)

        if intent == "count":
            return TEMPLATES["count"].strip()
        if intent == "correlations":
            return TEMPLATES["correlations"].strip()
        if intent == "completeness":
            return TEMPLATES["completeness"].strip()
        if intent == "compare":
            sides_val = "['L','R']" if side == "BOTH" else f"['{side}']"
            return TEMPLATES["compare_asd_td"].format(
                sides=sides_val, stat=stat, joint_filter=_joint_filter(joint)
            ).strip()

        # default: mean
        where_parts = []
        if cond != "BOTH":
            where_parts.append(f"s.pid STARTS WITH '{cond}:'")
        where_parts.append(f"f.stat='{stat}'")

        jf = _joint_filter(joint)

        if side == "BOTH":
            where_parts.append(f"{jf} AND (f.code =~ '(?i).*L\\s*$' OR f.code =~ '(?i).*R\\s*$')")
            side_expr = "CASE WHEN f.code =~ '(?i).*L\\s*$' THEN 'L' ELSE 'R' END"
        else:
            where_parts.append(f"{jf} AND f.code =~ '(?i).*{side}\\s*$'")
            side_expr = f"'{side}'"

        where_clause = " AND ".join(where_parts)
        return TEMPLATES["mean"].format(where_clause=where_clause, side_expr=side_expr).strip()

    def _fallback_joint(self) -> str:
        # Αν δεν ανέφερε joint, ξεκίνα με knee αν υπάρχει stem, αλλιώς με ό,τι υπάρχει στον κατάλογο
        if "knee" in JOINT_CODE_STEM:
            return "knee"
        if "knee" in self.catalog.joints:
            return "knee"
        return next(iter(sorted(self.catalog.joints)), "knee")