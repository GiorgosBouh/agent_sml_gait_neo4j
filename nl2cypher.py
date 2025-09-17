#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rule-based NL → Cypher for ASD Gait graph.

Graph schema (as provided):
  (:Subject {pid,sid})-[:HAS_TRIAL]->(:Trial {uid})
  (:Trial)-[:HAS_FILE]->(:File {uid,kind})
  (:Trial)-[:HAS_FEATURE]->(:FeatureValue {value})-[:OF_FEATURE]->(:Feature {code,stat})

This module exposes:
  class NL2Cypher:
      def to_cypher(self, question: str) -> str
"""

import re
from typing import Dict, Tuple, Optional

# ----------------------- Dictionaries -----------------------
JOINT_STEMS: Dict[str, Tuple[str, str]] = {
    # keyword → (feature code stem, nice name)
    "knee": ("HIAN", "Knee"),
    "γόνα": ("HIAN", "Knee"),
    "gonato": ("HIAN", "Knee"),
    "ankle": ("KNFO", "Ankle"),
    "ποδοκν": ("KNFO", "Ankle"),
    "hip": ("SPKN", "Hip"),
    "ισχ": ("SPKN", "Hip"),
    "trunk": ("THHTI", "TrunkTilt"),
    "κορμ": ("THHTI", "TrunkTilt"),
    "spine": ("SPINE", "Spine"),
    "pelvis": ("SPEL", "Pelvis"),
}

SPATIOTEMPORAL: Dict[str, Tuple[str, str]] = {
    # label → (Feature.code, unit)
    "velocity": ("Velocity", "m/s"),
    "stance": ("StaT", "ms"),
    "swing": ("SwiT", "ms"),
    "gait cycle": ("GaCT", "ms"),
    "stride length": ("StrLe", "m"),
    "step length": ("MaxStLe", "m"),
    "step width": ("MaxStWi", "m"),
}

# ----------------------- Helpers -----------------------
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

def detect_joint(q: str) -> Optional[Tuple[str, str]]:
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

def asks_coupling(q: str) -> Optional[Tuple[str, str]]:
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

# ----------------------- Cypher templates -----------------------
def cy_unwind_sides(side: str) -> str:
    return "WITH CASE WHEN '{s}'='BOTH' THEN ['L','R'] ELSE ['{s}'] END AS sides\nUNWIND sides AS side".format(s=side)

def cy_mean_per_subject(cond: str, side: str, joint_stem: str, stat: str = "mean") -> str:
    if side == "BOTH":
        side_filter = f"(f.code =~ '(?i).*{joint_stem}L\\s*$' OR f.code =~ '(?i).*{joint_stem}R\\s*$')"
        side_expr = "CASE WHEN f.code =~ '(?i).*L\\s*$' THEN 'L' ELSE 'R' END"
    else:
        side_filter = f"f.code =~ '(?i).*{joint_stem}{side}\\s*$'"
        side_expr = f"'{side}'"
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

def cy_compare_groups(code_pattern: str, stat: str = "mean") -> str:
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
    return """
MATCH (s:Subject)
RETURN
  sum(CASE WHEN s.pid STARTS WITH 'ASD:' THEN 1 ELSE 0 END) AS asd_cases,
  sum(CASE WHEN s.pid STARTS WITH 'TD:'  THEN 1 ELSE 0 END) AS td_cases;
""".strip()

# ----------------------- NL engine -----------------------
class NL2Cypher:
    """
    Minimal, deterministic NL → Cypher engine.
    Usage:
        engine = NL2Cypher()
        cypher = engine.to_cypher("compare ASD vs TD knee right mean")
    """
    def __init__(self) -> None:
        pass

    def to_cypher(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;"

        # counts
        if re.search(r"\b(count|how many|πόσα)\b.*\b(asd|td|subjects|participants|cases)\b", q, flags=re.IGNORECASE):
            return cy_count_subjects()

        # list features like/regex
        rgx = feature_regex_from_text(q)
        if rgx:
            return cy_list_features(rgx)

        # spatiotemporal
        sp = spatiotemporal_key(q)
        if sp:
            cond = detect_condition(q)
            code, _unit = SPATIOTEMPORAL[sp]
            return cy_spatiotemporal_mean(cond, code)

        # coupling/regression between joints
        cp = asks_coupling(q)
        if cp:
            cond = detect_condition(q)
            side = detect_side(q)
            a_stem, b_stem = cp
            return cy_coupling_ols(cond, side, a_stem, b_stem)

        # compare ASD vs TD for a joint (optionally side)
        if re.search(r"(compare|vs|diff|σύγκρι|διαφορ)", q, flags=re.IGNORECASE):
            side = detect_side(q)
            js = detect_joint(q) or ("HIAN", "Knee")
            code_pat = f"{js[0]}{'' if side=='BOTH' else side}"
            return cy_compare_groups(code_pat)

        # mean for joint by side/cond
        if re.search(r"(mean|avg|average|μ\.?ο\.?|μέση)", q, flags=re.IGNORECASE) or detect_joint(q):
            cond = detect_condition(q)
            side = detect_side(q)
            js = detect_joint(q) or ("HIAN", "Knee")
            return cy_mean_per_subject(cond, side, js[0], "mean")

        # fallback
        return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;"