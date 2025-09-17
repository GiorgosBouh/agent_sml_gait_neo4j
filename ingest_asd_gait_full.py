#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NL → Cypher for ASD Gait Knowledge Graph
Schema (as confirmed):
  (:Subject {pid,sid,pid_num?})-[:HAS_TRIAL]->(:Trial {uid})
  (:Trial)-[:HAS_FILE]->(:File {uid,kind})
  (:Trial)-[:HAS_FEATURE]->(:FeatureValue {value})-[:OF_FEATURE]->(:Feature {code,stat})
Notes:
  - Condition inferred from Subject.pid prefix: 'ASD:' or 'TD:'
  - Features filtered by Feature.stat and regex on Feature.code (case-insensitive)
  - Sides encoded as suffix in Feature.code (e.g., HIANL / HIANR)
"""

import os
import re
import json
import sys
from typing import Dict, List, Optional, Tuple, Any

# ------------- Defaults (can be overridden by external JSON files) ---------------- #

DEFAULT_INTENTS: List[Dict[str, Any]] = [
    # mean/average for a feature (joint+side) within a condition (ASD/TD)
    {
        "intent": "mean_of_feature_by_cond_side_joint",
        "patterns": [
            r"(?i)\b(mean|average|avg|μ\.?ο\.?|μέση)\b",
            r"(?i)\b(asd|td|τυπικ|\bcontrol\b)\b",
            r"(?i)\b(knee|γόνατο|hip|ισχίο|ankle|ποδοκνημ|trunk|κορμός|spine|σπονδυλικ)\b"
        ]
    },
    # compare ASD vs TD for a feature (joint+side)
    {
        "intent": "compare_asd_td_feature",
        "patterns": [
            r"(?i)\b(compare|σύγκρι|διαφορ(?:ά|ες)|vs|εναντίον)\b",
            r"(?i)\b(asd|td)\b",
            r"(?i)\b(knee|γόνατο|hip|ισχίο|ankle|ποδοκνημ|trunk|κορμός|spine|σπονδυλικ)\b"
        ]
    },
    # list top correlated feature pairs
    {
        "intent": "top_correlated_pairs",
        "patterns": [
            r"(?i)\b(top|most|υψηλ)\b",
            r"(?i)\b(correl|συσχετ)\b"
        ]
    },
    # trials completeness (4 kinds & 463 features)
    {
        "intent": "list_trials_files_complete",
        "patterns": [
            r"(?i)trials?.*files?|αρχε[ίι]α|kinds|πλ[ήη]ρη",
            r"(?i)\bcomplete|όλα\b"
        ]
    },
    # count nodes / relationships
    {
        "intent": "count_nodes_labels",
        "patterns": [
            r"(?i)\bcount|πόσα|μέτρησε\b",
            r"(?i)\bsubjects?|trials?|files?|features?|featurevalues?|κόμβοι|σχέσεις\b"
        ]
    }
]

# Feature code regexes — ADAPT if your codes differ
DEFAULT_FEATURES: Dict[str, Any] = {
    "joint_aliases": {
        "knee": ["knee", "γόνατο", "gonato"],
        "hip": ["hip", "ισχίο", "ischio"],
        "ankle": ["ankle", "ποδοκνημ", "podo"],
        "trunk": ["trunk", "κορμός", "trunk tilt"],
        "spine": ["spine", "σπονδυλ", "midspine", "spinebase"]
    },
    # Map joint_side → regex on Feature.code (case-insensitive). Adjust to your codes!
    "code_regex": {
        "knee_L": r"(?i).*HIANL\s*$",
        "knee_R": r"(?i).*HIANR\s*$",
        "hip_L": r"(?i).*SPKNL\s*$",
        "hip_R": r"(?i).*SPKNR\s*$",
        "ankle_L": r"(?i).*KNFOL\s*$",
        "ankle_R": r"(?i).*KNFOR\s*$",
        "trunk_L": r"(?i).*THHTIL\s*$",
        "trunk_R": r"(?i).*THHTIR\s*$",
        "spine_L": r"(?i).*SPINEL\s*$",
        "spine_R": r"(?i).*SPINER\s*$"
    },
    "metrics": {
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
        "μ.ο.": "mean",
        "μέση": "mean",
        "std": "std",
        "standard deviation": "std",
        "variance": "var",
        "var": "var"
    }
}

DEFAULT_TEMPLATES: Dict[str, str] = {
    "mean_of_feature_by_cond_side_joint": """
MATCH (s:Subject)-[:HAS_TRIAL]->(t:Trial)-[:HAS_FEATURE]->(xv:FeatureValue)-[:OF_FEATURE]->(xf:Feature)
WHERE s.pid STARTS WITH '{cond}:'
  AND xf.stat = '{stat}'
  AND xf.code =~ '{code_regex}'
RETURN s.pid AS subject_pid, avg(xv.value) AS mean_value
ORDER BY subject_pid;
""",
    "compare_asd_td_feature": """
CALL {
  MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(xv:FeatureValue)-[:OF_FEATURE]->(xf:Feature)
  WHERE s.pid STARTS WITH 'ASD:' AND xf.stat = '{stat}' AND xf.code =~ '{code_regex}'
  RETURN 'ASD' AS cond, avg(xv.value) AS avg_val
}
CALL {
  MATCH (s:Subject)-[:HAS_TRIAL]->(:Trial)-[:HAS_FEATURE]->(xv:FeatureValue)-[:OF_FEATURE]->(xf:Feature)
  WHERE s.pid STARTS WITH 'TD:' AND xf.stat = '{stat}' AND xf.code =~ '{code_regex}'
  RETURN 'TD' AS cond, avg(xv.value) AS avg_val
}
RETURN cond, avg_val
ORDER BY cond;
""",
    "top_correlated_pairs": """
MATCH (a:Feature)-[r:CORRELATED_WITH]->(b:Feature)
RETURN a.code AS A, b.code AS B, r.r AS r, r.n AS n
ORDER BY abs(r) DESC, n DESC
LIMIT 20;
""",
    "list_trials_files_complete": """
MATCH (t:Trial)
OPTIONAL MATCH (t)-[:HAS_FILE]->(f:File)
WITH t, collect(DISTINCT f.kind) AS kinds
OPTIONAL MATCH (t)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(feat:Feature)
WITH t, kinds, count(DISTINCT fv) AS nvals, count(DISTINCT feat) AS nfeats
RETURN t.uid AS trial_uid,
       size(kinds) AS kinds_n,
       (CASE WHEN size(kinds)=4 THEN true ELSE false END) AS has_all_4_kinds,
       (CASE WHEN nvals=463 AND nfeats=463 THEN true ELSE false END) AS is_complete;
""",
    "count_nodes_labels": """
CALL {
  MATCH (n) RETURN count(n) AS nodes
}
CALL {
  MATCH ()-[r]->() RETURN count(r) AS rels
}
RETURN nodes, rels;
"""
}

# ----------------------------- Utilities --------------------------------------- #

def _load_json_if_exists(path: str, default: Any) -> Any:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _compile_patterns(intents: List[Dict[str, Any]]) -> List[Tuple[str, List[re.Pattern]]]:
    compiled = []
    for item in intents:
        pats = [re.compile(p) for p in item.get("patterns", [])]
        compiled.append((item.get("intent", ""), pats))
    return compiled

def _text(s: Optional[str]) -> str:
    return (s or "").strip()

# ---------------------------- Entity Extraction -------------------------------- #

GREEK_LEFT = [r"\bαριστερ\w*", r"\bαρ\.", r"\bleft\b", r"\bL\b"]
GREEK_RIGHT = [r"\bδεξι\w*", r"\bδε\.", r"\bright\b", r"\bR\b"]

def detect_side(q: str) -> Optional[str]:
    for pat in GREEK_LEFT:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "L"
    for pat in GREEK_RIGHT:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "R"
    # common abbreviations
    if re.search(r"\bL(eft)?\b", q, flags=re.IGNORECASE):
        return "L"
    if re.search(r"\bR(ight)?\b", q, flags=re.IGNORECASE):
        return "R"
    return None

def detect_condition(q: str) -> Optional[str]:
    if re.search(r"(?i)\bASD\b|ASD[:：]", q) or re.search(r"(?i)αυτισ", q):
        return "ASD"
    if re.search(r"(?i)\bTD\b|TD[:：]", q) or re.search(r"(?i)τυπικ", q) or re.search(r"(?i)\bcontrol\b", q):
        return "TD"
    return None

def detect_joint(q: str, features: Dict[str, Any]) -> Optional[str]:
    aliases = features.get("joint_aliases", {})
    for joint, words in aliases.items():
        for w in words:
            if re.search(rf"(?i)\b{re.escape(w)}\b", q):
                return joint
    return None

def detect_stat(q: str, features: Dict[str, Any]) -> str:
    metrics = features.get("metrics", {})
    # try explicit matches
    for k, v in metrics.items():
        if re.search(rf"(?i)\b{re.escape(k)}\b", q):
            return v
    # defaults
    if re.search(r"(?i)\b(std|stdev|standard deviation)\b", q):
        return "std"
    if re.search(r"(?i)\b(var|variance)\b", q):
        return "var"
    return "mean"  # default

def joint_side_to_code_regex(joint: str, side: str, features: Dict[str, Any]) -> Optional[str]:
    key = f"{joint}_{side}"
    return features.get("code_regex", {}).get(key)

# ----------------------------- Intent Matching --------------------------------- #

def match_intent(question: str,
                 intents: List[Dict[str, Any]]) -> Optional[str]:
    compiled = _compile_patterns(intents)
    for intent_name, pats in compiled:
        ok = True
        for p in pats:
            if p.search(question) is None:
                ok = False
                break
        if ok:
            return intent_name
    # lightweight fallbacks based on keywords
    q = question.lower()
    if "correl" in q or "συσχετ" in q:
        return "top_correlated_pairs"
    if "complete" in q or "πλήρη" in q or "όλα" in q:
        return "list_trials_files_complete"
    if "count" in q or "πόσα" in q or "κόμβοι" in q or "σχέσεις" in q:
        return "count_nodes_labels"
    return None

# ----------------------------- Template Filling -------------------------------- #

def fill_template(name: str, templates: Dict[str, str], slots: Dict[str, str]) -> str:
    if name not in templates:
        # final fallback: safe default
        return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;"
    cy = templates[name]
    for k, v in slots.items():
        cy = cy.replace("{" + k + "}", v)
    return cy.strip() + ("\n" if not cy.endswith("\n") else "")

# ------------------------------- Core Engine ----------------------------------- #

class NL2Cypher:
    def __init__(self,
                 intents_path: str = "intents.json",
                 features_path: str = "features.json",
                 templates_path: str = "templates.json") -> None:
        self.intents = _load_json_if_exists(intents_path, DEFAULT_INTENTS)
        self.features = _load_json_if_exists(features_path, DEFAULT_FEATURES)
        self.templates = _load_json_if_exists(templates_path, DEFAULT_TEMPLATES)

    def to_cypher(self, question: str) -> str:
        q = _text(question)
        if not q:
            return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;"

        intent = match_intent(q, self.intents)

        # --- intent: mean_of_feature_by_cond_side_joint ---
        if intent == "mean_of_feature_by_cond_side_joint":
            cond = detect_condition(q)
            side = detect_side(q)
            joint = detect_joint(q, self.features)
            stat = detect_stat(q, self.features)

            # Soft fallbacks
            if cond is None:
                # default to ASD if user mentions ASD/TD implicitly; else leave NULL → fallback compare
                cond = "ASD"
            if side is None:
                # if no side, prefer Right
                side = "R"
            if joint is None:
                # default to knee if not specified
                joint = "knee"

            code_regex = joint_side_to_code_regex(joint, side, self.features)
            if not code_regex:
                # generic fallback: pass-through with just stat (dangerously broad)
                code_regex = r"(?i).*"

            slots = {
                "cond": cond,
                "stat": stat,
                "code_regex": code_regex
            }
            return fill_template("mean_of_feature_by_cond_side_joint", self.templates, slots)

        # --- intent: compare_asd_td_feature ---
        if intent == "compare_asd_td_feature":
            side = detect_side(q) or "R"
            joint = detect_joint(q, self.features) or "knee"
            stat = detect_stat(q, self.features)

            code_regex = joint_side_to_code_regex(joint, side, self.features) or r"(?i).*"
            slots = {
                "stat": stat,
                "code_regex": code_regex
            }
            return fill_template("compare_asd_td_feature", self.templates, slots)

        # --- intent: top_correlated_pairs ---
        if intent == "top_correlated_pairs":
            return fill_template("top_correlated_pairs", self.templates, {})

        # --- intent: list_trials_files_complete ---
        if intent == "list_trials_files_complete":
            return fill_template("list_trials_files_complete", self.templates, {})

        # --- intent: count_nodes_labels ---
        if intent == "count_nodes_labels":
            return fill_template("count_nodes_labels", self.templates, {})

        # ------------- Fallbacks if no intent matched ---------------- #
        # Heuristics: if the user mentions ASD/TD + joint, assume compare
        if re.search(r"(?i)\bASD\b|ASD[:：]|αυτισ", q) or re.search(r"(?i)\bTD\b|TD[:：]|τυπικ|control", q):
            side = detect_side(q) or "R"
            joint = detect_joint(q, self.features) or "knee"
            stat = detect_stat(q, self.features)
            code_regex = joint_side_to_code_regex(joint, side, self.features) or r"(?i).*"
            slots = {"stat": stat, "code_regex": code_regex}
            return fill_template("compare_asd_td_feature", self.templates, slots)

        # If mentions mean/average without cond → default to ASD
        if re.search(r"(?i)\b(mean|average|avg|μ\.?ο\.?|μέση)\b", q):
            cond = detect_condition(q) or "ASD"
            side = detect_side(q) or "R"
            joint = detect_joint(q, self.features) or "knee"
            stat = detect_stat(q, self.features)
            code_regex = joint_side_to_code_regex(joint, side, self.features) or r"(?i).*"
            slots = {"cond": cond, "stat": stat, "code_regex": code_regex}
            return fill_template("mean_of_feature_by_cond_side_joint", self.templates, slots)

        # Last resort
        return "MATCH (n) RETURN count(n) AS nodes LIMIT 1;"

# --------------------------------- CLI ----------------------------------------- #

def main(argv: List[str]) -> None:
    # Usage:
    #   python nl2cypher.py "Ερώτηση φυσικής γλώσσας"
    # or read from stdin if no args
    if len(argv) > 1:
        question = " ".join(argv[1:])
    else:
        question = sys.stdin.read()

    engine = NL2Cypher()
    cypher = engine.to_cypher(question)
    # IMPORTANT: print only Cypher
    sys.stdout.write(cypher)

if __name__ == "__main__":
    main(sys.argv)