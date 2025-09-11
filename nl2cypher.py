#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
from typing import Dict, Any, List, Optional

# Προαιρετικά μπορείς να απενεργοποιήσεις το SLM με env: NL2CYPHER_DISABLE_LLM=1
DISABLE_LLM = os.getenv("NL2CYPHER_DISABLE_LLM", "0") == "1"

# Default μικρό SLM. Μπορείς να αλλάξεις με env: NL2CYPHER_MODEL=...
DEFAULT_MODEL = os.getenv("NL2CYPHER_MODEL", "google/flan-t5-small")

# ---------------------- I/O helpers ----------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_greek(text: str) -> str:
    t = (text or "").strip().lower()
    repl = {
        "ά":"α","έ":"ε","ή":"η","ί":"ι","ό":"ο","ύ":"υ","ώ":"ω",
        "ϊ":"ι","ΐ":"ι","ϋ":"υ","ΰ":"υ"
    }
    for a,b in repl.items():
        t = t.replace(a,b)
    return t

def apply_synonyms(text: str, syn: Dict[str,str]) -> str:
    t = normalize_greek(text)
    for k,v in (syn or {}).items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t)
    return t

# ---------------------- deterministic routing ----------------------
def detect_intent(text: str, intents: List[Dict[str, Any]]) -> Optional[str]:
    t = text.lower()
    for item in intents or []:
        intent = item.get("intent")
        for pat in item.get("patterns", []):
            try:
                if re.search(pat, t):
                    return intent
            except re.error:
                # Αγνόησε τυχόν κακό regex
                continue
    return None

def extract_entities(text: str, feat: Dict[str, Any]) -> Dict[str, Optional[str]]:
    t = text.lower()

    # condition
    cond = None
    if re.search(r"\b(asd)\b", t): cond = "ASD"
    elif re.search(r"\b(td|typical|control|τυπικ)\b", t): cond = "TD"

    # side
    side = None
    if re.search(r"\b(left|αρισ|αρ\.)\b", t): side = "L"
    elif re.search(r"\b(right|δεξ|δεξ\.)\b", t): side = "R"

    # joint
    joint = None
    joint_aliases = (feat or {}).get("joint_aliases", {})
    for k, vals in joint_aliases.items():
        if any(v in t for v in vals):
            joint = k
            break

    # metric/stat
    stat = None
    metrics = (feat or {}).get("metrics", {})
    for k, v in metrics.items():
        if re.search(rf"\b{k}\b", t):
            stat = v
            break

    # code regex από joint/side
    code_regex = None
    if joint and side:
        code_key = f"{joint}_{side}"
        code_regex = (feat or {}).get("code_regex", {}).get(code_key)

    return {"cond": cond, "side": side, "joint": joint, "stat": stat, "code_regex": code_regex}

def fill_template(intent: str, ent: Dict[str, Optional[str]], templates: Dict[str, str]) -> str:
    if intent not in templates:
        return ""
    # defaults ασφαλείας
    d = {
        "cond": ent.get("cond") or "ASD",
        "side": ent.get("side") or "R",
        "joint": (ent.get("joint") or "knee").title(),
        "stat": ent.get("stat") or "mean",
        "code_regex": ent.get("code_regex") or "(?i).*HIANR\\s*$"
    }
    try:
        return templates[intent].format(**d)
    except KeyError:
        return ""

# ---------------------- prompt building (LLM fallback) ----------------------
def _truncate_fewshots(fewshots: List[Dict[str,str]], max_chars: int = 1200) -> List[Dict[str,str]]:
    out, total = [], 0
    for ex in fewshots or []:
        block = f"Q: {ex.get('q','')}\nCypher:\n{ex.get('cypher','')}\n---\n"
        if total + len(block) > max_chars:
            break
        out.append(ex)
        total += len(block)
    return out

def build_fewshot_prompt(user_q: str, fewshots: List[Dict[str,str]]) -> str:
    few = _truncate_fewshots(fewshots, max_chars=1200)
    lines = []
    lines.append("Task: Convert the user question to a valid Cypher query for a Neo4j gait knowledge graph.")
    lines.append("Only output Cypher (no prose, no numbering, no markdown).")
    lines.append("Graph schema summary:")
    lines.append("- Subject(pid) -[:HAS_CONDITION]-> Condition(name in ['ASD','TD'])")
    lines.append("- Subject -[:HAS_SAMPLE]-> Sample(sample_id, row, class, trial_idx)")
    lines.append("- Sample -[:HAS_VALUE]-> FeatureValue(value) -[:OF_FEATURE]-> Feature(code, stat)")
    lines.append("Feature codes examples: HIAN(L/R)=Knee angle, KNFO(L/R)=Ankle angle, THHTI(L/R)=Trunk tilt, SPKN(L/R)=Hip.")
    lines.append("Use mean/std/variance as Feature.stat and case-insensitive regex for codes, e.g. '(?i).*HIANR\\s*$'.")
    lines.append("")
    for ex in few:
        lines.append(f"Q: {ex.get('q','')}\nCypher:\n{ex.get('cypher','')}\n---")
    uq = (user_q or "").strip()
    if len(uq) > 600:
        uq = uq[:600]
    lines.append(f"Q: {uq}\nCypher:\n")
    return "\n".join(lines)

# ---------------------- sanitization & validation ----------------------
CY_START_RE = re.compile(r'(?is)\b(MATCH|WITH|CALL|UNWIND|CREATE|MERGE|RETURN)\b')
CY_VALID_START = re.compile(r'(?is)^\s*(MATCH|WITH|CALL|UNWIND|CREATE|MERGE|RETURN)\b')

def _sanitize_to_cypher(text: str) -> str:
    if not text:
        return ""
    # strip code fences & labels
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^```(?:cypher)?", "", text.strip(), flags=re.I)
    text = re.sub(r"```$", "", text.strip())
    text = re.sub(r'(?i)^\s*(cypher|query)\s*:\s*', '', text).strip()
    text = re.sub(r'^\s*\d+[\)\.]?\s*', '', text)
    # keep from first cypher keyword
    m = CY_START_RE.search(text)
    if m:
        text = text[m.start():]
    return text.strip()

def looks_like_cypher(s: str) -> bool:
    return bool(s and CY_VALID_START.search(s.strip()))

# ---------------------- NL2Cypher core ----------------------
class NL2Cypher:
    def __init__(self, model_name: Optional[str] = None):
        self.intents = load_json("intents.json")
        self.feat = load_json("features.json")
        self.templates = load_json("templates.json")

        self.llm = None
        if not DISABLE_LLM:
            from transformers import pipeline  # lazy import
            model_name = model_name or DEFAULT_MODEL
            self.llm = pipeline(
                "text2text-generation",
                model=model_name,
                max_new_tokens=220,
                do_sample=False,
                temperature=0.0
            )

    def _deterministic_route(self, user_q: str) -> str:
        """Πρώτα προσπαθεί με intents + templates. Αν όλα καλά, γυρίζει καθαρό Cypher."""
        intent = detect_intent(user_q or "", self.intents)
        if not intent:
            return ""
        ent = extract_entities(user_q or "", self.feat)
        cy = fill_template(intent, ent, self.templates)
        return cy or ""

    def _llm_fallback(self, user_q: str, fewshots: List[Dict[str,str]]) -> str:
        """Χρήση μικρού SLM μόνο ως έσχατη λύση, με αυστηρό sanitize/validate."""
        if not self.llm:
            return ""
        prompt = build_fewshot_prompt(user_q or "", fewshots or [])
        raw = self.llm(prompt)[0]["generated_text"]
        cy = _sanitize_to_cypher(raw)
        if not looks_like_cypher(cy):
            return ""
        return cy

    def __call__(self, user_q: str, synonyms: Dict[str,str], fewshots: List[Dict[str,str]]) -> str:
        # 1) Normalize + synonyms
        nq = apply_synonyms(user_q or "", synonyms or {})

        # 2) Deterministic routing (intents/templates)
        cy = self._deterministic_route(nq)
        if looks_like_cypher(cy):
            return cy

        # 3) Heuristic very-shortcuts (π.χ. mean knee ASD με έτοιμο fewshot)
        if re.search(r"(?i)\bmean\b.*\bHIAN", nq) and re.search(r"(?i)\bASD\b", nq):
            for ex in fewshots or []:
                c = ex.get("cypher","")
                if "HIAN" in c and "ASD" in c:
                    return c

        # 4) LLM fallback (strict sanitize)
        cy = self._llm_fallback(nq, fewshots)
        if looks_like_cypher(cy):
            return cy

        # 5) Τελικό: δώσε κενό ώστε το app να κάνει rule-based fallback στη μεριά του UI
        return ""