#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json
from typing import Dict, Any, List
from transformers import pipeline

# Μπορείς να το αλλάξεις με env var αν θέλεις
DEFAULT_MODEL = "google/flan-t5-small"

# ---------------------- I/O helpers ----------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_greek(text: str) -> str:
    t = text.strip().lower()
    repl = {
        "ά":"α","έ":"ε","ή":"η","ί":"ι","ό":"ο","ύ":"υ","ώ":"ω",
        "ϊ":"ι","ΐ":"ι","ϋ":"υ","ΰ":"υ"
    }
    for a,b in repl.items():
        t = t.replace(a,b)
    return t

def apply_synonyms(text: str, syn: Dict[str,str]) -> str:
    t = normalize_greek(text)
    for k,v in syn.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t)
    return t

# ---------------------- prompt building ----------------------

def _truncate_fewshots(fewshots: List[Dict[str,str]], max_chars: int = 1200) -> List[Dict[str,str]]:
    """Κόβει τα fewshots ώστε το συνολικό prompt να μην γίνεται τεράστιο για flan-t5-small."""
    out = []
    total = 0
    for ex in fewshots:
        block = f"Q: {ex['q']}\nCypher:\n{ex['cypher']}\n---\n"
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
    lines.append("Feature codes examples: HIAN(L/R)=Knee angle, KNFO(L/R)=Ankle angle, THHTI(L/R)=Trunk tilt, SPKN(L/R)=Hip, etc.")
    lines.append("Use mean/std/variance as Feature.stat and use case-insensitive regex when matching codes, e.g. '(?i).*HIANR\\s*$'.")
    lines.append("")
    for ex in few:
        lines.append(f"Q: {ex['q']}\nCypher:\n{ex['cypher']}\n---")
    # Κόψε υπερβολικά μακρύ user_q
    uq = user_q.strip()
    if len(uq) > 600:
        uq = uq[:600]
    lines.append(f"Q: {uq}\nCypher:\n")
    return "\n".join(lines)

# ---------------------- sanitization & validation ----------------------

CY_START_RE = re.compile(r'(?is)\b(MATCH|WITH|CALL|UNWIND|CREATE|MERGE|RETURN)\b')
CY_VALID_START = re.compile(r'(?is)^\s*(MATCH|WITH|CALL|UNWIND|CREATE|MERGE|RETURN)\b')

def _sanitize_to_cypher(text: str) -> str:
    """Καθαρίζει την έξοδο του SLM ώστε να μείνει καθαρό Cypher."""
    if not text:
        return ""
    # βγάλε code fences
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^```(?:cypher)?", "", text.strip(), flags=re.I)
    text = re.sub(r"```$", "", text.strip())
    # πέτα prefixes: "Cypher:", "Query:", "1) ", "2. " κ.λπ.
    text = re.sub(r'(?i)^\s*(cypher|query)\s*:\s*', '', text).strip()
    text = re.sub(r'^\s*\d+[\)\.]?\s*', '', text)
    # κράτα από το πρώτο κλασικό keyword του Cypher και μετά
    m = CY_START_RE.search(text)
    if m:
        text = text[m.start():]
    return text.strip()

def looks_like_cypher(s: str) -> bool:
    return bool(s and CY_VALID_START.search(s.strip()))

# ---------------------- NL2Cypher core ----------------------

class NL2Cypher:
    def __init__(self, model_name: str = None):
        model_name = model_name or DEFAULT_MODEL
        gen_kwargs = dict(
            max_new_tokens=220,    # συντομότερο output
            do_sample=False,       # deterministic
            temperature=0.0
        )
        # Transformers θα διαλέξει αυτόματα συσκευή (βλ. log "Device set to use cuda:0")
        self.pipe = pipeline("text2text-generation", model=model_name, **gen_kwargs)

    def __call__(self, user_q: str, synonyms: Dict[str,str], fewshots: List[Dict[str,str]]) -> str:
        # 1) Normalize + synonyms
        nq = apply_synonyms(user_q or "", synonyms or {})

        # 2) Μικρές ευρετικές για “σίγουρα” μοτίβα (π.χ. mean knee ASD)
        #    Αν ταιριάξει, επέστρεψε έτοιμο από fewshots για σταθερότητα/ταχύτητα.
        if re.search(r"(?i)\bmean\b.*\bHIAN", nq) and re.search(r"(?i)\bASD\b", nq):
            for ex in fewshots or []:
                if ("HIAN" in ex.get("cypher","")) and ("ASD" in ex.get("cypher","")):
                    return ex["cypher"]

        # 3) Prompting στο SLM
        prompt = build_fewshot_prompt(user_q or "", fewshots or [])
        raw = self.pipe(prompt)[0]["generated_text"]

        # 4) Sanitize
        cy = _sanitize_to_cypher(raw)

        # 5) Αν δεν μοιάζει με Cypher, μην επιστρέψεις “σκουπίδι”.
        #    Το app θα το πιάσει και θα στο εμφανίσει.
        if not looks_like_cypher(cy):
            return ""

        return cy