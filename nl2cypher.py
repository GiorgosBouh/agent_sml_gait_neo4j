#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json
from typing import Dict, Any, List
from transformers import pipeline

DEFAULT_MODEL = "google/flan-t5-small"

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

def build_fewshot_prompt(user_q: str, fewshots: List[Dict[str,str]]) -> str:
    lines = []
    lines.append("Task: Convert the user question to a Cypher query for a Neo4j gait knowledge graph.")
    lines.append("Only output Cypher. Use existing feature codes like HIAN, KNFO, THHTI, SPKN, etc.")
    lines.append("Nodes: Subject(pid), Condition(name), Sample, Feature(code, stat), FeatureValue(value).")
    lines.append("Use mean/std/variance and regex for codes (e.g., '(?i).*HIANR\\s*$').")
    lines.append("")
    for ex in fewshots:
        lines.append(f"Q: {ex['q']}\nCypher:\n{ex['cypher']}\n---")
    lines.append(f"Q: {user_q}\nCypher:\n")
    return "\n".join(lines)

class NL2Cypher:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        # μικρό & δωρεάν SLM από HF
        self.pipe = pipeline("text2text-generation", model=model_name, max_new_tokens=256)

    def __call__(self, user_q: str, synonyms: Dict[str,str], fewshots: List[Dict[str,str]]) -> str:
        # 1) heuristic normalization
        nq = apply_synonyms(user_q, synonyms)

        # 2) απλές ευρετικές (π.χ. mean knee ASD)
        if re.search(r"\bmean\b.*\bHIAN\b", nq) and "ASD" in nq:
            # demo: επιστρέφουμε έτοιμο Cypher από τα fewshots αν υπάρχει σχετικό
            for ex in fewshots:
                if "μέση γωνία του γόνατος" in ex["q"] or "mean" in ex["q"].lower():
                    return ex["cypher"]

        # 3) few-shot prompting στο SLM
        prompt = build_fewshot_prompt(user_q, fewshots)
        out = self.pipe(prompt)[0]["generated_text"]
        m = re.search(r"(?s)Cypher:\s*(.*)", out)
        cy = m.group(1).strip() if m else out.strip()
        # καθάρισε πιθανά fencing/backticks
        cy = re.sub(r"^```(cypher)?", "", cy).strip()
        cy = re.sub(r"```$", "", cy).strip()
        return cy