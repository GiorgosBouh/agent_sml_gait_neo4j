#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent API for NL → Cypher → Neo4j queries
Aligned with graph schema:
  (:Subject {pid,sid,pid_num?})-[:HAS_TRIAL]->(:Trial {uid})
  (:Trial)-[:HAS_FILE]->(:File {uid,kind})
  (:Trial)-[:HAS_FEATURE]->(:FeatureValue {value})-[:OF_FEATURE]->(:Feature {code,stat})

Endpoints:
  - GET  /health
  - POST /nl2cypher           { "question": "..." } -> { "cypher": "..." }
  - POST /query               { "cypher": "..." }   -> { "records": [...], "keys": [...] }
  - POST /ask                 { "question": "..." } -> { "cypher": "...", "records": [...], "keys": [...] }
  - POST /reports/trials-complete
  - POST /reports/top-correlated

Env:
  - NEO4J_URI       (default "bolt://localhost:7687")
  - NEO4J_USER      (default "neo4j")
  - NEO4J_PASSWORD  (default "palatiou")
  - ALLOW_WRITE     (default "false")
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Neo4j driver
from neo4j import GraphDatabase, basic_auth, Driver

# NL → Cypher engine (use the corrected file you pasted earlier)
try:
    from nl2cypher import NL2Cypher
except Exception as e:
    raise RuntimeError("Could not import nl2cypher. Ensure nl2cypher.py is in PYTHONPATH.") from e


# ----------------------------- Configuration ---------------------------------- #

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "palatiou")
ALLOW_WRITE = os.getenv("ALLOW_WRITE", "false").lower() in ("1", "true", "yes")

READ_ONLY_DENY = (
    r"(?i)\b(create|merge|delete|detach\s+delete|set\s+|remove\s+|"
    r"call\s+db\.(ms|msl|index|schema)|apoc\.(periodic|refactor|"
    r"trigger|schema|doIt|do.when|create)|load\s+csv)\b"
)

SAFE_WHITELIST = [
    # allow CALL { ... } subqueries, RETURN, MATCH, OPTIONAL MATCH, WITH, ORDER BY, LIMIT, SKIP
    r"(?i)\bcall\b", r"(?i)\breturn\b", r"(?i)\bmatch\b", r"(?i)\boptional\s+match\b",
    r"(?i)\bwith\b", r"(?i)\border\s+by\b", r"(?i)\blimit\b", r"(?i)\bskip\b"
]


# ----------------------------- Neo4j Client ----------------------------------- #

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, allow_write: bool = False) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        self.allow_write = allow_write

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    def _is_write_query(self, cypher: str) -> bool:
        if self.allow_write:
            return False
        # If any destructive/write keyword is present, block
        return re.search(READ_ONLY_DENY, cypher) is not None

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not isinstance(cypher, str) or not cypher.strip():
            raise ValueError("Empty Cypher")

        if self._is_write_query(cypher):
            raise PermissionError("Write operations are disabled in this API.")

        def run_tx(tx):
            result = tx.run(cypher, **(params or {}))
            keys = result.keys()
            records = [dict(rec) for rec in result]
            return records, list(keys)

        with self._driver.session() as session:
            try:
                return session.read_transaction(run_tx)  # read-only
            except Exception:
                # fallback to explicit run if tx fails (still safe)
                result = session.run(cypher, **(params or {}))
                keys = result.keys()
                records = [dict(rec) for rec in result]
                return records, list(keys)


# ----------------------------- FastAPI App ------------------------------------ #

app = FastAPI(title="ASD Gait Agent API", version="1.0.0")

# CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singletons
neo = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, allow_write=ALLOW_WRITE)
engine = NL2Cypher()


# ----------------------------- Schemas ---------------------------------------- #

class NLQuestion(BaseModel):
    question: str

class CypherQuery(BaseModel):
    cypher: str
    params: Optional[Dict[str, Any]] = None


# ----------------------------- Helpers ---------------------------------------- #

def _strip_bom(text: str) -> str:
    return text.lstrip("\ufeff").strip()

def _assert_read_only(cypher: str) -> None:
    if neo._is_write_query(cypher):
        raise HTTPException(status_code=403, detail="Write operations are disabled.")


# ----------------------------- Endpoints -------------------------------------- #

@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        recs, keys = neo.query("RETURN 1 AS ok")
        return {"status": "ok", "neo4j": bool(recs and recs[0].get("ok") == 1)}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/nl2cypher")
def nl2cypher_api(payload: NLQuestion) -> Dict[str, Any]:
    q = _strip_bom(payload.question)
    if not q:
        raise HTTPException(status_code=400, detail="Empty question.")
    cy = engine.to_cypher(q)
    if not isinstance(cy, str) or not cy.strip():
        raise HTTPException(status_code=500, detail="Failed to generate Cypher.")
    # still enforce read-only safety
    _assert_read_only(cy)
    return {"cypher": cy}

@app.post("/query")
def query_api(payload: CypherQuery) -> Dict[str, Any]:
    cy = _strip_bom(payload.cypher)
    if not cy:
        raise HTTPException(status_code=400, detail="Empty cypher.")
    _assert_read_only(cy)
    try:
        records, keys = neo.query(cy, payload.params or {})
        return {"records": records, "keys": keys}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_api(payload: NLQuestion) -> Dict[str, Any]:
    q = _strip_bom(payload.question)
    if not q:
        raise HTTPException(status_code=400, detail="Empty question.")
    cy = engine.to_cypher(q)
    if not isinstance(cy, str) or not cy.strip():
        raise HTTPException(status_code=500, detail="Failed to generate Cypher.")
    _assert_read_only(cy)
    try:
        records, keys = neo.query(cy, {})
        return {"cypher": cy, "records": records, "keys": keys}
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------- Convenience Reports (read-only) ----------------------- #

@app.post("/reports/trials-complete")
def trials_complete_report() -> Dict[str, Any]:
    cy = """
    MATCH (t:Trial)
    OPTIONAL MATCH (t)-[:HAS_FILE]->(f:File)
    WITH t, collect(DISTINCT f.kind) AS kinds
    OPTIONAL MATCH (t)-[:HAS_FEATURE]->(fv:FeatureValue)-[:OF_FEATURE]->(feat:Feature)
    WITH t, kinds, count(DISTINCT fv) AS nvals, count(DISTINCT feat) AS nfeats
    RETURN count(*) AS trials,
           sum(CASE WHEN size(kinds)=4 THEN 1 ELSE 0 END) AS trials_all_files,
           sum(CASE WHEN nvals=463 AND nfeats=463 THEN 1 ELSE 0 END) AS trials_complete;
    """
    records, keys = neo.query(cy)
    return {"cypher": cy.strip(), "records": records, "keys": keys}

@app.post("/reports/top-correlated")
def top_correlated_report(limit: int = 20) -> Dict[str, Any]:
    limit = max(1, min(200, limit))
    cy = f"""
    MATCH (a:Feature)-[r:CORRELATED_WITH]->(b:Feature)
    RETURN a.code AS A, b.code AS B, r.r AS r, r.n AS n
    ORDER BY abs(r) DESC, n DESC
    LIMIT {limit};
    """
    records, keys = neo.query(cy)
    return {"cypher": cy.strip(), "records": records, "keys": keys}


# ----------------------------- Shutdown Hook ---------------------------------- #

@app.on_event("shutdown")
def on_shutdown():
    neo.close()


# ----------------------------- Dev Server ------------------------------------- #
# Run with: uvicorn agent_app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)