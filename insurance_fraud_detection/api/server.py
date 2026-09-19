"""
api/server.py
----------------
FastAPI REST layer over the same engine used by the Streamlit dashboard.

Run standalone:
    uvicorn api.server:app --reload --port 8000
Docs at http://localhost:8000/docs
"""

import json
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine.knowledge_base import KnowledgeBase, build_working_memory
from engine.backward_chainer import BackwardChainer
from engine.risk_scorer import score_risk
from engine.fraud_trail import generate_fraud_trail
from database.db import (
    init_db, get_session, upsert_claim, save_rules_snapshot,
    log_inference, ClaimRecord,
)

app = FastAPI(
    title="Insurance-Claim Fraud Detection API",
    description="Backward Chaining Rule Inference engine, exposed as a REST API.",
    version="1.0.0",
)

KB = KnowledgeBase("data/rules.json")


@app.on_event("startup")
def _startup():
    init_db()
    session = get_session()
    save_rules_snapshot(session, KB.rules)
    session.close()


class ClaimIn(BaseModel):
    claim_id: str
    claim_amount: float
    claim_date: str
    claim_type: str
    avg_claim_for_type: Optional[float] = 60000
    policy_coverage: Optional[float] = 500000
    policy_start_date: str
    policy_expiry_date: Optional[str] = None
    customer_name: Optional[str] = None
    age: Optional[int] = 35
    previous_claims: Optional[int] = 0
    claim_frequency: Optional[float] = 0
    previous_fraud: Optional[bool] = False
    claims_in_last_30_days: Optional[int] = 0
    document_status: Optional[str] = "complete"
    damage_match: Optional[bool] = True
    report_consistency: Optional[bool] = True
    police_report: Optional[bool] = True
    accident_requires_police_report: Optional[bool] = True
    vehicle_age: Optional[int] = 5
    high_risk_location: Optional[bool] = False
    premium_payment_irregular: Optional[bool] = False
    no_witnesses: Optional[bool] = False
    multiple_policies_same_vehicle: Optional[bool] = False
    report_date_mismatch: Optional[bool] = False


@app.get("/health")
def health():
    return {"status": "ok", "rules_loaded": len(KB.rules)}


@app.get("/rules")
def list_rules():
    return [
        {
            "rule_id": r.rule_id, "name": r.name, "description": r.description,
            "category": r.category, "confidence": r.confidence,
            "backward_question": r.backward_question,
            "evidence_label": r.evidence_label,
        }
        for r in KB.rules
    ]


@app.post("/claims")
def submit_claim(claim: ClaimIn):
    session = get_session()
    record = upsert_claim(session, claim.dict())
    session.close()
    return {"status": "saved", "claim_id": record.claim_id}


@app.post("/infer")
def infer(claim: ClaimIn):
    claim_dict = claim.dict()
    if not claim_dict.get("policy_expiry_date"):
        claim_dict["policy_expiry_date"] = claim_dict["policy_start_date"]

    working_memory = build_working_memory(claim_dict)
    chainer = BackwardChainer(KB, working_memory)
    fraud_node = chainer.infer("fraudulent_claim")
    risk = score_risk(fraud_node)
    trail = generate_fraud_trail(claim.claim_id, fraud_node, risk)

    session = get_session()
    upsert_claim(session, claim_dict)
    log_inference(session, trail)
    session.close()

    return trail


@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    session = get_session()
    record = session.query(ClaimRecord).filter_by(claim_id=claim_id).first()
    session.close()
    if not record:
        raise HTTPException(status_code=404, detail="Claim not found")
    return json.loads(record.raw_json)
