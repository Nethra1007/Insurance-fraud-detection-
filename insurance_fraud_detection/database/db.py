"""
database/db.py
-----------------
SQLite persistence via SQLAlchemy: stores claims, a cached copy of the rule
base, and a log of every inference run (so the Fraud Trail Explorer page
can revisit past results without re-running the engine).
"""

import json
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

DB_PATH = ":memory:"
engine = create_engine(
    "sqlite:///:memory:",
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ClaimRecord(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String, unique=True, index=True, nullable=False)
    customer_name = Column(String)
    claim_type = Column(String)
    claim_amount = Column(Float)
    claim_date = Column(String)
    policy_type = Column(String)
    location = Column(String)
    raw_json = Column(Text)          # full raw claim dict, for re-running inference


class RuleRecord(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    category = Column(String)
    confidence = Column(Float)
    raw_json = Column(Text)


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String, index=True, nullable=False)
    is_fraudulent = Column(Boolean)
    risk_level = Column(String)
    overall_confidence = Column(Float)
    investigation_priority = Column(String)
    triggered_rule_ids = Column(String)      # comma-separated
    result_json = Column(Text)               # full fraud-trail dict (JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()


# ---------------------------------------------------------------------
# Convenience helpers used by the Streamlit app / API
# ---------------------------------------------------------------------
def upsert_claim(session, claim: dict):
    existing = session.query(ClaimRecord).filter_by(claim_id=claim["claim_id"]).first()
    if existing:
        existing.raw_json = json.dumps(claim, default=str)
        session.commit()
        return existing

    record = ClaimRecord(
        claim_id=claim["claim_id"],
        customer_name=claim.get("customer_name", ""),
        claim_type=claim.get("claim_type", ""),
        claim_amount=float(claim.get("claim_amount", 0)),
        claim_date=str(claim.get("claim_date", "")),
        policy_type=claim.get("policy_type", ""),
        location=claim.get("location", ""),
        raw_json=json.dumps(claim, default=str),
    )
    session.add(record)
    session.commit()
    return record


def save_rules_snapshot(session, rules):
    for r in rules:
        existing = session.query(RuleRecord).filter_by(rule_id=r.rule_id).first()
        payload = {
            "rule_id": r.rule_id, "name": r.name, "description": r.description,
            "category": r.category, "confidence": r.confidence,
            "backward_question": r.backward_question,
        }
        if existing:
            existing.raw_json = json.dumps(payload)
            existing.confidence = r.confidence
        else:
            session.add(RuleRecord(
                rule_id=r.rule_id, name=r.name, category=r.category,
                confidence=r.confidence, raw_json=json.dumps(payload),
            ))
    session.commit()


def log_inference(session, trail: dict):
    log = InferenceLog(
        claim_id=trail["claim_id"],
        is_fraudulent=trail["is_fraudulent"],
        risk_level=trail["risk_level"],
        overall_confidence=trail["overall_confidence"],
        investigation_priority=trail["investigation_priority"],
        triggered_rule_ids=",".join(r["rule_id"] for r in trail["triggered_rules"]),
        result_json=json.dumps(trail),
    )
    session.add(log)
    session.commit()
    return log


def get_recent_logs(session, limit: int = 50):
    return (session.query(InferenceLog)
            .order_by(InferenceLog.created_at.desc())
            .limit(limit)
            .all())