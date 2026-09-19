"""
engine/risk_scorer.py
-----------------------
Module 5 — Fraud Risk & Evidence Module (Phase 3: Intelligence & Explanation)

Turns a backward-chaining proof (TreeNode for `fraudulent_claim`) into:
    - risk_level              : Low / Medium / High / Critical
    - overall_confidence      : weighted confidence across fired rules
    - triggered_rules         : list of rule dicts (id, name, confidence, category)
    - evidence_count          : number of distinct true leaf facts used
    - investigation_priority  : phrased recommendation for the investigator
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from engine.backward_chainer import TreeNode, collect_fired_rules, collect_true_leaves

COMPOSITE_CATEGORIES = {"composite", "goal"}


@dataclass
class RiskReport:
    is_fraudulent: bool
    risk_level: str
    overall_confidence: float
    investigation_priority: str
    triggered_rules: List[Dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0
    category_hits: List[str] = field(default_factory=list)


def score_risk(fraud_node: TreeNode) -> RiskReport:
    fired = collect_fired_rules(fraud_node)
    # De-duplicate while preserving order
    seen, unique_fired = set(), []
    for r in fired:
        if r.rule_id not in seen:
            seen.add(r.rule_id)
            unique_fired.append(r)

    evidence_count = len(set(collect_true_leaves(fraud_node)))

    # distinct non-goal composite categories that actually fired
    category_hits = sorted({
        r.category for r in unique_fired
        if r.category not in ("goal",)
    })

    if unique_fired:
        overall_confidence = sum(r.confidence for r in unique_fired) / len(unique_fired)
    else:
        overall_confidence = 0.0

    if not fraud_node.proven:
        risk_level = "Low"
        priority = "Low — no investigation required"
    elif overall_confidence >= 0.85 and len(category_hits) >= 2:
        risk_level = "Critical"
        priority = "Critical — escalate for immediate senior investigation"
    elif overall_confidence >= 0.7:
        risk_level = "High"
        priority = "High — immediate investigation recommended"
    elif overall_confidence >= 0.5:
        risk_level = "Medium"
        priority = "Medium — flagged for investigator review"
    else:
        risk_level = "Low"
        priority = "Low — monitor, no immediate action"

    triggered_rules = [
        {
            "rule_id": r.rule_id,
            "name": r.name,
            "category": r.category,
            "confidence": r.confidence,
            "backward_question": r.backward_question,
            "evidence_label": r.evidence_label,
        }
        for r in unique_fired
    ]

    return RiskReport(
        is_fraudulent=fraud_node.proven,
        risk_level=risk_level,
        overall_confidence=round(overall_confidence, 2),
        investigation_priority=priority,
        triggered_rules=triggered_rules,
        evidence_count=evidence_count,
        category_hits=category_hits,
    )
