"""
engine/fraud_trail.py
------------------------
Module 6 — Fraud Trail Generator (Phase 3: Intelligence & Explanation)

Takes the raw backward-chaining TreeNode and produces:
    1. A JSON-serialisable tree (for the Plotly/NetworkX visual in
       utils/visualizations.py)
    2. A step-by-step textual narrative, e.g.:

        Hypothesis: Claim #CLM-1023 could be fraudulent.
        -> Rule R01 (Unusual Claim Amount) triggered: ...
        -> Evidence A: Claim & Policy Details verified.
        -> Rule R09 (Document Inconsistency) triggered: ...
        -> Conclusion: FLAGGED - HIGH PRIORITY.
"""

from typing import Dict, Any, List
from engine.backward_chainer import TreeNode, collect_fired_rules
from engine.risk_scorer import RiskReport

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_trail_tree(fraud_node: TreeNode) -> Dict[str, Any]:
    """JSON tree for visualization (node colour is derived client-side from
    `proven`: True -> green, False -> red)."""
    return fraud_node.to_dict()


def generate_narrative(claim_id: str, fraud_node: TreeNode, risk: RiskReport) -> str:
    lines: List[str] = []
    lines.append(f'Hypothesis: Claim #{claim_id} could be fraudulent.')

    fired = collect_fired_rules(fraud_node)
    seen = set()
    letter_i = 0

    if not fired:
        lines.append("-> No rule could be fully proven; insufficient evidence "
                      "of fraud was found in the available facts.")
    else:
        for rule in fired:
            if rule.rule_id in seen:
                continue
            seen.add(rule.rule_id)
            lines.append(f"-> Rule {rule.rule_id} ({rule.name}) triggered: "
                         f"{rule.backward_question}")
            letter = _LETTERS[letter_i % len(_LETTERS)]
            letter_i += 1
            lines.append(f"   Evidence {letter}: {rule.evidence_label} verified.")

    if risk.is_fraudulent:
        lines.append(f"-> Conclusion: FLAGGED — {risk.risk_level.upper()} PRIORITY "
                      f"(confidence {risk.overall_confidence:.0%}).")
    else:
        lines.append("-> Conclusion: NOT FLAGGED — insufficient evidence of fraud.")

    return "\n".join(lines)


def generate_fraud_trail(claim_id: str, fraud_node: TreeNode, risk: RiskReport) -> Dict[str, Any]:
    """Bundles everything the dashboard/API needs for one claim's trail."""
    return {
        "claim_id": claim_id,
        "is_fraudulent": risk.is_fraudulent,
        "risk_level": risk.risk_level,
        "overall_confidence": risk.overall_confidence,
        "investigation_priority": risk.investigation_priority,
        "triggered_rules": risk.triggered_rules,
        "evidence_count": risk.evidence_count,
        "category_hits": risk.category_hits,
        "tree": build_trail_tree(fraud_node),
        "narrative": generate_narrative(claim_id, fraud_node, risk),
    }
