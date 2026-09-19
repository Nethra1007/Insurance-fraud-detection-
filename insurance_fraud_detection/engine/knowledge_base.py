"""
engine/knowledge_base.py
--------------------------
Module 2 (Data Preprocessing) + Module 3 (Knowledge Base) glue.

- KnowledgeBase loads the rule base (via rule_parser.load_rules) once.
- build_working_memory() turns one raw claim record (a dict/row as produced
  by utils.data_generator or loaded from data/claims.csv) into the
  "working memory" of raw facts the backward chainer's rules test against.

Raw facts here are deliberately simple (numbers/booleans/strings) — the
*rules* are what combine them into meaningful red flags, and rules can also
depend on OTHER rules' conclusions (e.g. R22's antecedent is
`unusual_claim_amount`, which is itself R01's conclusion). That recursive
dependency is exactly what backward_chainer.py resolves.
"""

from datetime import datetime
from typing import Dict, Any, List
from engine.rule_parser import load_rules, Rule

DATE_FMT = "%Y-%m-%d"


def _parse_date(value):
    if isinstance(value, datetime):
        return value
    return datetime.strptime(value, DATE_FMT)


class KnowledgeBase:
    """Loads and caches the rule base for the whole application."""

    def __init__(self, rules_path: str = "data/rules.json"):
        self.rules_path = rules_path
        self.rules: List[Rule] = load_rules(rules_path)
        self.rules_by_conclusion: Dict[str, List[Rule]] = {}
        for r in self.rules:
            self.rules_by_conclusion.setdefault(r.conclusion, []).append(r)

    def rules_for(self, goal: str) -> List[Rule]:
        return self.rules_by_conclusion.get(goal, [])

    def all_categories(self) -> List[str]:
        return sorted({r.category for r in self.rules})


def build_working_memory(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Module 2: Data Preprocessing.

    Converts one raw claim record into the numeric/boolean facts referenced
    by data/rules.json's `condition.fact` fields.
    """
    claim_date = _parse_date(claim["claim_date"])
    policy_start = _parse_date(claim["policy_start_date"])
    policy_expiry = _parse_date(claim.get(
        "policy_expiry_date", claim["policy_start_date"]))

    claim_amount = float(claim["claim_amount"])
    avg_claim = float(claim.get("avg_claim_for_type", 60000))
    policy_coverage = float(claim.get("policy_coverage", 500000))

    wm: Dict[str, Any] = {
        # raw / derived numeric facts
        "claim_amount": claim_amount,
        "claim_amount_ratio": claim_amount / avg_claim if avg_claim else 0,
        "claim_to_coverage_ratio": claim_amount / policy_coverage if policy_coverage else 0,
        "is_round_number_claim": claim_amount % 10000 == 0 and claim_amount > 0,

        "previous_claims_count": int(claim.get("previous_claims", 0)),
        "claim_frequency_per_year": float(claim.get("claim_frequency", 0)),
        "previous_fraud": bool(claim.get("previous_fraud", False)),
        "claims_in_last_30_days": int(claim.get("claims_in_last_30_days", 0)),

        "days_since_policy_start": (claim_date - policy_start).days,
        "days_to_policy_expiry": (policy_expiry - claim_date).days,

        "document_incomplete": claim.get("document_status", "complete") != "complete",
        "report_date_mismatch": bool(claim.get("report_date_mismatch", False)),
        "missing_police_report": (
            bool(claim.get("accident_requires_police_report", True))
            and not bool(claim.get("police_report", True))
        ),
        "damage_match": bool(claim.get("damage_match", True)),
        "vehicle_age": int(claim.get("vehicle_age", 5)),
        "high_risk_location": bool(claim.get("high_risk_location", False)),
        "premium_payment_irregular": bool(claim.get("premium_payment_irregular", False)),
        "no_witnesses": bool(claim.get("no_witnesses", False)),
        "report_consistency": bool(claim.get("report_consistency", True)),
        "multiple_policies_same_vehicle": bool(claim.get("multiple_policies_same_vehicle", False)),
        "claim_type": claim.get("claim_type", "Collision"),
        "age": int(claim.get("age", 35)),
    }
    return wm
