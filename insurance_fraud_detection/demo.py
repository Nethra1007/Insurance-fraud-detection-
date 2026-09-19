"""
demo.py
----------
Quick command-line demo: runs a single end-to-end inference so you can
show the backward-chaining engine working in 10 seconds during the review,
without opening the full dashboard.

Run:
    python demo.py                  # runs a built-in high-risk example
    python demo.py --claim-id CLM-1010   # runs a row from data/claims.csv
"""

import argparse
import json
import pandas as pd

from engine.knowledge_base import KnowledgeBase, build_working_memory
from engine.backward_chainer import BackwardChainer
from engine.risk_scorer import score_risk
from engine.fraud_trail import generate_fraud_trail

BUILT_IN_CLAIM = {
    "claim_id": "CLM-DEMO-1",
    "claim_amount": 280000,
    "claim_date": "2026-01-10",
    "claim_type": "Theft",
    "avg_claim_for_type": 90000,
    "policy_coverage": 500000,
    "policy_start_date": "2025-12-25",
    "policy_expiry_date": "2026-12-25",
    "previous_claims": 4,
    "claim_frequency": 3,
    "previous_fraud": True,
    "document_status": "incomplete",
    "damage_match": False,
    "report_consistency": False,
    "police_report": False,
    "accident_requires_police_report": True,
    "vehicle_age": 6,
    "age": 41,
}


def main():
    parser = argparse.ArgumentParser(description="Run a single backward-chaining inference.")
    parser.add_argument("--claim-id", help="Row to load from data/claims.csv instead of the built-in example.")
    parser.add_argument("--rules", default="data/rules.json", help="Path to the rule base JSON.")
    parser.add_argument("--json", action="store_true", help="Print the full Fraud Trail as JSON instead of text.")
    args = parser.parse_args()

    if args.claim_id:
        df = pd.read_csv("data/claims.csv")
        row = df[df["claim_id"] == args.claim_id]
        if row.empty:
            print(f"Claim '{args.claim_id}' not found in data/claims.csv")
            return
        claim = row.iloc[0].to_dict()
    else:
        claim = BUILT_IN_CLAIM

    kb = KnowledgeBase(args.rules)
    working_memory = build_working_memory(claim)
    chainer = BackwardChainer(kb, working_memory)
    fraud_node = chainer.infer("fraudulent_claim")
    risk = score_risk(fraud_node)
    trail = generate_fraud_trail(claim["claim_id"], fraud_node, risk)

    if args.json:
        print(json.dumps(trail, indent=2))
        return

    print("=" * 70)
    print("INSURANCE-CLAIM FRAUD DETECTION — BACKWARD CHAINING DEMO")
    print("=" * 70)
    print(f"Rules loaded: {len(kb.rules)}")
    print(f"Claim ID    : {claim['claim_id']}")
    print(f"Claim type  : {claim.get('claim_type')}")
    print(f"Claim amount: {claim.get('claim_amount')}")
    print("-" * 70)
    print(trail["narrative"])
    print("-" * 70)
    print(f"Risk level             : {risk.risk_level}")
    print(f"Overall confidence     : {risk.overall_confidence:.0%}")
    print(f"Evidence points used   : {risk.evidence_count}")
    print(f"Investigation priority : {risk.investigation_priority}")
    print("=" * 70)


if __name__ == "__main__":
    main()
