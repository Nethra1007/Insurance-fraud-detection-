"""
utils/data_generator.py
--------------------------
Module 1 — Data Collection (Phase 1)

Generates a synthetic dataset of insurance claims and saves it to
data/claims.csv. ~20% of records are deliberately constructed to trip
enough rules to be genuinely fraudulent under the rule base, so the
Analytics Dashboard and Fraud Trail Explorer have real signal to show.

The schema matches the "Dataset / API Identification" slide:
    Claim, Customer, Policy, Accident, Vehicle, History, Evidence

To swap in a real dataset (e.g. the Kaggle insurance-claim-fraud CSV),
replace this generator's output with a loader that maps the real columns
onto this same schema, then everything downstream (knowledge_base.py,
backward_chainer.py, the dashboard) works unchanged.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from faker import Faker
    fake = Faker()
except ImportError:  # pragma: no cover - Faker is in requirements.txt
    fake = None

RNG = np.random.default_rng(42)
random.seed(42)

POLICY_TYPES = ["Comprehensive Motor", "Third-Party Motor", "Zero Depreciation"]
CLAIM_TYPES = ["Collision", "Theft", "Total Loss", "Windscreen Damage", "Fire", "Natural Calamity"]
VEHICLE_TYPES = ["Sedan", "Hatchback", "SUV", "Truck", "Two-Wheeler"]
LOCATIONS = ["Chennai", "Trichy", "Coimbatore", "Madurai", "Salem", "Bengaluru", "Hyderabad"]
HIGH_RISK_LOCATIONS = {"Trichy", "Salem"}  # illustrative only, not a real claim

AVG_CLAIM_FOR_TYPE = {
    "Collision": 55000, "Theft": 90000, "Total Loss": 250000,
    "Windscreen Damage": 12000, "Fire": 150000, "Natural Calamity": 120000,
}


def _random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(days=random.randint(0, max(delta.days, 0)))


def _customer_name(i: int) -> str:
    return fake.name() if fake else f"Customer {i}"


def generate_claims(n: int = 500, fraud_ratio: float = 0.2) -> pd.DataFrame:
    n_fraud = int(n * fraud_ratio)
    rows = []
    policy_epoch = datetime(2022, 1, 1)
    today = datetime(2026, 9, 1)

    for i in range(n):
        is_fraud_design = i < n_fraud   # first n_fraud rows built to be fraud-like
        claim_type = random.choice(CLAIM_TYPES)
        vehicle_type = random.choice(VEHICLE_TYPES)
        policy_type = random.choice(POLICY_TYPES)
        location = random.choice(LOCATIONS)
        avg_claim = AVG_CLAIM_FOR_TYPE[claim_type]

        policy_start = _random_date(policy_epoch, today - timedelta(days=60))
        policy_expiry = policy_start + timedelta(days=365)

        if is_fraud_design:
            claim_amount = avg_claim * RNG.uniform(2.2, 4.5)
            claim_date = policy_start + timedelta(days=int(RNG.uniform(1, 25)))
            previous_claims = int(RNG.integers(2, 6))
            claim_frequency = round(RNG.uniform(1.5, 4.0), 1)
            previous_fraud = bool(RNG.random() < 0.35)
            document_status = random.choice(["incomplete", "incomplete", "complete"])
            damage_match = bool(RNG.random() < 0.3)
            report_consistency = bool(RNG.random() < 0.3)
            police_report = bool(RNG.random() < 0.35)
            claims_last_30 = int(RNG.integers(0, 3))
            premium_irregular = bool(RNG.random() < 0.4)
            no_witnesses = bool(RNG.random() < 0.5)
            vehicle_age = int(RNG.integers(0, 15))
        else:
            claim_amount = avg_claim * RNG.uniform(0.4, 1.6)
            claim_date = _random_date(policy_start + timedelta(days=30), today)
            previous_claims = int(RNG.integers(0, 2))
            claim_frequency = round(RNG.uniform(0, 1.2), 1)
            previous_fraud = False
            document_status = "complete"
            damage_match = True
            report_consistency = True
            police_report = True
            claims_last_30 = 0
            premium_irregular = bool(RNG.random() < 0.05)
            no_witnesses = bool(RNG.random() < 0.15)
            vehicle_age = int(RNG.integers(0, 15))

        rows.append({
            "claim_id": f"CLM-{1000 + i}",
            "claim_amount": round(float(claim_amount), 2),
            "claim_date": claim_date.strftime("%Y-%m-%d"),
            "claim_type": claim_type,
            "avg_claim_for_type": avg_claim,

            "customer_id": f"CUST-{2000 + i}",
            "customer_name": _customer_name(i),
            "age": int(RNG.integers(19, 70)),
            "location": location,
            "high_risk_location": location in HIGH_RISK_LOCATIONS,

            "policy_id": f"POL-{3000 + i}",
            "policy_type": policy_type,
            "policy_coverage": avg_claim * random.choice([3, 5, 8]),
            "policy_start_date": policy_start.strftime("%Y-%m-%d"),
            "policy_expiry_date": policy_expiry.strftime("%Y-%m-%d"),
            "premium_payment_irregular": premium_irregular,

            "accident_type": claim_type,
            "accident_requires_police_report": claim_type in
                ("Collision", "Theft", "Total Loss", "Fire"),
            "police_report": police_report,
            "report_date_mismatch": bool(RNG.random() < 0.25) if is_fraud_design else False,
            "no_witnesses": no_witnesses,

            "vehicle_id": f"VEH-{4000 + i}",
            "vehicle_type": vehicle_type,
            "vehicle_age": vehicle_age,
            "damage_match": damage_match,

            "previous_claims": previous_claims,
            "previous_fraud": previous_fraud,
            "claim_frequency": claim_frequency,
            "claims_in_last_30_days": claims_last_30,

            "document_status": document_status,
            "report_consistency": report_consistency,
            "multiple_policies_same_vehicle": bool(RNG.random() < 0.05) if is_fraud_design else False,

            "label_is_fraud_synthetic": is_fraud_design,   # ground-truth design label (for evaluation only)
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def save_claims(path: str = "data/claims.csv", n: int = 500, fraud_ratio: float = 0.2):
    df = generate_claims(n=n, fraud_ratio=fraud_ratio)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = save_claims()
    print(f"Generated {len(df)} claims -> data/claims.csv")
    print(df["label_is_fraud_synthetic"].value_counts())
