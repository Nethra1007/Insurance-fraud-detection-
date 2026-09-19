"""
utils/preprocessing.py
-------------------------
Lightweight data-cleaning helpers applied to the raw claims DataFrame
before individual rows are handed to engine.knowledge_base.build_working_memory().

This is deliberately separate from build_working_memory(): this module
cleans the *dataset* (missing values, type coercion), while
build_working_memory() derives *rule-ready facts* from one already-clean
claim record.
"""

import pandas as pd
import numpy as np

REQUIRED_NUMERIC = [
    "claim_amount", "avg_claim_for_type", "policy_coverage",
    "previous_claims", "claim_frequency", "vehicle_age", "age",
]
REQUIRED_BOOL = [
    "previous_fraud", "damage_match", "report_consistency", "police_report",
    "high_risk_location", "premium_payment_irregular", "no_witnesses",
    "multiple_policies_same_vehicle", "accident_requires_police_report",
]


def clean_claims_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value handling + type coercion (Module 2: Data Preprocessing)."""
    df = df.copy()

    for col in REQUIRED_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in REQUIRED_BOOL:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    if "document_status" in df.columns:
        df["document_status"] = df["document_status"].fillna("complete")

    if "claims_in_last_30_days" in df.columns:
        df["claims_in_last_30_days"] = df["claims_in_last_30_days"].fillna(0).astype(int)

    return df


def normalize_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Simple min-max normalization — used only for analytics visuals, not
    for rule evaluation (rules use raw thresholds for interpretability)."""
    series = df[col].astype(float)
    span = series.max() - series.min()
    if span == 0:
        return series * 0
    return (series - series.min()) / span


def encode_categorical(df: pd.DataFrame, col: str) -> pd.Series:
    """Simple label encoding for a categorical column (analytics use only)."""
    return df[col].astype("category").cat.codes
