"""
pages/3_📊_Analytics_Dashboard.py
------------------------------------
Runs inference across the whole dataset (cached) and shows aggregate
analytics: risk distribution, fraud rate by category, time series,
top triggered rules.
"""

import streamlit as st
import pandas as pd

from engine.knowledge_base import build_working_memory
from engine.backward_chainer import BackwardChainer
from engine.risk_scorer import score_risk
from utils.ui_common import inject_css, get_knowledge_base, load_claims_df, metric_card
from utils.visualizations import (
    risk_distribution_donut, fraud_by_category_bar, claims_over_time, top_triggered_rules_bar,
)

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
inject_css()

kb = get_knowledge_base()
df = load_claims_df()

st.title("📊 Analytics Dashboard")
st.caption("Aggregate view of backward-chaining results across the full claims dataset.")


@st.cache_data(show_spinner="Running backward chaining across all claims...")
def run_batch_inference(_df_hash: int) -> pd.DataFrame:
    rows = []
    rule_hits = {}
    for _, row in df.iterrows():
        claim = row.to_dict()
        wm = build_working_memory(claim)
        chainer = BackwardChainer(kb, wm)
        node = chainer.infer("fraudulent_claim")
        risk = score_risk(node)
        rows.append({
            "claim_id": claim["claim_id"],
            "claim_type": claim["claim_type"],
            "location": claim.get("location", "Unknown"),
            "vehicle_age": claim.get("vehicle_age", 0),
            "claim_date": claim["claim_date"],
            "is_fraudulent": risk.is_fraudulent,
            "risk_level": risk.risk_level,
            "confidence": risk.overall_confidence,
        })
        for r in risk.triggered_rules:
            rule_hits[r["rule_id"] + " — " + r["name"]] = rule_hits.get(
                r["rule_id"] + " — " + r["name"], 0) + 1
    return pd.DataFrame(rows), rule_hits


results_df, rule_hits = run_batch_inference(len(df))

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Total Claims", f"{len(results_df):,}")
with c2:
    fraud_count = int(results_df["is_fraudulent"].sum())
    metric_card("Flagged Fraudulent", f"{fraud_count:,}")
with c3:
    rate = fraud_count / len(results_df) * 100 if len(results_df) else 0
    metric_card("Fraud Rate", f"{rate:.1f}%")
with c4:
    avg_conf = results_df.loc[results_df["is_fraudulent"], "confidence"].mean()
    metric_card("Avg. Confidence (flagged)", f"{avg_conf:.0%}" if pd.notna(avg_conf) else "—")

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(risk_distribution_donut(results_df), use_container_width=True)
with col_b:
    st.plotly_chart(fraud_by_category_bar(results_df, "claim_type"), use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(fraud_by_category_bar(results_df, "location"), use_container_width=True)
with col_d:
    st.plotly_chart(top_triggered_rules_bar(rule_hits), use_container_width=True)

st.plotly_chart(claims_over_time(results_df), use_container_width=True)

st.markdown("### Vehicle age vs fraud rate")
vehicle_bins = pd.cut(results_df["vehicle_age"], bins=[0, 3, 6, 10, 15, 100],
                       labels=["0-3", "4-6", "7-10", "11-15", "15+"])
vehicle_df = results_df.assign(vehicle_age_band=vehicle_bins)
st.plotly_chart(fraud_by_category_bar(vehicle_df, "vehicle_age_band"), use_container_width=True)

with st.expander("View full results table"):
    st.dataframe(results_df, use_container_width=True, hide_index=True)
