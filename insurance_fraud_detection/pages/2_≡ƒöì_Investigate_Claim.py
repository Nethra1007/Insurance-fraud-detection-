"""
pages/2_🔍_Investigate_Claim.py
----------------------------------
Pick an existing claim or enter a new one, run the backward-chaining
engine, and view the risk verdict + Fraud Trail.
"""

import streamlit as st
import pandas as pd
import json

from engine.knowledge_base import build_working_memory
from engine.backward_chainer import BackwardChainer
from engine.risk_scorer import score_risk
from engine.fraud_trail import generate_fraud_trail
from utils.ui_common import inject_css, get_knowledge_base, load_claims_df, risk_pill_html, metric_card
from utils.visualizations import render_tree_figure
from database.db import init_db, get_session, upsert_claim, log_inference

st.set_page_config(page_title="Investigate Claim", page_icon="🔍", layout="wide")
inject_css()
init_db()

kb = get_knowledge_base()
df = load_claims_df()

st.title("🔍 Investigate Claim")
st.caption("Run the backward-chaining engine on an existing claim, or enter a new one.")

mode = st.radio("Claim source", ["Select existing claim", "Enter new claim"], horizontal=True)

claim: dict = {}

if mode == "Select existing claim":
    claim_id = st.selectbox("Claim ID", df["claim_id"].tolist())
    claim = df[df["claim_id"] == claim_id].iloc[0].to_dict()
    with st.expander("View raw claim attributes"):
        st.json(claim)
else:
    st.subheader("New claim form")
    c1, c2, c3 = st.columns(3)
    with c1:
        claim_id = st.text_input("Claim ID", value="CLM-NEW-1")
        claim_amount = st.number_input("Claim amount", min_value=0.0, value=150000.0, step=1000.0)
        claim_type = st.selectbox("Claim type", ["Collision", "Theft", "Total Loss",
                                                   "Windscreen Damage", "Fire", "Natural Calamity"])
        claim_date = st.date_input("Claim date")
        avg_claim_for_type = st.number_input("Average claim for this type", min_value=1.0, value=60000.0)
    with c2:
        policy_start_date = st.date_input("Policy start date")
        policy_expiry_date = st.date_input("Policy expiry date")
        policy_coverage = st.number_input("Policy coverage", min_value=1.0, value=500000.0)
        previous_claims = st.number_input("Previous claims count", min_value=0, value=0, step=1)
        claim_frequency = st.number_input("Claim frequency / year", min_value=0.0, value=0.5)
        previous_fraud = st.checkbox("Prior confirmed fraud record?")
    with c3:
        document_status = st.selectbox("Document status", ["complete", "incomplete"])
        damage_match = st.checkbox("Damage matches accident description?", value=True)
        report_consistency = st.checkbox("Report consistent with claim?", value=True)
        police_report = st.checkbox("Police report filed?", value=True)
        vehicle_age = st.number_input("Vehicle age (years)", min_value=0, value=5, step=1)
        no_witnesses = st.checkbox("No witnesses reported?")

    claim = {
        "claim_id": claim_id, "claim_amount": claim_amount, "claim_type": claim_type,
        "claim_date": str(claim_date), "avg_claim_for_type": avg_claim_for_type,
        "policy_start_date": str(policy_start_date), "policy_expiry_date": str(policy_expiry_date),
        "policy_coverage": policy_coverage, "previous_claims": previous_claims,
        "claim_frequency": claim_frequency, "previous_fraud": previous_fraud,
        "document_status": document_status, "damage_match": damage_match,
        "report_consistency": report_consistency, "police_report": police_report,
        "accident_requires_police_report": claim_type in ("Collision", "Theft", "Total Loss", "Fire"),
        "vehicle_age": vehicle_age, "no_witnesses": no_witnesses,
        "age": 35, "high_risk_location": False, "premium_payment_irregular": False,
        "multiple_policies_same_vehicle": False, "report_date_mismatch": False,
        "claims_in_last_30_days": 0,
    }

run = st.button("▶ Run Inference", type="primary")

if run:
    with st.spinner("Running backward-chaining inference..."):
        working_memory = build_working_memory(claim)
        chainer = BackwardChainer(kb, working_memory)
        fraud_node = chainer.infer("fraudulent_claim")
        risk = score_risk(fraud_node)
        trail = generate_fraud_trail(claim["claim_id"], fraud_node, risk)

        session = get_session()
        upsert_claim(session, claim)
        log_inference(session, trail)
        session.close()

    st.markdown("---")
    st.subheader("Result")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"**Fraud Status**  \n{'🚩 FRAUDULENT' if risk.is_fraudulent else '✅ Not Flagged'}")
    with r2:
        st.markdown(f"**Risk Level**  \n{risk_pill_html(risk.risk_level)}", unsafe_allow_html=True)
    with r3:
        metric_card("Confidence", f"{risk.overall_confidence:.0%}")
    with r4:
        metric_card("Evidence Points", f"{risk.evidence_count}")

    st.info(f"**Investigation priority:** {risk.investigation_priority}")

    st.markdown("### Fraud Trail — Reasoning Tree")
    st.caption("Green = proven along the successful reasoning path · Red = could not be proven. "
               "Hover a node for the rule's question, evidence, and confidence.")
    fig = render_tree_figure(trail["tree"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Step-by-step narrative")
    st.code(trail["narrative"], language=None)

    st.markdown("### Triggered rules")
    if risk.triggered_rules:
        rules_df = pd.DataFrame(risk.triggered_rules)[
            ["rule_id", "name", "category", "confidence", "evidence_label"]
        ]
        rules_df["confidence"] = (rules_df["confidence"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(rules_df, use_container_width=True, hide_index=True)
    else:
        st.write("No rules were triggered for this claim.")

    st.download_button(
        "⬇ Export Fraud Trail as JSON",
        data=json.dumps(trail, indent=2),
        file_name=f"fraud_trail_{claim['claim_id']}.json",
        mime="application/json",
    )
