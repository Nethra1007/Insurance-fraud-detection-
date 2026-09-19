"""
pages/4_🌳_Fraud_Trail_Explorer.py
--------------------------------------
Browse past inference runs stored in SQLite and re-render their Fraud
Trail tree + narrative, with JSON export.
"""

import json
import streamlit as st
import pandas as pd

from utils.ui_common import inject_css, risk_pill_html
from utils.visualizations import render_tree_figure
from database.db import init_db, get_session, get_recent_logs

st.set_page_config(page_title="Fraud Trail Explorer", page_icon="🌳", layout="wide")
inject_css()
init_db()

st.title("🌳 Fraud Trail Explorer")
st.caption("Revisit any past inference run without re-running the engine.")

session = get_session()
logs = get_recent_logs(session, limit=200)
session.close()

if not logs:
    st.warning("No inference logs yet. Run a few claims through "
               "**🔍 Investigate Claim** or the Analytics Dashboard first.")
    st.stop()

log_options = {
    f"{log.claim_id} — {log.risk_level} ({log.created_at:%Y-%m-%d %H:%M})": log
    for log in logs
}
choice = st.selectbox("Select a past inference", list(log_options.keys()))
selected = log_options[choice]
trail = json.loads(selected.result_json)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"**Fraud Status**  \n{'🚩 FRAUDULENT' if trail['is_fraudulent'] else '✅ Not Flagged'}")
with c2:
    st.markdown(f"**Risk Level**  \n{risk_pill_html(trail['risk_level'])}", unsafe_allow_html=True)
with c3:
    st.markdown(f"**Confidence**  \n{trail['overall_confidence']:.0%}")

st.markdown("### Reasoning Tree (zoom / pan / hover)")
fig = render_tree_figure(trail["tree"])
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Narrative")
st.code(trail["narrative"], language=None)

st.markdown("### Triggered rules")
if trail["triggered_rules"]:
    rules_df = pd.DataFrame(trail["triggered_rules"])[
        ["rule_id", "name", "category", "confidence", "evidence_label"]
    ]
    rules_df["confidence"] = (rules_df["confidence"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

st.download_button(
    "⬇ Export as JSON",
    data=json.dumps(trail, indent=2),
    file_name=f"fraud_trail_{trail['claim_id']}.json",
    mime="application/json",
)

st.markdown("---")
st.markdown("### Recent inference log")
log_table = pd.DataFrame([{
    "claim_id": l.claim_id, "risk_level": l.risk_level,
    "is_fraudulent": l.is_fraudulent, "confidence": l.overall_confidence,
    "triggered_rules": l.triggered_rule_ids, "created_at": l.created_at,
} for l in logs])
st.dataframe(log_table, use_container_width=True, hide_index=True)
