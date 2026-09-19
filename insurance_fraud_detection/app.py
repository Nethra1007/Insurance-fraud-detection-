"""
app.py
---------
Main Streamlit entry point (Home page). Additional pages live in pages/
and appear automatically in the sidebar (Streamlit multi-page app).

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

from utils.ui_common import inject_css, get_knowledge_base, load_claims_df, metric_card

st.set_page_config(
    page_title="Insurance Fraud Detection — Backward Chaining",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

kb = get_knowledge_base()
df = load_claims_df()

st.title("🕵️ Insurance-Claim Fraud Detection")
st.caption("Backward Chaining Rule Inference — goal-driven, explainable fraud investigation")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Total Claims (dataset)", f"{len(df):,}")
with col2:
    fraud_designed = int(df["label_is_fraud_synthetic"].sum())
    metric_card("Design-Fraud Claims", f"{fraud_designed:,}")
with col3:
    metric_card("Rules in Knowledge Base", f"{len(kb.rules)}")
with col4:
    metric_card("Rule Categories", f"{len(kb.all_categories())}")

st.markdown("###")

left, right = st.columns([3, 2])

with left:
    st.subheader("How it works")
    st.markdown("""
    Traditional fraud-detection systems mostly **predict** — they output a
    score with little explanation of *why*. This project instead asks a
    **goal-driven question**: *"Could this claim be fraudulent?"* — and
    works **backward** through a rule base to verify supporting evidence,
    exactly the way a human investigator reasons.

    **Pipeline:**
    1. **Data Collection** — claim, policy, customer, vehicle, accident and
       history attributes (`utils/data_generator.py`, or a real Kaggle /
       API feed).
    2. **Data Preprocessing** — raw attributes become clean working-memory
       facts (`engine/knowledge_base.py`).
    3. **Knowledge Base** — 30+ domain rules, IF–THEN, with confidence and
       category (`data/rules.json`).
    4. **Backward Chaining Engine** — goal-driven recursive inference
       (`engine/backward_chainer.py`) — the core AI technique.
    5. **Fraud Risk & Evidence** — risk level, confidence, investigation
       priority (`engine/risk_scorer.py`).
    6. **Fraud Trail Generator** — a visual proof tree + step-by-step
       narrative (`engine/fraud_trail.py`).
    7. **Investigator Dashboard** — this app.
    """)

with right:
    st.subheader("Architecture")
    st.markdown("""
    ```mermaid
    flowchart TD
        A[Claim Input] --> B[Data Preprocessing]
        B --> C[Working Memory: Facts]
        C --> D[Backward Chaining Engine]
        E[Knowledge Base: Rules] --> D
        D --> F[Fraud Risk & Evidence]
        F --> G[Fraud Trail Generator]
        G --> H[Investigator Dashboard]
    ```
    """)
    st.info("Use the sidebar to **Investigate a Claim**, view **Analytics**, "
            "explore past **Fraud Trails**, or browse the **Rule Base**.")

st.markdown("---")
st.caption("Principles of AI — Insurance-Claim Fraud Detection using Backward Chaining Rule Inference")
