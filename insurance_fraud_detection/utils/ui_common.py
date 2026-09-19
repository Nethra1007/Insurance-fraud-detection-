"""
utils/ui_common.py
---------------------
Shared Streamlit helpers: custom CSS (dark investigator theme), cached
KnowledgeBase loader, and small formatting helpers used across every page.
"""

import streamlit as st
import pandas as pd
from engine.knowledge_base import KnowledgeBase
from database.db import init_db, get_session, save_rules_snapshot

RISK_COLORS = {
    "Low": "#2ecc71",
    "Medium": "#f1c40f",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}

RISK_EMOJI = {
    "Low": "🟢",
    "Medium": "🟡",
    "High": "🔴",
    "Critical": "⚫",
}


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: #1b2331;
        border: 1px solid #2b3648;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #9fb0d6; font-weight: 500; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }

    .risk-pill {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #10151f;
    }
    .rule-card {
        background: #1b2331;
        border-left: 3px solid #21c1a6;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
    }
    .rule-card .rid { color: #21c1a6; font-weight: 700; margin-right: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase("data/rules.json")
    init_db()
    session = get_session()
    save_rules_snapshot(session, kb.rules)
    session.close()
    return kb


@st.cache_data
def load_claims_df() -> pd.DataFrame:
    return pd.read_csv("data/claims.csv")


def risk_pill_html(risk_level: str) -> str:
    color = RISK_COLORS.get(risk_level, "#7f8c8d")
    emoji = RISK_EMOJI.get(risk_level, "⚪")
    return f'<span class="risk-pill" style="background:{color};">{emoji} {risk_level}</span>'


def metric_card(label: str, value: str):
    st.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)
