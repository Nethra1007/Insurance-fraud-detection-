"""
pages/5_📜_Rule_Base.py
--------------------------
Browse all rules, filter by category, and add/edit rules — persisted back
to data/rules.json (and re-synced to the DB rules table).
"""

import json
import streamlit as st
import pandas as pd

from utils.ui_common import inject_css, get_knowledge_base
from database.db import get_session, save_rules_snapshot
from engine.rule_parser import load_rules

st.set_page_config(page_title="Rule Base", page_icon="📜", layout="wide")
inject_css()

RULES_PATH = "data/rules.json"

kb = get_knowledge_base()

st.title("📜 Rule Base")
st.caption(f"{len(kb.rules)} domain rules currently loaded from `{RULES_PATH}`.")

categories = ["All"] + kb.all_categories()
selected_cat = st.selectbox("Filter by category", categories)

rules_to_show = kb.rules if selected_cat == "All" else [
    r for r in kb.rules if r.category == selected_cat
]

st.markdown(f"**{len(rules_to_show)} rule(s)**")

for r in rules_to_show:
    with st.container():
        st.markdown(f"""
        <div class="rule-card">
            <span class="rid">{r.rule_id}</span><b>{r.name}</b>
            <span style="float:right; color:#9fb0d6;">{r.category} · confidence {r.confidence:.0%}</span>
            <br><span style="color:#c7d1e3;">{r.description}</span>
            <br><i style="color:#7f8ca3;">"{r.backward_question}"</i>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("➕ Add a new rule")

with st.form("add_rule_form"):
    c1, c2 = st.columns(2)
    with c1:
        rule_id = st.text_input("Rule ID (e.g. R34)")
        name = st.text_input("Name")
        description = st.text_area("Description")
        category = st.text_input("Category", value="custom")
    with c2:
        conclusion = st.text_input("Conclusion (goal this rule proves)")
        confidence = st.slider("Confidence", 0.0, 1.0, 0.6, 0.05)
        operator_type = st.selectbox("Antecedent operator", ["AND", "OR"])
        backward_question = st.text_input("Backward question")
        evidence_label = st.text_input("Evidence label", value="Custom Evidence")

    st.markdown("**Conditions** — one per line, format: `fact,op,value` "
                "(op is one of `> >= < <= == !=`, value can be a number, "
                "`true`/`false`, or text)")
    conditions_raw = st.text_area(
        "Conditions", value="claim_amount_ratio,>,2.0",
        help="Example: unusual_claim_amount,==,true",
    )

    submitted = st.form_submit_button("Save rule")

    if submitted:
        if not (rule_id and name and conclusion and conditions_raw.strip()):
            st.error("Rule ID, name, conclusion, and at least one condition are required.")
        else:
            conditions = []
            for line in conditions_raw.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    st.error(f"Could not parse condition line: '{line}'")
                    st.stop()
                fact, op, raw_val = parts
                if raw_val.lower() == "true":
                    value = True
                elif raw_val.lower() == "false":
                    value = False
                else:
                    try:
                        value = float(raw_val) if "." in raw_val else int(raw_val)
                    except ValueError:
                        value = raw_val
                conditions.append({"fact": fact, "op": op, "value": value})

            new_rule = {
                "rule_id": rule_id, "name": name, "description": description,
                "if": {"operator": operator_type, "conditions": conditions},
                "then": {"conclusion": conclusion, "confidence": confidence, "category": category},
                "backward_question": backward_question or f"Does '{conclusion}' hold?",
                "evidence_label": evidence_label,
            }

            with open(RULES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            if any(r["rule_id"] == rule_id for r in data["rules"]):
                st.error(f"Rule ID '{rule_id}' already exists — choose a different ID.")
            else:
                data["rules"].append(new_rule)
                with open(RULES_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                # Re-sync DB snapshot and clear the cached KnowledgeBase
                refreshed = load_rules(RULES_PATH)
                session = get_session()
                save_rules_snapshot(session, refreshed)
                session.close()
                st.cache_resource.clear()

                st.success(f"Rule {rule_id} saved. Reload the page to see it applied everywhere.")
                st.rerun()
