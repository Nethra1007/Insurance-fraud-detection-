"""
pages/6_ℹ️_About.py
----------------------
Methodology write-up and literature references, for the viva.
"""

import streamlit as st
from utils.ui_common import inject_css

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
inject_css()

st.title("ℹ️ About This Project")

st.markdown("""
## Insurance-Claim Fraud Detection using Backward Chaining Rule Inference

### Problem
Insurance fraud causes major financial losses industry-wide. Most existing
systems focus on **prediction** (statistical / ML scoring) with limited
**explanation** — an investigator sees a score, not a reason.

### Approach: Backward Chaining
This project uses **backward chaining**, a goal-driven AI reasoning
technique, instead of a black-box classifier:

1. Start with a **hypothesis**: *"Could this claim be fraudulent?"*
2. Search the domain rule base for rules whose conclusion matches the
   hypothesis.
3. Identify each rule's **required conditions** (antecedents).
4. Recursively **verify** those conditions — some are raw facts from the
   claim/policy/customer/vehicle/accident/history data; others are
   themselves the conclusion of *other* rules, so the engine recurses
   deeper.
5. Continue backward until **base facts** are reached (working memory) —
   the recursion bottoms out and results propagate back up.
6. Return the inference result: **risk level**, **triggered rules**,
   **evidence count**, **confidence**, and **investigation priority**,
   plus the full **Fraud Trail** — the exact chain of reasoning used.

This mirrors how a human investigator actually works: start suspicious,
then look for supporting evidence — rather than a purely statistical
score with no story behind it.

### Why backward chaining fits this domain
- **Explainable by construction** — every verdict carries the exact rules
  and facts that proved it, which is what audits and regulators need.
- **Expert-knowledge driven** — rules come directly from domain
  expertise; no large labelled training set is required to get a working
  system.
- **Extensible** — a new fraud indicator is added as a new rule in
  `data/rules.json`, with no retraining.

### System architecture

```mermaid
flowchart TD
    A[Claim Input] --> B[Data Preprocessing]
    B --> C[Working Memory: Facts]
    C --> D[Backward Chaining Engine]
    E[Knowledge Base: 30+ Rules] --> D
    D --> F[Fraud Risk & Evidence Module]
    F --> G[Fraud Trail Generator]
    G --> H[Investigator Dashboard]
```

### Module map

| Module | Responsibility | File |
|---|---|---|
| 1. Data Collection | Synthetic / real claims dataset | `utils/data_generator.py`, `data/claims.csv` |
| 2. Data Preprocessing | Raw claim → working-memory facts | `engine/knowledge_base.py` |
| 3. Knowledge Base | Rule parsing + storage | `engine/rule_parser.py`, `data/rules.json` |
| 4. Backward Chaining Engine | Core goal-driven inference | `engine/backward_chainer.py` |
| 5. Fraud Risk & Evidence | Risk level, confidence, priority | `engine/risk_scorer.py` |
| 6. Fraud Trail Generator | Proof tree + narrative | `engine/fraud_trail.py` |
| 7. Investigator Dashboard | This Streamlit app | `app.py`, `pages/` |

### References
- Aslam, F. et al. (2022). *Insurance fraud detection: Evidence from
  artificial intelligence and machine learning.*
- Farbmacher, H. et al. (2022). *An explainable attention network for
  fraud detection in claims management.*
- Hancock, J. T. et al. (2023). *Survey on categorical data for neural
  networks — applied to insurance fraud detection.*
- Liang, C. et al. (2020). *Uncovering insurance fraud conspiracy with
  network learning.*

*(Full citations as presented in the project review deck.)*
""")
