"""
utils/visualizations.py
--------------------------
Plotly charts for the Analytics Dashboard, and a NetworkX + Plotly
rendering of the backward-chaining proof tree for the Fraud Trail
Explorer / Investigate Claim pages.
"""

from typing import Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx


# ---------------------------------------------------------------------
# Analytics Dashboard charts
# ---------------------------------------------------------------------
def risk_distribution_donut(df: pd.DataFrame, risk_col: str = "risk_level"):
    counts = df[risk_col].value_counts().reset_index()
    counts.columns = ["risk_level", "count"]
    color_map = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}
    fig = px.pie(counts, names="risk_level", values="count", hole=0.55,
                 color="risk_level", color_discrete_map=color_map,
                 title="Risk Level Distribution")
    fig.update_traces(textinfo="percent+label")
    return fig


def fraud_by_category_bar(df: pd.DataFrame, category_col: str, fraud_col: str = "is_fraudulent"):
    grouped = df.groupby(category_col)[fraud_col].mean().reset_index()
    grouped[fraud_col] = (grouped[fraud_col] * 100).round(1)
    fig = px.bar(grouped, x=category_col, y=fraud_col,
                 labels={fraud_col: "Fraud rate (%)"},
                 title=f"Fraud Rate by {category_col.replace('_', ' ').title()}",
                 color=fraud_col, color_continuous_scale="Reds")
    return fig


def claims_over_time(df: pd.DataFrame, date_col: str = "claim_date", fraud_col: str = "is_fraudulent"):
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp["month"] = tmp[date_col].dt.to_period("M").astype(str)
    grouped = tmp.groupby("month").agg(
        total_claims=("claim_id", "count"),
        fraud_claims=(fraud_col, "sum"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["month"], y=grouped["total_claims"],
                              mode="lines+markers", name="Total claims"))
    fig.add_trace(go.Scatter(x=grouped["month"], y=grouped["fraud_claims"],
                              mode="lines+markers", name="Flagged as fraud"))
    fig.update_layout(title="Claims Over Time", xaxis_title="Month", yaxis_title="Count")
    return fig


def top_triggered_rules_bar(rule_counts: Dict[str, int]):
    items = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if not items:
        return go.Figure()
    labels, values = zip(*items)
    fig = px.bar(x=values, y=labels, orientation="h",
                 labels={"x": "Times triggered", "y": "Rule"},
                 title="Top 10 Triggered Rules")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------------
# Fraud Trail tree visualization (backward-chaining proof tree)
# ---------------------------------------------------------------------
def _add_nodes(graph: nx.DiGraph, node: Dict[str, Any], parent_id: str = None,
               counter: Dict[str, int] = None, node_id: str = None):
    if counter is None:
        counter = {"i": 0}
    if node_id is None:
        node_id = "n0"

    label = node.get("rule_name") or node["goal"]
    graph.add_node(node_id, label=label, proven=node["proven"],
                   is_leaf=node["is_leaf"], goal=node["goal"],
                   backward_question=node.get("backward_question"),
                   evidence_label=node.get("evidence_label"),
                   confidence=node.get("confidence", 0))
    if parent_id:
        graph.add_edge(parent_id, node_id)

    for child in node.get("children", []):
        counter["i"] += 1
        child_id = f"n{counter['i']}"
        _add_nodes(graph, child, node_id, counter, child_id)

    return graph


def build_tree_graph(tree_json: Dict[str, Any]) -> nx.DiGraph:
    """Builds a NetworkX DiGraph from the JSON tree produced by
    engine.fraud_trail.build_trail_tree()."""
    graph = nx.DiGraph()
    _add_nodes(graph, tree_json)
    return graph


def render_tree_figure(tree_json: Dict[str, Any]) -> go.Figure:
    """Renders the proof tree as an interactive Plotly figure using a
    top-down hierarchical layout (hypothesis at top, evidence at leaves).
    Node colour: green = proven, red = failed."""
    graph = build_tree_graph(tree_json)

    # Hierarchical layout (top-down): root at y=0, deeper nodes more negative
    levels: Dict[str, int] = {}
    root = "n0"
    levels[root] = 0
    for parent, child in nx.bfs_edges(graph, root):
        levels[child] = levels[parent] + 1

    # Group nodes by level for horizontal spacing
    by_level: Dict[int, list] = {}
    for node_id, lvl in levels.items():
        by_level.setdefault(lvl, []).append(node_id)

    pos = {}
    for lvl, node_ids in by_level.items():
        n = len(node_ids)
        for i, node_id in enumerate(node_ids):
            x = (i - (n - 1) / 2)
            pos[node_id] = (x, -lvl)

    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="#888", width=1.5), hoverinfo="none")

    node_x, node_y, colors, texts, hover = [], [], [], [], []
    for node_id, data in graph.nodes(data=True):
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        colors.append("#2ecc71" if data["proven"] else "#e74c3c")
        texts.append(data["label"][:22])
        hover_text = f"<b>{data['label']}</b><br>Proven: {data['proven']}"
        if data.get("backward_question"):
            hover_text += f"<br>Q: {data['backward_question']}"
        if data.get("evidence_label"):
            hover_text += f"<br>Evidence: {data['evidence_label']}"
        if data.get("confidence"):
            hover_text += f"<br>Confidence: {data['confidence']:.0%}"
        hover.append(hover_text)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=texts,
        textposition="bottom center", hovertext=hover, hoverinfo="text",
        marker=dict(size=26, color=colors, line=dict(width=2, color="white")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, height=520,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=30, b=10),
        title="Backward-Chaining Fraud Trail (green = proven, red = not proven)",
    )
    return fig
