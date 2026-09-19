"""
engine/backward_chainer.py
----------------------------
⭐ Module 4 — Backward Chaining Engine (Phase 2: AI Reasoning Engine)

This is the core AI technique of the project: pure, hand-written,
class-based backward chaining (goal-driven inference) — no external
inference/expert-system library is used, so every step is demonstrable
in the viva.

Algorithm (classic goal-driven resolution, as in Prolog):

    infer(goal):
        1. If `goal` is already a raw fact in working memory -> base case,
           return its truth value directly (a LEAF node).
        2. Otherwise, look up every rule whose `conclusion` equals `goal`.
           Several rules may share one conclusion -> they act as
           alternative (OR) ways to prove it.
        3. For each candidate rule, recursively try to prove every fact/
           sub-goal in its antecedents (its `conditions`). This is where
           the "backward" part happens: proving a rule's antecedents may
           itself require firing more rules, deeper in the tree.
        4. If a rule's antecedents are satisfied (AND/OR per the rule's
           own operator), the rule fires: `goal` is proven, and the rule +
           its child proofs are recorded as a TreeNode for the Fraud Trail.
        5. If no rule can fire and `goal` is not a known fact, `goal`
           cannot be proven (result = False).

A visited-set prevents infinite recursion if the rule base ever contains
a cycle (defensive; the shipped rule base is acyclic).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from engine.knowledge_base import KnowledgeBase
from engine.rule_parser import Rule, Condition, OPS


@dataclass
class TreeNode:
    """One node of the backward-chaining proof tree.

    This tree IS the Fraud Trail — engine/fraud_trail.py renders it into a
    JSON tree (for the Plotly/NetworkX visual) and a step-by-step
    narrative.
    """
    goal: str
    proven: bool
    is_leaf: bool
    rule: Optional[Rule] = None
    confidence: float = 0.0
    children: List["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "proven": self.proven,
            "is_leaf": self.is_leaf,
            "rule_id": self.rule.rule_id if self.rule else None,
            "rule_name": self.rule.name if self.rule else None,
            "backward_question": self.rule.backward_question if self.rule else None,
            "evidence_label": self.rule.evidence_label if self.rule else None,
            "confidence": self.confidence,
            "children": [c.to_dict() for c in self.children],
        }


class BackwardChainer:
    """Goal-driven inference engine over a KnowledgeBase + working memory."""

    def __init__(self, knowledge_base: KnowledgeBase, facts: Dict[str, Any]):
        self.kb = knowledge_base
        self.facts = facts
        self.trail: List[TreeNode] = []   # top-level proofs run this session

    # ------------------------------------------------------------------
    def infer(self, hypothesis: str) -> TreeNode:
        """Public entry point. hypothesis e.g. 'fraudulent_claim'."""
        node = self._prove_goal(hypothesis, visited=set())
        self.trail.append(node)
        return node

    # ------------------------------------------------------------------
    def _prove_goal(self, goal: str, visited: set) -> TreeNode:
        """Prove a GOAL — i.e. a name that is the `conclusion` of one or
        more rules (e.g. 'unusual_claim_amount', 'fraudulent_claim')."""
        # Cycle guard
        if goal in visited:
            return TreeNode(goal=goal, proven=False, is_leaf=False)
        visited = visited | {goal}

        candidates = self.kb.rules_for(goal)
        if not candidates:
            # No rule concludes this goal — fall back to treating it as a
            # plain boolean fact in working memory (defensive fallback).
            value = self.facts.get(goal, False)
            return TreeNode(goal=goal, proven=bool(value), is_leaf=True,
                             confidence=1.0 if value else 0.0)

        # Try each candidate rule (OR across rules sharing a conclusion)
        for rule in candidates:
            child_nodes = [self._prove_condition(cond, visited) for cond in rule.conditions]
            satisfied = (
                any(c.proven for c in child_nodes) if rule.operator_type == "OR"
                else all(c.proven for c in child_nodes)
            )
            if satisfied:
                return TreeNode(
                    goal=goal, proven=True, is_leaf=False, rule=rule,
                    confidence=rule.confidence, children=child_nodes,
                )

        # No rule fired for this goal
        return TreeNode(goal=goal, proven=False, is_leaf=False)

    # ------------------------------------------------------------------
    def _prove_condition(self, cond: Condition, visited: set) -> TreeNode:
        """Prove a single CONDITION (fact, op, value) from a rule's
        antecedent list. Two cases:

        1. `cond.fact` is itself the conclusion of other rule(s) — it's a
           sub-goal (e.g. 'unusual_claim_amount'), so recurse into
           `_prove_goal`. Sub-goal conditions are conventionally written
           as `{"fact": "...", "op": "==", "value": true}`.
        2. `cond.fact` is a raw fact in working memory (e.g.
           'claim_amount_ratio') — this is a LEAF: apply the condition's
           own operator/threshold directly against working memory,
           instead of just checking truthiness.
        """
        fact = cond.fact

        if fact in self.kb.rules_by_conclusion:
            return self._prove_goal(fact, visited)

        actual = self.facts.get(fact)
        if actual is None:
            return TreeNode(goal=fact, proven=False, is_leaf=True, confidence=0.0)

        proven = OPS[cond.op](actual, cond.value)
        return TreeNode(goal=fact, proven=bool(proven), is_leaf=True,
                         confidence=1.0 if proven else 0.0)


# ----------------------------------------------------------------------
# Helpers for flattening a proof tree (used by risk_scorer / fraud_trail)
# ----------------------------------------------------------------------
def collect_fired_rules(node: TreeNode) -> List[Rule]:
    fired: List[Rule] = []
    if node.rule is not None and node.proven:
        fired.append(node.rule)
    for child in node.children:
        fired.extend(collect_fired_rules(child))
    return fired


def collect_true_leaves(node: TreeNode) -> List[str]:
    leaves: List[str] = []
    if node.is_leaf and node.proven:
        leaves.append(node.goal)
    for child in node.children:
        leaves.extend(collect_true_leaves(child))
    return leaves
