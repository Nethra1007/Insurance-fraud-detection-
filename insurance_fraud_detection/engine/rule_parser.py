"""
engine/rule_parser.py
----------------------
Module 3 (Knowledge Base) — helper: parses data/rules.json into structured
Rule objects the backward-chaining engine can consume.

Each rule in rules.json looks like:
    {
      "rule_id": "R01",
      "name": "...",
      "if": {"operator": "AND" | "OR", "conditions": [{"fact": ..., "op": ..., "value": ...}]},
      "then": {"conclusion": "...", "confidence": 0.85, "category": "financial"},
      "backward_question": "...",
      "evidence_label": "..."
    }

A "condition" can reference either:
  - a raw numeric/boolean fact computed during preprocessing
    (e.g. claim_amount_ratio > 2.0), or
  - a boolean sub-goal that is itself the conclusion of another rule
    (e.g. unusual_claim_amount == true) — this is what makes the rule base
    recursive / chainable for backward chaining.
"""

import json
import operator
from dataclasses import dataclass
from typing import List, Dict, Any

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass
class Condition:
    fact: str
    op: str
    value: Any

    def evaluate(self, working_memory: Dict[str, Any]) -> bool:
        if self.fact not in working_memory:
            return False
        actual = working_memory[self.fact]
        return OPS[self.op](actual, self.value)


@dataclass
class Rule:
    rule_id: str
    name: str
    description: str
    operator_type: str            # "AND" or "OR" across conditions
    conditions: List[Condition]
    conclusion: str
    confidence: float
    category: str
    backward_question: str
    evidence_label: str

    @property
    def antecedent_facts(self) -> List[str]:
        """The list of fact/sub-goal names this rule depends on — this is
        what the backward chainer recursively tries to prove."""
        return [c.fact for c in self.conditions]

    def is_satisfied(self, working_memory: Dict[str, Any]) -> bool:
        results = [c.evaluate(working_memory) for c in self.conditions]
        if self.operator_type == "OR":
            return any(results)
        return all(results)  # default AND


def load_rules(path: str = "data/rules.json") -> List[Rule]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rules: List[Rule] = []
    for r in raw["rules"]:
        conditions = [
            Condition(fact=c["fact"], op=c["op"], value=c["value"])
            for c in r["if"]["conditions"]
        ]
        rules.append(Rule(
            rule_id=r["rule_id"],
            name=r["name"],
            description=r.get("description", ""),
            operator_type=r["if"].get("operator", "AND"),
            conditions=conditions,
            conclusion=r["then"]["conclusion"],
            confidence=float(r["then"].get("confidence", 0.5)),
            category=r["then"].get("category", "general"),
            backward_question=r.get("backward_question", ""),
            evidence_label=r.get("evidence_label", "Evidence"),
        ))
    return rules
