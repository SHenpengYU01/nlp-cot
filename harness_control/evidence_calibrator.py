"""Evidence calibration utilities for adaptive CoT harnesses.

The calibrator is intentionally lightweight: it does not know the gold answer.
It only turns heterogeneous probe outputs into a support score and an
escalation decision.
"""
import re
from collections import Counter
from typing import Any, Dict, Iterable, List


class RiskClassifier:
    """Detect hidden reasoning risks from the question and Base CoT output."""

    ARITHMETIC_MARKERS = {
        "percent", "percentage", "discount", "tax", "ratio", "rate", "speed",
        "distance", "time", "average", "probability", "interest", "fraction",
        "equation", "solve", "total", "cost", "price", "profit", "loss",
        "work", "mixture", "area", "volume",
    }
    SEMANTIC_MARKERS = {
        "except", "not", "least", "greatest", "more than", "less than",
        "at least", "at most", "only if", "unless", "after", "before",
    }
    APPROX_MARKERS = {
        "about", "approximately", "approx", "close", "near", "roughly",
        "round", "rounded",
    }

    def classify(self, question: str, options: Iterable[str], output: str) -> List[str]:
        text = f"{question} {' '.join(options or [])} {output}".lower()
        risks: List[str] = []

        number_count = len(re.findall(r"\d+(?:\.\d+)?", text))
        if number_count >= 4 or any(marker in text for marker in self.ARITHMETIC_MARKERS):
            risks.append("arithmetic_risk")
        if any(marker in text for marker in self.SEMANTIC_MARKERS):
            risks.append("semantic_misread_risk")
        if any(marker in text for marker in self.APPROX_MARKERS):
            risks.append("approximation_risk")
        if len(list(options or [])) >= 4:
            risks.append("option_mismatch_risk")
        if len(re.findall(r"[.;]", question)) >= 2:
            risks.append("constraint_missing_risk")

        if not risks:
            risks.append("low_risk")
        return risks


class ProbeSelector:
    """Select a small set of heterogeneous probes from risk types."""

    def select(self, risk_types: List[str], max_probes: int = 3) -> List[str]:
        selected: List[str] = []

        def add(name: str) -> None:
            if name not in selected and len(selected) < max_probes:
                selected.append(name)

        # For multiple-choice math, start with probes that inspect the answer
        # from the option side. These are less correlated with the forward CoT
        # path than simply asking the model to solve the problem again.
        add("back_substitution")

        if "option_mismatch_risk" in risk_types:
            add("option_elimination")
        if "arithmetic_risk" in risk_types or "approximation_risk" in risk_types:
            add("arithmetic_audit")
        if "semantic_misread_risk" in risk_types:
            add("equation_audit")
        if "constraint_missing_risk" in risk_types:
            add("constraint_check")
        add("alternative_cot")

        return selected[:max_probes]


class EvidenceAggregator:
    """Aggregate probe evidence into a support score and a decision."""

    AUDIT_PROBES = {
        "back_substitution",
        "option_elimination",
        "equation_audit",
        "arithmetic_audit",
        "constraint_check",
    }

    SUPPORT_WEIGHTS = {
        "alternative_cot": 1.0,
        "back_substitution": 2.0,
        "option_elimination": 2.0,
        "equation_audit": 2.0,
        "arithmetic_audit": 2.0,
        "constraint_check": 2.0,
    }

    def __init__(self, accept_threshold: float = 3.0, reject_threshold: float = 0.0):
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold

    def aggregate(self, base_answer: str, probe_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        score = 0.0
        critical_errors: List[str] = []
        suggested_answers: List[str] = []

        for probe in probe_results:
            probe_type = probe.get("probe_type", "")
            stance = probe.get("stance", "uncertain")
            suggested = (probe.get("suggested_answer") or "").upper()
            weight = self.SUPPORT_WEIGHTS.get(probe_type, 1.0)
            suggested_conflicts_with_base = bool(
                suggested
                and base_answer
                and suggested != base_answer
                and suggested in {"A", "B", "C", "D", "E"}
            )

            if suggested:
                suggested_answers.append(suggested)

            # Audit probes are stronger than a free-form stance label. If an
            # audit probe suggests a different concrete answer, it is direct
            # contradictory evidence even if the model accidentally wrote
            # "Evidence: SUPPORT".
            if probe_type in self.AUDIT_PROBES and suggested_conflicts_with_base:
                score -= weight + 2.0
                critical_errors.append(
                    probe.get("issue", "")
                    or f"{probe_type} suggested {suggested}, conflicting with base {base_answer}"
                )
                continue

            if stance == "support":
                score += weight
            elif stance == "contradict":
                score -= weight + 1.0
                critical_errors.append(probe.get("issue", "") or f"{probe_type} contradicted base")
            else:
                score -= 0.25

            if suggested_conflicts_with_base:
                score -= 1.0

        answer_votes = Counter(a for a in suggested_answers if a)
        top_suggested = answer_votes.most_common(1)[0][0] if answer_votes else ""

        if critical_errors or score <= self.reject_threshold:
            decision = "escalate"
        elif score >= self.accept_threshold:
            decision = "accept"
        else:
            decision = "uncertain"

        return {
            "support_score": score,
            "decision": decision,
            "critical_errors": critical_errors,
            "suggested_answer_votes": dict(answer_votes),
            "top_suggested_answer": top_suggested,
        }
