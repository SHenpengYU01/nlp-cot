"""Rule-based router for adaptive CoT strategy escalation."""
from typing import Dict


class AdaptiveRouter:
    """Choose the next strategy from observed failure signals."""

    def __init__(
        self,
        base_threshold: float = 0.80,
        vote_threshold: float = 0.70,
        verifier_threshold: float = 0.70,
        max_stage: str = "debate",
    ):
        self.base_threshold = base_threshold
        self.vote_threshold = vote_threshold
        self.verifier_threshold = verifier_threshold
        self.max_stage = max_stage

    def accept_base(self, analysis: Dict) -> bool:
        return analysis.get("reliability", 0.0) >= self.base_threshold

    def accept_vote(self, analysis: Dict) -> bool:
        return analysis.get("reliability", 0.0) >= self.vote_threshold

    def accept_verifier(self, analysis: Dict) -> bool:
        return analysis.get("reliability", 0.0) >= self.verifier_threshold

    def next_after_base(self, failure_type: str, knowledge_needed: bool = False) -> str:
        if knowledge_needed:
            return "rag_cot"
        if failure_type in {"format_error", "short_reasoning"}:
            return "few_shot_cot"
        return "prefix_consistency"

    def should_use_debate(self) -> bool:
        return self.max_stage == "debate"
