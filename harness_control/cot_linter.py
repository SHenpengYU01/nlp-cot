"""Lightweight linter for Chain-of-Thought outputs.

The linter does not need gold labels. It inspects whether a model output is
usable: answer format, reasoning depth, answer conflicts, and length.
"""
import re
from typing import Any, Callable, Dict, Optional


class CoTLinter:
    """Check CoT output quality and return a reliability signal."""

    ANSWER_RE = re.compile(
        r"(?i)(?:final\s+answer|answer|答案)\s*[:：]?\s*[\(\[]?\s*([A-E])\s*[\)\]]?"
    )
    STEP_RE = re.compile(
        r"(?:^|\n)\s*(?:step\s*\d+[:.)]?|\d+[:.)]\s+|\(\d+\)\s+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        answer_extractor: Optional[Callable[[str], str]] = None,
        min_steps: int = 2,
        min_chars: int = 80,
    ):
        self.answer_extractor = answer_extractor
        self.min_steps = min_steps
        self.min_chars = min_chars

    def analyze(self, output: str) -> Dict[str, Any]:
        output = output or ""
        answers = self._find_answers(output)
        extracted = ""
        if self.answer_extractor is not None:
            try:
                extracted = self.answer_extractor(output) or ""
            except Exception:
                extracted = ""
        if extracted and extracted.upper() not in answers:
            answers.append(extracted.upper())

        unique_answers = sorted(set(a for a in answers if a in {"A", "B", "C", "D", "E"}))
        step_count = self._count_steps(output)
        too_short = len(output.strip()) < self.min_chars
        has_answer = len(unique_answers) > 0
        conflicting_answers = len(unique_answers) > 1
        enough_steps = step_count >= self.min_steps

        penalties = 0.0
        failure_types = []
        if not has_answer:
            penalties += 0.45
            failure_types.append("format_error")
        if conflicting_answers:
            penalties += 0.25
            failure_types.append("conflicting_answer")
        if not enough_steps:
            penalties += 0.25
            failure_types.append("short_reasoning")
        if too_short:
            penalties += 0.15
            failure_types.append("low_quality")

        quality_score = max(0.0, min(1.0, 1.0 - penalties))
        valid = has_answer and not conflicting_answers and enough_steps and not too_short
        primary_failure = failure_types[0] if failure_types else "ok"

        return {
            "valid": valid,
            "quality_score": quality_score,
            "failure_type": primary_failure,
            "failure_types": failure_types,
            "signals": {
                "has_answer": has_answer,
                "answers": unique_answers,
                "step_count": step_count,
                "conflicting_answers": conflicting_answers,
                "too_short": too_short,
                "length": len(output.strip()),
            },
        }

    def _find_answers(self, text: str):
        return [m.group(1).upper() for m in self.ANSWER_RE.finditer(text or "")]

    def _count_steps(self, text: str) -> int:
        text = text or ""
        matches = self.STEP_RE.findall(text)
        if matches:
            return len(matches)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        if paragraphs:
            return len(paragraphs)
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
        return len(lines)
