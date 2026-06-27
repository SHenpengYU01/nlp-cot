"""
AQuA (Algebraic Word Problems) task definition.
Dataset: https://github.com/deepmind/AQuA
"""
import json
import os
import re
from typing import List, Dict, Any

from .base import BaseTask


class AQuATask(BaseTask):
    """AQuA math word problem task."""

    def __init__(self, data_dir: str = "data/AQuA", **kwargs):
        super().__init__(name="aqua", **kwargs)
        self.data_dir = data_dir

    def load_data(self, split: str = "dev") -> List[Dict[str, Any]]:
        filepath = os.path.join(self.data_dir, f"{split}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"AQuA data file not found: {filepath}")

        examples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
        return examples

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format an AQuA example into a prompt."""
        question = example.get("question", "")
        options = example.get("options", [])
        options_text = " ".join(options) if options else ""
        return f"Question: {question}\nOptions: {options_text}\n"

    def extract_answer(self, text: str) -> str:
        """
        Extract the answer choice (A-E) from generated text.
        Looks for patterns like 'Answer: B', 'correct answer is C', etc.
        """
        text = (text or "").strip()
        if text.startswith("[EMPTY_RESPONSE"):
            return ""

        # Try explicit answer patterns (ordered by specificity)
        patterns = [
            r"(?im)^\s*(?:final\s+answer|answer|答案|最终答案|正确答案)\s*[:：\-]?\s*[\(\[]?\s*([A-E])\s*[\)\]]?\s*$",
            r"(?i)(?:final\s+answer|answer|correct\s*(?:answer|option)|答案|最终答案|正确答案)\s*(?:is|为|是|:|：|-)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?",
            r"(?i)(?:option|choice|选项)\s*[\(\[]?\s*([A-E])\s*[\)\]]?\s*(?:is|为|是)?\s*(?:correct|right|正确|答案)?",
            r"(?i)\b([A-E])\)\s*(?:is correct|is the answer|is right|正确|是答案)",
            r"(?i)(?:choose|select|pick|选择)\s*[\(\[]?\s*([A-E])\s*[\)\]]?\b",
            r"\\boxed\{\s*([A-E])\s*\}",
            r"【\s*([A-E])\s*】",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.MULTILINE)
            if match:
                return match.group(1).upper()

        # Fallback: find the last standalone A-E letter
        matches = re.findall(r"\b([A-E])\b", text)
        if matches:
            return matches[-1].upper()

        return ""

    def evaluate(self, prediction: str, example: Dict[str, Any]) -> bool:
        """Check if the predicted answer matches the ground truth."""
        correct = example.get("correct", "")
        return prediction.upper() == correct.upper()

    def get_task_info(self) -> Dict[str, Any]:
        info = super().get_task_info()
        info["dataset"] = "AQuA"
        info["data_dir"] = self.data_dir
        return info
