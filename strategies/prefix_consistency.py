"""
Prefix Consistency strategy.
Based on: https://arxiv.org/abs/2605.07654

Sample multiple Chain-of-Thought traces, truncate each partway through,
regenerate the remainder, and weight votes by how often the original answer
reappears under regeneration (prefix consistency).

This reaches standard Self-Consistency plateau accuracy with up to 21x
fewer tokens (median 4.6x).
"""
import os
from collections import Counter
from typing import Dict, Any, List

from .base import BaseStrategy


class PrefixConsistencyStrategy(BaseStrategy):
    """
    Prefix-Consistency Weighted Majority Voting (PC-WMV).

    1. Generate N CoT traces for the same question.
    2. Truncate each trace at a fixed ratio (default 50%).
    3. Regenerate the remainder from the prefix multiple times.
    4. Compute prefix consistency = fraction of regenerations that reproduce
       the original answer.
    5. Aggregate answers via weighted majority voting using prefix consistency
       as the per-sample weight.
    """

    def harness_subsystems(self) -> Dict[str, bool]:
        return {
            "instructions": True,
            "tools": False,
            "environment": True,
            "state": True,      # tracks multiple reasoning paths
            "feedback": True,   # prefix consistency is a reliability feedback signal
        }

    def __init__(
        self,
        model,
        task,
        prompt_template_path: str = "prompts/base_cot.txt",
        n_paths: int = 5,
        truncation_ratio: float = 0.5,
        regen_count: int = 3,
        weight_fn: str = "linear",
        temperature: float = 0.7,
        min_paths: int = 3,
        early_stop_agreement: float = 1.0,
        min_regen: int = 2,
        target_consistency: float = 1.0,
        max_empty_initial: int = 3,
        **kwargs
    ):
        super().__init__(name="prefix_consistency", model=model, task=task, **kwargs)
        self.n_paths = n_paths
        self.truncation_ratio = truncation_ratio
        self.regen_count = regen_count
        self.weight_fn = weight_fn
        self.temperature = temperature
        self.min_paths = min_paths
        self.early_stop_agreement = early_stop_agreement
        self.min_regen = min_regen
        self.target_consistency = target_consistency
        self.max_empty_initial = max_empty_initial
        self.prompt_template_path = prompt_template_path
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        if not os.path.exists(self.prompt_template_path):
            return (
                "You are solving a math word problem. Think step by step and explain your reasoning clearly.\n\n"
                "Question: {question}\n"
                "Options: {options}\n\n"
                "At the end of your response, you must state your final answer choice on a single line in exactly this format:\n"
                "Answer: X\n"
                "where X is one of A, B, C, D, or E."
            )
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            return f.read()

    def _truncate_text(self, text: str, ratio: float) -> str:
        """Truncate text at the given character ratio."""
        if not text:
            return text
        truncate_at = max(1, int(len(text) * ratio))
        return text[:truncate_at]

    def _build_regeneration_prompt(self, question: str, options_text: str, prefix: str) -> str:
        """Build a continuation prompt that preserves the original task context."""
        return (
            "You are continuing a partially written chain-of-thought solution for the same multiple-choice math problem.\n"
            "Keep the given prefix unchanged in meaning, continue the reasoning, and end with exactly one final answer line.\n\n"
            f"Question: {question}\n"
            f"Options: {options_text}\n\n"
            "Partial reasoning prefix:\n"
            f"{prefix}\n\n"
            "Continue from the prefix and finish the solution.\n"
            "At the end, write the final answer on a single line in exactly this format:\n"
            "Answer: X\n"
            "where X is one of A, B, C, D, or E."
        )

    def _compute_weight(self, consistency: float) -> float:
        """Map consistency rate to vote weight."""
        if self.weight_fn == "linear":
            return consistency
        elif self.weight_fn == "quadratic":
            return consistency ** 2
        elif self.weight_fn == "cubic":
            return consistency ** 3
        elif self.weight_fn == "unanimous":
            return 1.0 if consistency >= 1.0 else 0.0
        else:
            return consistency

    def _vote_agreement(self, predictions: List[str]) -> float:
        """Return top-vote ratio among non-empty predictions."""
        valid = [p.upper() for p in predictions if p]
        if not valid:
            return 0.0
        counts = Counter(valid)
        return counts.most_common(1)[0][1] / len(valid)

    def _can_stop_regen(
        self,
        attempts: int,
        match_count: int,
        regen_count: int,
        min_regen: int,
        target_consistency: float,
    ) -> bool:
        """Stop regeneration when consistency is already decided."""
        if attempts < min_regen:
            return False
        current = match_count / attempts if attempts > 0 else 0.0
        if current >= target_consistency:
            return True
        remaining = regen_count - attempts
        best_possible = (match_count + remaining) / regen_count if regen_count > 0 else 0.0
        return best_possible < target_consistency

    def run(self, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        question = example.get("question", "")
        options = example.get("options", [])
        options_text = " ".join(options)

        prompt = self.prompt_template.format(question=question, options=options_text)

        n = kwargs.get("n_paths", self.n_paths)
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 1024)
        trunc_ratio = kwargs.get("truncation_ratio", self.truncation_ratio)
        regen_count = kwargs.get("regen_count", self.regen_count)
        min_paths = kwargs.get("min_paths", self.min_paths)
        early_stop_agreement = kwargs.get("early_stop_agreement", self.early_stop_agreement)
        min_regen = kwargs.get("min_regen", self.min_regen)
        target_consistency = kwargs.get("target_consistency", self.target_consistency)
        max_empty_initial = kwargs.get("max_empty_initial", self.max_empty_initial)

        # Phase 1: Generate initial CoT traces
        outputs = []
        init_predictions: List[str] = []
        empty_initial = 0
        for i in range(n):
            print(f"    [Prefix-Consistency] Generating path {i+1}/{n}...", end=" ")
            batch = self.model.generate(
                prompt,
                temperature=temp,
                max_tokens=max_tokens,
                n=1,
            )
            outputs.extend(batch)
            pred_preview = self.task.extract_answer(batch[0]) if batch else ""
            init_predictions.append(pred_preview)
            if not pred_preview:
                empty_initial += 1
                if empty_initial >= max_empty_initial:
                    print("-> EMPTY")
                    print(
                        f"    [Prefix-Consistency] Early stop initial paths: "
                        f"{empty_initial} empty predictions; escalating to verifier"
                    )
                    break
            if len(outputs) >= min_paths:
                agreement = self._vote_agreement(init_predictions)
                if agreement >= early_stop_agreement:
                    print(f"-> {pred_preview}")
                    print(
                        f"    [Prefix-Consistency] Early stop initial paths: "
                        f"agreement={agreement:.2f} after {len(outputs)} paths"
                    )
                    break
            print(f"→ {pred_preview}")

        # Extract initial predictions if an unusual batch response changed lengths
        if len(init_predictions) != len(outputs):
            init_predictions = []
            for raw in outputs:
                pred = self.task.extract_answer(raw)
                init_predictions.append(pred)

        # Phase 2: Truncate and regenerate for each trace
        # Structure: regen_results[i] = list of regen outputs for trace i
        regen_results: List[List[str]] = []
        regen_predictions: List[List[str]] = []
        consistencies: List[float] = []

        for i, (raw_output, init_pred) in enumerate(zip(outputs, init_predictions)):
            prefix = self._truncate_text(raw_output, trunc_ratio)
            regen_prompt = self._build_regeneration_prompt(question, options_text, prefix)
            print(f"    [Prefix-Consistency] Trace {i+1}/{n} trunc@{trunc_ratio} → regen×{regen_count}...", end=" ")

            regen_outputs = []
            regen_preds = []
            match_count = 0
            for j in range(regen_count):
                regen_batch = self.model.generate(
                    regen_prompt,
                    temperature=temp,
                    max_tokens=max_tokens,
                    n=1,
                )
                regen_text = regen_batch[0] if regen_batch else ""
                regen_pred = self.task.extract_answer(regen_text)
                regen_outputs.append(regen_text)
                regen_preds.append(regen_pred)

                if regen_pred and regen_pred.upper() == init_pred.upper():
                    match_count += 1
                attempts = j + 1
                if self._can_stop_regen(
                    attempts=attempts,
                    match_count=match_count,
                    regen_count=regen_count,
                    min_regen=min_regen,
                    target_consistency=target_consistency,
                ):
                    break

            attempts = len(regen_outputs)
            consistency = match_count / attempts if attempts > 0 else 0.0
            consistencies.append(consistency)
            regen_results.append(regen_outputs)
            regen_predictions.append(regen_preds)
            print(f"regen_preds={regen_preds} consistency={consistency:.2f}")

        # Phase 3: Weighted majority voting by prefix consistency
        weighted_votes: Counter = Counter()
        for pred, consistency in zip(init_predictions, consistencies):
            if pred:
                weight = self._compute_weight(consistency)
                weighted_votes[pred.upper()] += weight

        if not weighted_votes:
            fallback_votes = Counter(p.upper() for p in init_predictions if p)
            final_prediction = fallback_votes.most_common(1)[0][0] if fallback_votes else ""
        elif sum(weighted_votes.values()) <= 0:
            fallback_votes = Counter(p.upper() for p in init_predictions if p)
            final_prediction = fallback_votes.most_common(1)[0][0] if fallback_votes else ""
        else:
            final_prediction = weighted_votes.most_common(1)[0][0]

        # Build summary output for logging
        summary_lines = [
            f"=== Prefix-Consistency ({n} paths, trunc={trunc_ratio}, regen={regen_count}) ===",
            f"Weight function: {self.weight_fn}",
            f"Weighted votes: {dict(weighted_votes)}",
            f"Final Answer: {final_prediction}",
            "",
            "--- Initial paths ---",
        ]
        for idx, (pred, cons) in enumerate(zip(init_predictions, consistencies)):
            summary_lines.append(f"Path {idx+1}: pred={pred} consistency={cons:.2f}")

        summary_lines.extend(["", "--- Regeneration examples (Path 1) ---"])
        if regen_results:
            for j, regen in enumerate(regen_results[0][:2]):
                summary_lines.append(f"Regen {j+1}: {regen[:200]}...")

        summary_output = "\n".join(summary_lines)

        return {
            "prediction": final_prediction,
            "output": summary_output,
            "metadata": {
                "prompt": prompt,
                "n_paths": n,
                "truncation_ratio": trunc_ratio,
                "regen_count": regen_count,
                "weight_fn": self.weight_fn,
                "min_paths": min_paths,
                "early_stop_agreement": early_stop_agreement,
                "min_regen": min_regen,
                "target_consistency": target_consistency,
                "max_empty_initial": max_empty_initial,
                "all_outputs": outputs,
                "all_predictions": init_predictions,
                "regen_results": regen_results,
                "regen_predictions": regen_predictions,
                "consistencies": consistencies,
                "weighted_votes": dict(weighted_votes),
            },
        }
