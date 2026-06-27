"""Dry-run verification for Evidence-Calibrated Harness and entropy checks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

from harness_control import EntropyManager
from models.base import BaseModel
from strategies.evidence_calibrated_harness import EvidenceCalibratedHarnessStrategy
from tasks.aqua_task import AQuATask


class DummyModel(BaseModel):
    def __init__(self):
        super().__init__("dummy")
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        # First answer is intentionally low-quality so the harness escalates.
        if self.calls == 1:
            return ["Answer: B"]
        return [
            "Step 1: Identify the quantities.\n"
            "Step 2: Compute carefully.\n"
            "Step 3: Match the result to the options.\n"
            "Answer: A"
        ]

    def chat(self, messages, **kwargs):
        return self.generate(messages[-1]["content"] if messages else "", **kwargs)


def test_evidence_calibrated_strategy():
    model = DummyModel()
    task = AQuATask(data_dir="data/AQuA")
    strategy = EvidenceCalibratedHarnessStrategy(
        model=model,
        task=task,
        n_paths=1,
        regen_count=1,
        n_prompts=1,
        n_agents=1,
        n_rounds=1,
        n_shots=1,
        max_stage="verifier",
    )
    example = {
        "question": "If 2 + 2 = ?, choose the answer.",
        "options": ["A) 4", "B) 3", "C) 5", "D) 6", "E) 7"],
        "correct": "A",
    }
    result = strategy.run(example, temperature=0.1, max_tokens=128)
    assert result["prediction"] in {"A", "B", "C", "D", "E"}
    assert "harness_trace" in result["metadata"]
    assert result["metadata"]["activated_strategies"][0] == "base_cot"
    assert len(result["metadata"]["harness_trace"]) >= 1
    print("  evidence_calibrated_harness dry-run: PASS")


def test_entropy_manager_report():
    manager = EntropyManager(project_root=os.path.join(os.path.dirname(__file__), ".."))
    report = manager.build_report()
    assert "runs" in report
    assert "prompts" in report
    assert "recommendations" in report
    print("  entropy manager report build: PASS")


def main():
    print("=" * 60)
    print("feat-012 Dry-run Verification: Harness Feedback + Entropy")
    print("=" * 60)
    test_evidence_calibrated_strategy()
    test_entropy_manager_report()
    print("=" * 60)
    print("PASS: feat-012 checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
