"""Experiment entropy management for long-running CoT projects."""
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class EntropyManager:
    """Scan runs, prompts, and reports for drift or stale artifacts."""

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)

    def build_report(self) -> Dict[str, Any]:
        runs = self._scan_runs()
        prompt_report = self._scan_prompts()
        report_checks = self._scan_report_mentions(runs["best_runs_by_strategy"])
        recommendations = self._recommend(runs, prompt_report, report_checks)

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "runs": runs,
            "prompts": prompt_report,
            "reports": report_checks,
            "recommendations": recommendations,
        }

    def save_report(self, output_dir: str = "experiments/gc_reports") -> str:
        report = self.build_report()
        out_dir = self.root / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"gc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return str(path)

    def _scan_runs(self) -> Dict[str, Any]:
        runs_dir = self.root / "experiments" / "runs"
        broken = []
        suspicious = []
        valid = []
        fingerprints = defaultdict(list)
        best_by_strategy = {}

        for path in sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []:
            try:
                with path.open("r", encoding="utf-8") as f:
                    record = json.load(f)
            except Exception as exc:
                broken.append({"file": str(path), "reason": str(exc)})
                continue

            config = record.get("config", {})
            metrics = record.get("metrics", {})
            results = record.get("results", [])
            strategy = config.get("strategy", "unknown")
            run_id = record.get("run_id", path.stem)
            total = metrics.get("total")

            if total is not None and len(results) != int(total):
                suspicious.append({
                    "run_id": run_id,
                    "file": str(path),
                    "reason": "results length does not match metrics.total",
                })

            error_count = sum(1 for r in results if "ERROR:" in str(r.get("output", "")))
            if results and error_count / len(results) > 0.1:
                suspicious.append({
                    "run_id": run_id,
                    "file": str(path),
                    "reason": f"high error rate: {error_count}/{len(results)}",
                })

            if strategy == "multi_agent_debate":
                suspicious.append({
                    "run_id": run_id,
                    "file": str(path),
                    "reason": "multi_agent_debate token cost may be under-counted if only summaries are saved",
                })

            fingerprint = (
                strategy,
                config.get("task"),
                config.get("model"),
                config.get("dataset_split"),
                config.get("n_samples"),
            )
            fingerprints[fingerprint].append(run_id)

            accuracy = float(metrics.get("accuracy", 0.0))
            current = best_by_strategy.get(strategy)
            if current is None or accuracy > current["accuracy"]:
                best_by_strategy[strategy] = {
                    "run_id": run_id,
                    "accuracy": accuracy,
                    "file": str(path),
                }

            valid.append(run_id)

        duplicates = [
            {"config": list(map(str, key)), "run_ids": ids}
            for key, ids in fingerprints.items()
            if len(ids) > 1
        ]

        return {
            "runs_checked": len(valid) + len(broken),
            "valid_runs": valid,
            "broken_runs": broken,
            "duplicate_runs": duplicates,
            "suspicious_runs": suspicious,
            "best_runs_by_strategy": best_by_strategy,
        }

    def _scan_prompts(self) -> Dict[str, Any]:
        prompts_dir = self.root / "prompts"
        strategy_dir = self.root / "strategies"
        prompt_files = sorted(prompts_dir.glob("*.txt")) if prompts_dir.exists() else []
        strategy_text = ""
        if strategy_dir.exists():
            for path in strategy_dir.glob("*.py"):
                try:
                    strategy_text += path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass

        unused = []
        for prompt in prompt_files:
            if prompt.name not in strategy_text:
                unused.append(str(prompt))
        return {
            "prompts_checked": len(prompt_files),
            "unused_prompts": unused,
        }

    def _scan_report_mentions(self, best_by_strategy: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        report_paths = [
            self.root / "README.md",
            self.root / "report" / "md" / "report.md",
            self.root / "experiments" / "report_100_sample_20260615.md",
        ]
        checked = []
        for path in report_paths:
            if path.exists():
                checked.append(str(path))
        return {
            "reports_checked": checked,
            "note": "Automatic numeric synchronization is not applied; compare these reports with best_runs_by_strategy.",
            "best_runs_by_strategy": best_by_strategy,
        }

    def _recommend(self, runs: Dict[str, Any], prompts: Dict[str, Any], reports: Dict[str, Any]) -> List[str]:
        recommendations = []
        if runs["broken_runs"]:
            recommendations.append("Inspect or remove broken run JSON files.")
        if runs["duplicate_runs"]:
            recommendations.append("Archive duplicate runs or mark the canonical run per strategy.")
        if runs["suspicious_runs"]:
            recommendations.append("Review suspicious runs, especially token accounting and ERROR-heavy runs.")
        if prompts["unused_prompts"]:
            recommendations.append("Remove unused prompt templates or document why they are kept.")
        if reports["reports_checked"]:
            recommendations.append("Verify report tables against best_runs_by_strategy before submission.")
        if not recommendations:
            recommendations.append("No major entropy issues detected.")
        return recommendations
