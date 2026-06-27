"""Failure analysis helpers for Harness-Guided CoT."""
import math
from collections import Counter
from typing import Any, Dict, List


class FailureAnalyzer:
    """Convert raw strategy metadata into reliability and failure signals."""

    KNOWLEDGE_KEYWORDS = {
        "speed", "distance", "time", "rate", "probability", "percent", "discount",
        "interest", "average", "ratio", "geometry", "triangle", "circle", "area",
        "volume", "permutation", "combination", "work", "mixture",
    }

    def analyze_initial(self, lint_result: Dict[str, Any]) -> Dict[str, Any]:
        score = float(lint_result.get("quality_score", 0.0))
        failure = lint_result.get("failure_type", "low_quality")
        return {
            "reliability": score,
            "failure_type": failure if score < 0.8 else "ok",
            "reasons": lint_result.get("failure_types", []),
        }

    def analyze_vote(self, result: Dict[str, Any]) -> Dict[str, Any]:
        metadata = result.get("metadata", {})
        predictions = [
            (p or "").upper().strip()
            for p in metadata.get("all_predictions", [])
            if (p or "").upper().strip() in {"A", "B", "C", "D", "E"}
        ]
        weighted_votes = {
            k.upper(): float(v)
            for k, v in metadata.get("weighted_votes", {}).items()
            if k and k.upper() in {"A", "B", "C", "D", "E"}
        }

        if weighted_votes:
            total = sum(weighted_votes.values())
            top = max(weighted_votes.values()) if weighted_votes else 0.0
            agreement = top / total if total > 0 else 0.0
        elif predictions:
            counts = Counter(predictions)
            agreement = counts.most_common(1)[0][1] / len(predictions)
        else:
            agreement = 0.0

        entropy = self._vote_entropy(predictions)
        vote_reliability = max(0.0, min(1.0, agreement * (1.0 - 0.25 * entropy)))

        # Prefix Consistency has an extra reliability signal: whether a partial
        # reasoning prefix can regenerate the same answer. Initial votes can be
        # unanimous while the underlying reasoning is unstable, e.g. A/A/A with
        # regeneration consistency 0.50. In that case the vote should not be
        # treated as reliable.
        consistencies = [
            float(c)
            for c in metadata.get("consistencies", [])
            if isinstance(c, (int, float))
        ]
        avg_consistency = sum(consistencies) / len(consistencies) if consistencies else None
        min_consistency = min(consistencies) if consistencies else None

        if avg_consistency is not None:
            reliability = max(0.0, min(1.0, vote_reliability * avg_consistency))
        else:
            reliability = vote_reliability

        if reliability >= 0.7:
            failure = "ok"
        elif avg_consistency is not None and avg_consistency < 0.7:
            failure = "prefix_low_stability"
        else:
            failure = "vote_divergence"
        return {
            "reliability": reliability,
            "agreement": agreement,
            "entropy": entropy,
            "vote_reliability": vote_reliability,
            "avg_consistency": avg_consistency,
            "min_consistency": min_consistency,
            "failure_type": failure,
            "predictions": predictions,
            "weighted_votes": weighted_votes,
        }

    def analyze_verifier(self, result: Dict[str, Any]) -> Dict[str, Any]:
        metadata = result.get("metadata", {})
        scores = [float(s) for s in metadata.get("path_scores", []) if isinstance(s, (int, float))]
        best_score = float(metadata.get("best_path_avg_score", max(scores) if scores else 0.0))
        reliability = max(0.0, min(1.0, best_score / 10.0))
        failure = "ok" if reliability >= 0.7 else "low_verifier_score"
        return {
            "reliability": reliability,
            "best_score": best_score,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "failure_type": failure,
        }

    def analyze_debate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        metadata = result.get("metadata", {})
        valid_vote_ratio = float(metadata.get("valid_vote_ratio", 0.0))
        top_vote_ratio = float(metadata.get("top_vote_ratio", 0.0))
        reliability = max(0.0, min(1.0, valid_vote_ratio * top_vote_ratio))
        failure = "ok" if reliability >= 0.7 else "weak_debate_consensus"
        return {
            "reliability": reliability,
            "valid_vote_ratio": valid_vote_ratio,
            "top_vote_ratio": top_vote_ratio,
            "empty_votes": int(metadata.get("empty_votes", 0)),
            "failure_type": failure,
        }

    def detect_knowledge_need(self, question: str) -> bool:
        text = (question or "").lower()
        return any(keyword in text for keyword in self.KNOWLEDGE_KEYWORDS)

    def _vote_entropy(self, predictions: List[str]) -> float:
        if not predictions:
            return 1.0
        counts = Counter(predictions)
        total = len(predictions)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p, 2)
        max_entropy = math.log(max(2, len(counts)), 2)
        return entropy / max_entropy if max_entropy > 0 else 0.0
