"""
Simple keyword-based retriever for quick RAG experiments.

The retriever is intentionally lightweight, but it now includes:
- query-topic inference for math word problems
- synonym-aware query expansion
- overlap + phrase + numeric matching for scoring
- richer metadata for IRCoT-style retrieval traces
"""
import json
import os
import re
from typing import List, Dict, Any, Iterable, Optional, Tuple

from .base import BaseRetriever


TOPIC_HINTS: Dict[str, Dict[str, Any]] = {
    "percentage": {
        "keywords": {
            "percentage", "percent", "discount", "tax", "profit", "loss",
            "increase", "decrease", "rate",
        },
        "queries": [
            "percentage formula discount tax profit loss",
            "percent increase decrease original value final value",
        ],
    },
    "ratio": {
        "keywords": {"ratio", "proportion", "proportional", "share"},
        "queries": [
            "ratio proportion solve share divide quantity",
        ],
    },
    "speed_time": {
        "keywords": {"speed", "distance", "time", "average speed", "rate"},
        "queries": [
            "speed distance time average speed rate formula",
        ],
    },
    "work_rate": {
        "keywords": {"work", "rate", "job", "together", "combined"},
        "queries": [
            "work rate combined rate job completion formula",
        ],
    },
    "mixture": {
        "keywords": {"mixture", "concentration", "solution", "mix", "blend"},
        "queries": [
            "mixture concentration weighted average formula",
        ],
    },
    "geometry": {
        "keywords": {"area", "perimeter", "circumference", "radius", "triangle", "rectangle", "circle"},
        "queries": [
            "geometry area perimeter circle triangle rectangle formula",
        ],
    },
    "probability": {
        "keywords": {"probability", "chance", "independent", "random", "outcomes"},
        "queries": [
            "probability independent events favorable outcomes formula",
        ],
    },
    "algebra": {
        "keywords": {"equation", "solve", "linear", "quadratic", "variable", "expression"},
        "queries": [
            "algebra linear equation quadratic formula solve for x",
        ],
    },
    "sequence": {
        "keywords": {"sequence", "arithmetic", "geometric", "term", "nth", "sum"},
        "queries": [
            "arithmetic sequence geometric sequence nth term sum formula",
        ],
    },
}


def infer_math_topic(text: str) -> Optional[str]:
    """Infer the closest math topic from a query string."""
    lowered = (text or "").lower()
    for topic, payload in TOPIC_HINTS.items():
        if any(keyword in lowered for keyword in payload["keywords"]):
            return topic
    return None


def build_query_candidates(query: str) -> List[str]:
    """Build a small ordered set of query variants for retrieval hops."""
    candidates = [query.strip()]
    topic = infer_math_topic(query)
    if topic:
        candidates.extend(TOPIC_HINTS[topic]["queries"])
    normalized = re.sub(r"\s+", " ", query.strip())
    if normalized and normalized not in candidates:
        candidates.append(normalized)
    return [c for c in candidates if c]


class SimpleKeywordRetriever(BaseRetriever):
    """Keyword-based retriever with a local knowledge base."""

    def __init__(self, knowledge_path: str = "data/knowledge_base.json", top_k: int = 3, **kwargs):
        super().__init__(name="simple_keyword", **kwargs)
        self.knowledge_path = knowledge_path
        self.default_top_k = top_k
        self.entries: List[Dict[str, Any]] = []
        self._load_knowledge()

    def _load_knowledge(self):
        if not os.path.exists(self.knowledge_path):
            # Initialize with an empty list if file doesn't exist
            self.entries = []
            return
        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.entries = data if isinstance(data, list) else data.get("entries", [])

    def _tokenize(self, text: str) -> set:
        """Tokenize text into lowercase word/number tokens."""
        text = text.lower()
        tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)
        # Filter out very common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "shall",
                      "can", "need", "dare", "ought", "used", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "as", "into",
                      "through", "during", "before", "after", "above", "below",
                      "between", "under", "and", "but", "or", "yet", "so", "if",
                      "because", "although", "though", "while", "where", "when",
                      "that", "which", "who", "whom", "whose", "what", "this",
                      "these", "those", "i", "you", "he", "she", "it", "we", "they",
                      "me", "him", "her", "us", "them", "my", "your", "his",
                      "its", "our", "their", "what", "how", "all", "any", "both",
                      "each", "few", "more", "most", "other", "some", "such",
                      "no", "nor", "not", "only", "own", "same", "than", "too",
                      "very", "just", "then", "now", "here", "there", "why",
                      "again", "once", "upon", "out", "up", "down", "off", "over",
                      "also", "get", "got", "gets", "one", "two", "three", "four",
                      "five", "six", "seven", "eight", "nine", "ten"}
        return set(t for t in tokens if t not in stop_words and len(t) > 2)

    def _phrase_bonus(self, query: str, content: str) -> float:
        """Give a small bonus for obvious formula/topic phrase overlap."""
        q = query.lower()
        c = content.lower()
        topic = infer_math_topic(q)
        bonus = 0.0
        if topic and topic in TOPIC_HINTS:
            for keyword in TOPIC_HINTS[topic]["keywords"]:
                if keyword in c and keyword in q:
                    bonus += 0.15
        for phrase in ("formula", "equation", "rate", "discount", "tax", "probability", "ratio"):
            if phrase in q and phrase in c:
                bonus += 0.05
        return min(bonus, 0.35)

    def _score(self, query: str, query_tokens: set, entry_tokens: set, content: str) -> float:
        """Compute a relevance score with overlap, numeric and phrase bonuses."""
        if not entry_tokens:
            return 0.0
        intersection = query_tokens & entry_tokens
        union = query_tokens | entry_tokens
        if not union:
            return 0.0
        jaccard = len(intersection) / len(union)
        overlap_bonus = min(len(intersection), 6) * 0.04
        numeric_bonus = 0.0
        query_numbers = {tok for tok in query_tokens if re.fullmatch(r"\d+(?:\.\d+)?", tok)}
        if query_numbers and query_numbers & entry_tokens:
            numeric_bonus = 0.08
        topic = infer_math_topic(query)
        topic_bonus = 0.0
        if topic and any(keyword in content.lower() for keyword in TOPIC_HINTS[topic]["keywords"]):
            topic_bonus = 0.12
        return min(1.0, jaccard + overlap_bonus + numeric_bonus + topic_bonus + self._phrase_bonus(query, content))

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k if top_k is not None else self.default_top_k
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.entries:
            return []

        scored = []
        for entry in self.entries:
            content = entry.get("content", "")
            entry_tokens = self._tokenize(content)
            score = self._score(query, query_tokens, entry_tokens, content)
            if score > 0:
                scored.append({
                    "content": content,
                    "score": score,
                    "source": entry.get("source", "unknown"),
                    "topic": infer_math_topic(content) or infer_math_topic(query) or "general",
                    "matched_terms": sorted((query_tokens & entry_tokens), key=len, reverse=True)[:8],
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]
