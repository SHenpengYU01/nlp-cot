"""
Retrieval-Augmented Chain-of-Thought (RAG+COT) strategy.

This version upgrades the previous one-shot retrieval pipeline into a small
IRCoT-style loop:
- build a topic-aware seed query
- retrieve evidence
- let the model propose a follow-up retrieval query from the evidence gap
- retrieve again and merge the evidence
- solve with a compact, cited evidence block
"""
import os
import re
from typing import Dict, Any, List, Tuple

from retrieval.simple_retriever import build_query_candidates, infer_math_topic

from .base import BaseStrategy


class RAGCOTStrategy(BaseStrategy):
    """RAG-enhanced COT with interleaved retrieval and reasoning."""

    def harness_subsystems(self) -> Dict[str, bool]:
        return {
            "instructions": True,
            "tools": True,
            "environment": True,
            "state": True,
            "feedback": False,
        }

    def __init__(
        self,
        model,
        task,
        retriever,
        prompt_template_path: str = "prompts/rag_cot.txt",
        base_prompt_path: str = "prompts/base_cot.txt",
        top_k: int = 3,
        max_hops: int = 2,
        max_context_docs: int = 5,
        use_query_planner: bool = True,
        planner_temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(
            name="rag_cot",
            model=model,
            task=task,
            top_k=top_k,
            max_hops=max_hops,
            max_context_docs=max_context_docs,
            use_query_planner=use_query_planner,
            planner_temperature=planner_temperature,
            **kwargs
        )
        self.retriever = retriever
        self.top_k = top_k
        self.max_hops = max(1, max_hops)
        self.max_context_docs = max(1, max_context_docs)
        self.use_query_planner = use_query_planner
        self.planner_temperature = planner_temperature
        self.prompt_template_path = prompt_template_path
        self.base_prompt_path = base_prompt_path
        self.prompt_template = self._load_template(prompt_template_path)
        self.base_template = self._load_template(base_prompt_path)

    def _load_template(self, path: str) -> str:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _summarize_options(self, options: List[str]) -> str:
        return " ".join(options) if options else ""

    def _seed_queries(self, question: str, options_text: str) -> List[str]:
        base_query = f"{question} {options_text}".strip()
        queries = build_query_candidates(base_query)
        if options_text and options_text not in queries:
            queries.append(options_text)
        return queries

    def _dedupe_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep the highest-scoring copy of each unique content snippet."""
        best_by_content: Dict[str, Dict[str, Any]] = {}
        for doc in docs:
            content = (doc.get("content", "") or "").strip()
            if not content:
                continue
            key = re.sub(r"\s+", " ", content.lower())
            current = best_by_content.get(key)
            if current is None or doc.get("score", 0.0) > current.get("score", 0.0):
                best_by_content[key] = doc
        deduped = list(best_by_content.values())
        deduped.sort(key=lambda x: (x.get("score", 0.0), x.get("source", "")), reverse=True)
        return deduped

    def _format_retrieval_trace(self, trace: List[Dict[str, Any]]) -> str:
        lines = []
        for hop in trace:
            lines.append(
                f"Hop {hop['hop']}: query={hop['query']!r} | docs={hop['num_docs']}"
            )
            for idx, doc in enumerate(hop.get("docs", []), 1):
                content = doc.get("content", "")
                lines.append(
                    f"  [{idx}] score={doc.get('score', 0.0):.3f} "
                    f"source={doc.get('source', 'unknown')} topic={doc.get('topic', 'general')} "
                    f"{content[:140]}"
                )
        return "\n".join(lines)

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return ""
        lines = []
        for idx, doc in enumerate(docs[: self.max_context_docs], 1):
            content = doc.get("content", "").strip()
            source = doc.get("source", "unknown")
            score = doc.get("score", 0.0)
            topic = doc.get("topic", "general")
            lines.append(
                f"[{idx}] ({source} | topic={topic} | score={score:.3f}) {content}"
            )
        return "\n".join(lines)

    def _plan_follow_up_query(
        self,
        question: str,
        options_text: str,
        current_query: str,
        trace: List[Dict[str, Any]],
        query_temperature: float,
    ) -> str:
        """Ask the model for a focused next-hop retrieval query."""
        if not self.use_query_planner:
            return current_query

        evidence_block = self._format_retrieval_trace(trace[-1:])
        planner_prompt = (
            "You are planning the next retrieval hop for a math reasoning system.\n"
            "Identify the single most useful missing formula, concept, or fact.\n"
            "Output exactly one line in the form:\n"
            "Next query: <short search query>\n\n"
            f"Question: {question}\n"
            f"Options: {options_text}\n"
            f"Current query: {current_query}\n\n"
            f"Retrieved evidence:\n{evidence_block}\n"
        )
        outputs = self.model.generate(
            planner_prompt,
            temperature=query_temperature,
            max_tokens=96,
            n=1,
        )
        raw = outputs[0] if outputs else ""
        match = re.search(r"Next query\s*:\s*(.+)", raw, re.IGNORECASE)
        candidate = match.group(1).strip() if match else raw.strip()
        candidate = re.sub(r"[\r\n]+", " ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate:
            return current_query
        if len(candidate) > 140:
            candidate = candidate[:140].rsplit(" ", 1)[0].strip()
        return candidate or current_query

    def _interleaved_retrieve(
        self,
        question: str,
        options_text: str,
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str]:
        """Run a short retrieve-plan-retrieve loop and return trace + docs."""
        queries = self._seed_queries(question, options_text)
        current_query = queries[0] if queries else question
        trace: List[Dict[str, Any]] = []
        collected_docs: List[Dict[str, Any]] = []
        final_query = current_query

        for hop in range(self.max_hops):
            docs = self.retriever.retrieve(current_query, top_k=top_k)
            docs_with_meta = [
                {
                    **doc,
                    "hop": hop + 1,
                    "query": current_query,
                }
                for doc in docs
            ]
            trace.append({
                "hop": hop + 1,
                "query": current_query,
                "num_docs": len(docs_with_meta),
                "docs": docs_with_meta,
            })
            collected_docs.extend(docs_with_meta)
            final_query = current_query

            if hop >= self.max_hops - 1:
                break
            if not docs_with_meta and len(queries) > hop + 1:
                current_query = queries[hop + 1]
                continue
            current_query = self._plan_follow_up_query(
                question,
                options_text,
                current_query,
                trace,
                self.planner_temperature,
            )
            if current_query == final_query:
                break

        return trace, self._dedupe_docs(collected_docs), final_query, infer_math_topic(question) or "general"

    def _build_prompt(self, question: str, options_text: str, context: str, retrieval_trace: str) -> str:
        if self.prompt_template:
            return self.prompt_template.format(
                question=question,
                options=options_text,
                context=context,
                retrieval_trace=retrieval_trace,
            )

        base = self.base_template or (
            "You are solving a math word problem. Think step by step and explain your reasoning clearly.\n\n"
            "Question: {question}\n"
            "Options: {options}\n\n"
            "At the end of your response, you must state your final answer choice on a single line in exactly this format:\n"
            "Answer: X\n"
            "where X is one of A, B, C, D, or E."
        )
        context_block = (
            f"Relevant knowledge:\n{context}\n\n"
            f"Retrieval trace:\n{retrieval_trace}\n\n"
            "Use the retrieved knowledge carefully and prefer formulas that directly match the question. "
        ) if context else ""
        return context_block + base.format(question=question, options=options_text)

    def run(self, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        question = example.get("question", "")
        options = example.get("options", [])
        options_text = self._summarize_options(options)

        top_k = kwargs.get("top_k", self.top_k)

        print(f"    [RAG] Running interleaved retrieval with top-{top_k} and {self.max_hops} hops...", end=" ")
        trace, docs, final_query, topic = self._interleaved_retrieve(question, options_text, top_k)
        print(f"collected {len(docs)} unique docs")

        for hop in trace:
            print(f"      hop {hop['hop']}: query={hop['query']!r} docs={hop['num_docs']}")
            for i, doc in enumerate(hop.get("docs", []), 1):
                print(
                    f"        [{i}] score={doc.get('score', 0.0):.3f} "
                    f"{doc.get('content', '')[:70]}..."
                )

        context = self._format_context(docs)
        retrieval_trace = self._format_retrieval_trace(trace)
        prompt = self._build_prompt(question, options_text, context, retrieval_trace)

        outputs = self.model.generate(
            prompt,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
            n=kwargs.get("n", 1),
        )

        raw_output = outputs[0] if outputs else ""
        prediction = self.task.extract_answer(raw_output)

        return {
            "prediction": prediction,
            "output": raw_output,
            "metadata": {
                "prompt": prompt,
                "retrieved_context": docs,
                "retrieval_trace": trace,
                "retrieval_trace_text": retrieval_trace,
                "seed_query": build_query_candidates(f"{question} {options_text}".strip())[0] if question else "",
                "final_query": final_query,
                "topic": topic,
                "num_docs": len(docs),
                "num_hops": len(trace),
                "num_samples": len(outputs),
            },
        }
