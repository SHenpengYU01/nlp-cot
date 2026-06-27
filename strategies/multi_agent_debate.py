"""
Multi-Agent Debate strategy.

The strategy runs several role-specialized agents, asks them to critique each
other, lets them revise, and aggregates only valid A-E votes. Empty votes are
tracked but never counted as a majority.
"""
import concurrent.futures
from collections import Counter
from typing import Any, Dict, List

from .base import BaseStrategy


class MultiAgentDebateStrategy(BaseStrategy):
    """Multi-Agent Debate with cross-review and valid-vote aggregation."""

    def harness_subsystems(self) -> Dict[str, bool]:
        return {
            "instructions": True,
            "tools": False,
            "environment": True,
            "state": True,
            "feedback": True,
        }

    def __init__(
        self,
        model,
        task,
        n_agents: int = 5,
        n_rounds: int = 3,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(name="multi_agent_debate", model=model, task=task, **kwargs)
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.temperature = temperature
        self.agent_configs = self._build_agent_configs()

    def _build_agent_configs(self) -> List[Dict[str, Any]]:
        role_pool = [
            (
                "analyst",
                "You are a rigorous math analyst. Decompose conditions, derive step by step, and verify calculations.",
                0.3,
            ),
            (
                "critic",
                "You are a skeptical critic. Look for hidden assumptions, arithmetic mistakes, and edge cases.",
                0.8,
            ),
            (
                "intuitive_solver",
                "You are an intuitive solver. Use pattern recognition and known problem types to find a concise solution.",
                1.0,
            ),
            (
                "verifier",
                "You are a careful verifier. Check candidate answers by substitution, boundary conditions, or alternative derivations.",
                0.5,
            ),
            (
                "synthesizer",
                "You are a synthesizer. Compare the arguments and produce the most reliable final answer.",
                0.7,
            ),
        ]
        configs = []
        for i in range(self.n_agents):
            name, role, temp = role_pool[i % len(role_pool)]
            configs.append({
                "id": i,
                "name": name,
                "system_prompt": (
                    f"You are {name}. {role}\n"
                    "Always end your final response with exactly one line: Answer: X, "
                    "where X is one of A, B, C, D, or E."
                ),
                "temperature": temp,
            })
        return configs

    def _generate(self, agent_cfg: Dict[str, Any], prompt: str) -> str:
        outputs = self.model.generate(
            prompt=prompt,
            temperature=agent_cfg["temperature"],
            max_tokens=1024,
            n=1,
            system_prompt=agent_cfg["system_prompt"],
        )
        return outputs[0] if outputs else ""

    def _parallel_generate(self, prompt: str) -> Dict[int, str]:
        results: Dict[int, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_agents) as ex:
            futures = {
                ex.submit(self._generate, cfg, prompt): cfg["id"]
                for cfg in self.agent_configs
            }
            for future in concurrent.futures.as_completed(futures):
                agent_id = futures[future]
                try:
                    results[agent_id] = future.result()
                except Exception as exc:
                    results[agent_id] = f"ERROR: {exc}"
        return results

    def _cross_review(
        self,
        question: str,
        options_text: str,
        answers: Dict[int, str],
    ) -> Dict[int, str]:
        results: Dict[int, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_agents) as ex:
            def review_one(cfg):
                others = {
                    c["id"]: answers.get(c["id"], "")
                    for c in self.agent_configs
                    if c["id"] != cfg["id"]
                }
                others_text = "\n\n".join(
                    f"[{self.agent_configs[k]['name']}]\n{v}"
                    for k, v in others.items()
                )
                prompt = (
                    "Review the other agents' solutions. Identify logical gaps, "
                    "calculation errors, or unsupported assumptions. Do not give a final answer yet.\n\n"
                    f"Question: {question}\n"
                    f"Options: {options_text}\n\n"
                    f"Other agents' answers:\n{others_text}"
                )
                return cfg["id"], self._generate(cfg, prompt)

            futures = {ex.submit(review_one, cfg): cfg["id"] for cfg in self.agent_configs}
            for future in concurrent.futures.as_completed(futures):
                aid, critique = future.result()
                results[aid] = critique
        return results

    def _format_debate_context(
        self,
        answers: Dict[int, str],
        critiques: Dict[int, str],
    ) -> str:
        parts = []
        for cfg in self.agent_configs:
            aid = cfg["id"]
            parts.append(
                f"### {cfg['name']}\n"
                f"Answer draft:\n{answers.get(aid, '')}\n\n"
                f"Critique:\n{critiques.get(aid, '')}\n"
            )
        return "\n".join(parts)

    def _extract_predictions(self, answers: Dict[int, str]) -> List[str]:
        return [self.task.extract_answer(v) for v in answers.values()]

    def _check_convergence(self, old_answers: Dict[int, str], new_answers: Dict[int, str]) -> bool:
        old_valid = [p for p in self._extract_predictions(old_answers) if p]
        new_valid = [p for p in self._extract_predictions(new_answers) if p]
        return bool(old_valid) and old_valid == new_valid and len(set(new_valid)) == 1

    def _majority_vote(self, answers: Dict[int, str]) -> str:
        valid_predictions = [p for p in self._extract_predictions(answers) if p]
        if not valid_predictions:
            return ""
        return Counter(valid_predictions).most_common(1)[0][0]

    def run(self, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        question = example.get("question", "")
        options = example.get("options", [])
        options_text = " ".join(options)
        n_rounds = kwargs.get("n_rounds", self.n_rounds)

        round1_prompt = (
            "Solve the following multiple-choice math problem. Reason step by step. "
            "The last line must be exactly 'Answer: X' where X is A, B, C, D, or E.\n\n"
            f"Question: {question}\n"
            f"Options: {options_text}"
        )
        answers = self._parallel_generate(round1_prompt)
        debate_rounds = [
            {
                cfg["name"]: {"pred": self.task.extract_answer(answers.get(cfg["id"], ""))}
                for cfg in self.agent_configs
            }
        ]

        for _ in range(1, n_rounds):
            critiques = self._cross_review(question, options_text, answers)
            context = self._format_debate_context(answers, critiques)
            revise_prompt = (
                "Based on the debate record, solve the original problem again. "
                "If another agent made a stronger argument, revise your answer. "
                "The last line must be exactly 'Answer: X' where X is A, B, C, D, or E.\n\n"
                f"Original question: {question}\n"
                f"Options: {options_text}\n\n"
                f"Debate record:\n{context}"
            )
            new_answers = self._parallel_generate(revise_prompt)
            debate_rounds.append({
                cfg["name"]: {"pred": self.task.extract_answer(new_answers.get(cfg["id"], ""))}
                for cfg in self.agent_configs
            })
            if self._check_convergence(answers, new_answers):
                answers = new_answers
                break
            answers = new_answers

        final_prediction = self._majority_vote(answers)
        all_predictions = self._extract_predictions(answers)
        valid_predictions = [p for p in all_predictions if p]
        vote_counts = Counter(valid_predictions)
        empty_votes = len(all_predictions) - len(valid_predictions)
        valid_vote_ratio = len(valid_predictions) / len(all_predictions) if all_predictions else 0.0
        top_vote_ratio = (
            vote_counts.most_common(1)[0][1] / len(valid_predictions)
            if valid_predictions else 0.0
        )

        summary_lines = [
            f"=== Multi-Agent Debate ({self.n_agents} agents, {n_rounds} rounds) ===",
            f"Final Answer: {final_prediction}",
            f"Valid votes: {dict(vote_counts)}",
            f"Empty votes: {empty_votes}",
            f"Valid vote ratio: {valid_vote_ratio:.2f}",
            f"Top vote ratio: {top_vote_ratio:.2f}",
            "",
        ]
        for ri, rd in enumerate(debate_rounds):
            summary_lines.append(f"--- Round {ri + 1} ---")
            for name, info in rd.items():
                summary_lines.append(f"  {name}: -> {info['pred']}")

        return {
            "prediction": final_prediction,
            "output": "\n".join(summary_lines),
            "metadata": {
                "n_agents": self.n_agents,
                "n_rounds": min(len(debate_rounds), n_rounds),
                "debate_rounds": debate_rounds,
                "vote_counts": dict(vote_counts),
                "empty_votes": empty_votes,
                "valid_vote_ratio": valid_vote_ratio,
                "top_vote_ratio": top_vote_ratio,
                "final_answers": {
                    cfg["name"]: self.task.extract_answer(answers.get(cfg["id"], ""))
                    for cfg in self.agent_configs
                },
            },
        }
