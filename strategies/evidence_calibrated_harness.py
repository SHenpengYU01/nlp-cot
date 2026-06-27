"""Evidence-Calibrated Adaptive CoT Harness.

This strategy keeps the original Harness-Guided CoT idea, but adds an
evidence calibration layer for the most common remaining error mode:
confident wrong Base CoT answers.

Base CoT that passes surface lint is not accepted immediately. The harness
selects a small set of heterogeneous probes, aggregates their evidence, and
only accepts the Base answer when independent evidence supports it.
"""
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from harness_control import (
    AdaptiveRouter,
    CoTLinter,
    EvidenceAggregator,
    FailureAnalyzer,
    ProbeSelector,
    RiskClassifier,
)
from retrieval import SimpleKeywordRetriever

from .base import BaseStrategy
from .base_cot import BaseCOTStrategy
from .few_shot_cot import FewShotCOTStrategy
from .multi_agent_debate import MultiAgentDebateStrategy
from .prefix_consistency import PrefixConsistencyStrategy
from .rag_cot import RAGCOTStrategy
from .step_verifier import StepAwareVerifierStrategy


class EvidenceCalibratedHarnessStrategy(BaseStrategy):
    """Adaptive CoT harness with risk-aware heterogeneous evidence probes."""

    def harness_subsystems(self) -> Dict[str, bool]:
        return {
            "instructions": True,
            "tools": True,
            "environment": True,
            "state": True,
            "feedback": True,
        }

    def __init__(
        self,
        model,
        task,
        base_threshold: float = 0.80,
        vote_threshold: float = 0.70,
        verifier_threshold: float = 0.70,
        evidence_accept_threshold: float = 3.0,
        evidence_reject_threshold: float = 0.0,
        max_stage: str = "debate",
        max_probes: int = 3,
        probe_max_tokens: int = 384,
        probe_temperature: float = 0.0,
        retriever=None,
        top_k: int = 3,
        n_paths: int = 3,
        regen_count: int = 2,
        min_paths: int = 3,
        early_stop_agreement: float = 1.0,
        min_regen: int = 2,
        target_consistency: float = 1.0,
        n_prompts: int = 1,
        n_agents: int = 3,
        n_rounds: int = 2,
        n_shots: int = 3,
        local_verifier=None,
        **kwargs,
    ):
        super().__init__(
            name="evidence_calibrated_harness",
            model=model,
            task=task,
            base_threshold=base_threshold,
            vote_threshold=vote_threshold,
            verifier_threshold=verifier_threshold,
            evidence_accept_threshold=evidence_accept_threshold,
            evidence_reject_threshold=evidence_reject_threshold,
            max_stage=max_stage,
            max_probes=max_probes,
            **kwargs,
        )
        self.linter = CoTLinter(answer_extractor=task.extract_answer)
        self.analyzer = FailureAnalyzer()
        self.router = AdaptiveRouter(
            base_threshold=base_threshold,
            vote_threshold=vote_threshold,
            verifier_threshold=verifier_threshold,
            max_stage=max_stage,
        )
        self.risk_classifier = RiskClassifier()
        self.probe_selector = ProbeSelector()
        self.evidence_aggregator = EvidenceAggregator(
            accept_threshold=evidence_accept_threshold,
            reject_threshold=evidence_reject_threshold,
        )
        self.max_probes = max_probes
        self.probe_max_tokens = probe_max_tokens
        self.probe_temperature = probe_temperature

        self.base = BaseCOTStrategy(model=model, task=task)
        self.few_shot = FewShotCOTStrategy(model=model, task=task, n_shots=n_shots)
        self.prefix = PrefixConsistencyStrategy(
            model=model,
            task=task,
            n_paths=n_paths,
            regen_count=regen_count,
            min_paths=min_paths,
            early_stop_agreement=early_stop_agreement,
            min_regen=min_regen,
            target_consistency=target_consistency,
        )
        self.verifier = StepAwareVerifierStrategy(
            model=model,
            task=task,
            n_paths=n_paths,
            n_prompts=n_prompts,
            local_verifier=local_verifier,
        )
        self.debate = MultiAgentDebateStrategy(
            model=model,
            task=task,
            n_agents=n_agents,
            n_rounds=n_rounds,
        )
        self.retriever = retriever or SimpleKeywordRetriever(
            knowledge_path=kwargs.get("knowledge_path", "data/knowledge_base.json")
        )
        self.rag = RAGCOTStrategy(model=model, task=task, retriever=self.retriever, top_k=top_k)

    def run(self, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        activated: List[str] = []

        print("    [Evidence] Running Base CoT...")
        base_result = self.base.run(example, **kwargs)
        activated.append("base_cot")
        base_lint = self.linter.analyze(base_result.get("output", ""))
        base_analysis = self.analyzer.analyze_initial(base_lint)
        trace.append(self._trace_entry("base_cot", base_result, base_analysis, base_lint))
        self._log_stage_result("Base CoT", base_result, base_analysis)

        if not self.router.accept_base(base_analysis):
            return self._repair_visible_failure(example, base_analysis, trace, activated, **kwargs)

        evidence_result = self._calibrate_base_answer(example, base_result, trace, **kwargs)
        evidence_decision = evidence_result["aggregation"].get("decision", "uncertain")
        if evidence_decision == "accept":
            print("    [Evidence] Accepted Base CoT after evidence calibration.")
            return self._finalize(
                base_result,
                trace,
                activated + ["evidence_calibration"],
                "accepted_evidence_calibrated_base",
                evidence_result,
            )

        print(
            "    [Evidence] Base CoT not sufficiently supported; "
            f"decision={evidence_decision}. Escalating to prefix_consistency..."
        )
        prefix_result = self.prefix.run(example, **kwargs)
        activated.append("prefix_consistency")
        vote_analysis = self.analyzer.analyze_vote(prefix_result)
        trace.append(self._trace_entry("prefix_consistency", prefix_result, vote_analysis))
        self._log_stage_result("prefix_consistency", prefix_result, vote_analysis)

        if self.router.accept_vote(vote_analysis):
            final_result, arbiter_result = self._apply_final_arbiter(
                candidate_result=prefix_result,
                base_result=base_result,
                evidence_result=evidence_result,
                prefix_result=prefix_result,
            )
            return self._finalize(
                final_result,
                trace,
                activated,
                "accepted_vote_after_evidence_conflict",
                evidence_result,
                arbiter_result,
            )

        print("    [Evidence] Prefix vote still weak; escalating to step_verifier...")
        verifier_result = self.verifier.run(example, **kwargs)
        activated.append("step_verifier")
        verifier_analysis = self.analyzer.analyze_verifier(verifier_result)
        trace.append(self._trace_entry("step_verifier", verifier_result, verifier_analysis))
        self._log_stage_result("step_verifier", verifier_result, verifier_analysis)

        if self.router.accept_verifier(verifier_analysis):
            final_result, arbiter_result = self._apply_final_arbiter(
                candidate_result=verifier_result,
                base_result=base_result,
                evidence_result=evidence_result,
                prefix_result=prefix_result,
                verifier_result=verifier_result,
            )
            return self._finalize(
                final_result,
                trace,
                activated,
                "accepted_verifier_after_evidence_conflict",
                evidence_result,
                arbiter_result,
            )

        if self.router.should_use_debate():
            print("    [Evidence] Verifier still weak; escalating to multi_agent_debate...")
            debate_result = self.debate.run(example, **kwargs)
            activated.append("multi_agent_debate")
            debate_analysis = self.analyzer.analyze_debate(debate_result)
            debate_lint = self.linter.analyze(debate_result.get("output", ""))
            trace.append(self._trace_entry("multi_agent_debate", debate_result, debate_analysis, debate_lint))
            self._log_stage_result("multi_agent_debate", debate_result, debate_analysis)
            final_result, arbiter_result = self._apply_final_arbiter(
                candidate_result=debate_result,
                base_result=base_result,
                evidence_result=evidence_result,
                prefix_result=prefix_result,
                verifier_result=verifier_result,
                debate_result=debate_result,
            )
            return self._finalize(
                final_result,
                trace,
                activated,
                "fallback_debate_after_evidence_conflict",
                evidence_result,
                arbiter_result,
            )

        final_result, arbiter_result = self._apply_final_arbiter(
            candidate_result=verifier_result,
            base_result=base_result,
            evidence_result=evidence_result,
            prefix_result=prefix_result,
            verifier_result=verifier_result,
        )
        return self._finalize(
            final_result,
            trace,
            activated,
            "accepted_max_stage_after_evidence_conflict",
            evidence_result,
            arbiter_result,
        )

    def _repair_visible_failure(
        self,
        example: Dict[str, Any],
        base_analysis: Dict[str, Any],
        trace: List[Dict[str, Any]],
        activated: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        question = example.get("question", "")
        repair_stage = self.router.next_after_base(
            base_analysis.get("failure_type", "low_quality"),
            knowledge_needed=self.analyzer.detect_knowledge_need(question),
        )
        print(f"    [Evidence] Visible failure detected; escalating to {repair_stage}...")
        if repair_stage == "rag_cot":
            repair_result = self.rag.run(example, **kwargs)
        elif repair_stage == "few_shot_cot":
            repair_result = self.few_shot.run(example, **kwargs)
        else:
            repair_result = self.prefix.run(example, **kwargs)
        activated.append(repair_stage)

        repair_lint = self.linter.analyze(repair_result.get("output", ""))
        repair_analysis = self.analyzer.analyze_initial(repair_lint)
        trace.append(self._trace_entry(repair_stage, repair_result, repair_analysis, repair_lint))
        self._log_stage_result(repair_stage, repair_result, repair_analysis)

        if self.router.accept_base(repair_analysis):
            return self._finalize(repair_result, trace, activated, f"accepted_{repair_stage}")

        prefix_result = repair_result
        if repair_stage != "prefix_consistency":
            print("    [Evidence] Repair still weak; escalating to prefix_consistency...")
            prefix_result = self.prefix.run(example, **kwargs)
            activated.append("prefix_consistency")
            vote_analysis = self.analyzer.analyze_vote(prefix_result)
            trace.append(self._trace_entry("prefix_consistency", prefix_result, vote_analysis))
        else:
            vote_analysis = self.analyzer.analyze_vote(prefix_result)

        self._log_stage_result("prefix_consistency", prefix_result, vote_analysis)
        if self.router.accept_vote(vote_analysis):
            return self._finalize(prefix_result, trace, activated, "accepted_vote")

        print("    [Evidence] Prefix vote weak; escalating to step_verifier...")
        verifier_result = self.verifier.run(example, **kwargs)
        activated.append("step_verifier")
        verifier_analysis = self.analyzer.analyze_verifier(verifier_result)
        trace.append(self._trace_entry("step_verifier", verifier_result, verifier_analysis))
        self._log_stage_result("step_verifier", verifier_result, verifier_analysis)
        if self.router.accept_verifier(verifier_analysis):
            return self._finalize(verifier_result, trace, activated, "accepted_verifier")

        if self.router.should_use_debate():
            print("    [Evidence] Escalating to multi_agent_debate...")
            debate_result = self.debate.run(example, **kwargs)
            activated.append("multi_agent_debate")
            debate_analysis = self.analyzer.analyze_debate(debate_result)
            debate_lint = self.linter.analyze(debate_result.get("output", ""))
            trace.append(self._trace_entry("multi_agent_debate", debate_result, debate_analysis, debate_lint))
            self._log_stage_result("multi_agent_debate", debate_result, debate_analysis)
            return self._finalize(debate_result, trace, activated, "fallback_debate")

        return self._finalize(verifier_result, trace, activated, "accepted_max_stage")

    def _calibrate_base_answer(
        self,
        example: Dict[str, Any],
        base_result: Dict[str, Any],
        trace: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        question = example.get("question", "")
        options = example.get("options", [])
        risk_types = self.risk_classifier.classify(question, options, base_result.get("output", ""))
        probes = self.probe_selector.select(risk_types, max_probes=self.max_probes)
        print(f"    [Evidence] Risk types: {risk_types}")
        print(f"    [Evidence] Selected probes: {probes}")

        probe_results: List[Dict[str, Any]] = []
        for probe_type in probes:
            probe_result = self._run_probe(probe_type, example, base_result, **kwargs)
            probe_results.append(probe_result)
            print(
                f"    [Evidence] Probe {probe_type}: stance={probe_result['stance']} "
                f"suggested={probe_result.get('suggested_answer') or 'EMPTY'}"
            )

        aggregation = self.evidence_aggregator.aggregate(
            base_result.get("prediction", ""),
            probe_results,
        )
        print(
            f"    [Evidence] Aggregation -> score={aggregation['support_score']:.2f} "
            f"decision={aggregation['decision']}"
        )

        evidence_result = {
            "risk_types": risk_types,
            "selected_probes": probes,
            "probe_results": probe_results,
            "aggregation": aggregation,
        }
        trace.append({
            "stage": "evidence_calibration",
            "prediction": base_result.get("prediction", ""),
            "analysis": {
                "reliability": max(0.0, min(1.0, aggregation["support_score"] / 4.0)),
                "failure_type": "ok" if aggregation["decision"] == "accept" else "confident_wrong_risk",
                "risk_types": risk_types,
                "support_score": aggregation["support_score"],
                "decision": aggregation["decision"],
            },
            "probe_results": probe_results,
            "aggregation": aggregation,
        })
        return evidence_result

    def _run_probe(
        self,
        probe_type: str,
        example: Dict[str, Any],
        base_result: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        prompt = self._build_probe_prompt(probe_type, example, base_result)
        outputs = self.model.generate(
            prompt,
            temperature=kwargs.get("probe_temperature", self.probe_temperature),
            max_tokens=kwargs.get("probe_max_tokens", self.probe_max_tokens),
            n=1,
        )
        raw_output = outputs[0] if outputs else ""
        stance = self._extract_stance(raw_output)
        suggested = self.task.extract_answer(raw_output)
        issue = self._extract_issue(raw_output)

        return {
            "probe_type": probe_type,
            "stance": stance,
            "suggested_answer": suggested,
            "issue": issue,
            "output": raw_output,
            "prompt": prompt,
        }

    def _build_probe_prompt(
        self,
        probe_type: str,
        example: Dict[str, Any],
        base_result: Dict[str, Any],
    ) -> str:
        question = example.get("question", "")
        options = " ".join(example.get("options", []))
        base_answer = base_result.get("prediction", "")
        base_output = base_result.get("output", "")

        instructions = {
            "alternative_cot": (
                "Solve the problem again using a different reasoning path. "
                "Do not copy the previous solution. You should not assume the proposed answer is correct."
            ),
            "back_substitution": (
                f"Try to disprove the proposed answer {base_answer}. Substitute it back "
                "into the original problem and check whether every condition is satisfied. "
                "If another option satisfies the conditions better, report that option."
            ),
            "option_elimination": (
                "Check the options A-E one by one. Eliminate options that violate the "
                "problem conditions, then identify the best supported option. Do not favor "
                "the proposed answer unless it passes the same checks as every other option."
            ),
            "equation_audit": (
                "Do not solve from scratch. Audit whether the equations and variable "
                "relationships in the proposed reasoning correctly represent the problem."
            ),
            "arithmetic_audit": (
                "Independently recompute the key arithmetic, percentages, ratios, rounding, "
                "and option matching. Do not rely on the proposed reasoning. If the computed "
                "value points to a different option, report that option."
            ),
            "constraint_check": (
                "Do not solve from scratch. Check whether the proposed reasoning uses all "
                "conditions and respects every constraint in the problem."
            ),
        }
        task_instruction = instructions.get(probe_type, instructions["alternative_cot"])

        reasoning_visible = probe_type in {"equation_audit", "constraint_check"}
        reasoning_block = (
            "Proposed reasoning, shown only for audit of modeling/constraints:\n"
            f"{base_output}\n\n"
            if reasoning_visible
            else (
                "The proposed reasoning is intentionally hidden for this blind probe. "
                "Judge from the problem, options, and proposed answer only.\n\n"
            )
        )

        return (
            "You are an evidence probe inside a reasoning harness.\n"
            "Your job is to evaluate whether the proposed answer is supported.\n\n"
            f"Question: {question}\n"
            f"Options: {options}\n\n"
            f"Proposed answer: {base_answer}\n"
            f"{reasoning_block}"
            f"Probe task: {task_instruction}\n\n"
            "Important rules:\n"
            "- Be skeptical of the proposed answer.\n"
            "- SUPPORT means the proposed answer itself is supported.\n"
            "- If your suggested answer is different from the proposed answer, Evidence must be CONTRADICT.\n"
            "- If you cannot verify the proposed answer, use UNCERTAIN rather than SUPPORT.\n\n"
            "Return a compact audit with exactly these final fields:\n"
            "Evidence: SUPPORT / CONTRADICT / UNCERTAIN\n"
            "Suggested Answer: A/B/C/D/E or UNKNOWN\n"
            "Issue: one short sentence describing the main reason\n"
        )

    def _extract_stance(self, text: str) -> str:
        match = re.search(r"(?im)^\s*Evidence\s*:\s*(SUPPORT|CONTRADICT|UNCERTAIN)\b", text or "")
        if match:
            return match.group(1).lower()
        lowered = (text or "").lower()
        if "contradict" in lowered or "does not satisfy" in lowered or "incorrect" in lowered:
            return "contradict"
        if "support" in lowered or "satisfies" in lowered or "correct" in lowered:
            return "support"
        return "uncertain"

    def _extract_issue(self, text: str) -> str:
        match = re.search(r"(?im)^\s*Issue\s*:\s*(.+?)\s*$", text or "")
        if match:
            return match.group(1).strip()
        return (text or "").strip().splitlines()[-1][:200] if (text or "").strip() else ""

    def _apply_final_arbiter(
        self,
        candidate_result: Dict[str, Any],
        base_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        prefix_result: Optional[Dict[str, Any]] = None,
        verifier_result: Optional[Dict[str, Any]] = None,
        debate_result: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Choose a final answer from all stage-level evidence.

        This arbiter is only used after evidence calibration has already found
        a conflict. In that case, trusting the last stage blindly can discard
        earlier evidence such as arithmetic audits or prefix regenerations.
        """
        arbiter_result = self._final_arbiter(
            base_result=base_result,
            evidence_result=evidence_result,
            prefix_result=prefix_result,
            verifier_result=verifier_result,
            debate_result=debate_result,
            candidate_result=candidate_result,
        )
        final_answer = arbiter_result.get("final_answer", "") or candidate_result.get("prediction", "")
        if final_answer == candidate_result.get("prediction", ""):
            return candidate_result, arbiter_result

        print(
            f"    [Evidence-Arbiter] Override final answer: "
            f"{candidate_result.get('prediction', '')} -> {final_answer}"
        )
        patched = dict(candidate_result)
        patched["prediction"] = final_answer
        patched["output"] = (
            "=== Evidence-Aware Final Arbiter Override ===\n"
            f"Original candidate: {candidate_result.get('prediction', '')}\n"
            f"Arbiter final answer: {final_answer}\n"
            f"Arbiter scores: {arbiter_result.get('scores', {})}\n"
            f"Reason: {arbiter_result.get('reason', '')}\n\n"
            f"{candidate_result.get('output', '')}"
        )
        metadata = dict(candidate_result.get("metadata", {}))
        metadata["final_arbiter"] = arbiter_result
        patched["metadata"] = metadata
        return patched, arbiter_result

    def _final_arbiter(
        self,
        base_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        candidate_result: Dict[str, Any],
        prefix_result: Optional[Dict[str, Any]] = None,
        verifier_result: Optional[Dict[str, Any]] = None,
        debate_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scores: Counter = Counter()
        reasons: Dict[str, List[str]] = {a: [] for a in ["A", "B", "C", "D", "E"]}
        base_answer = (base_result.get("prediction") or "").upper()
        candidate_answer = (candidate_result.get("prediction") or "").upper()

        def add(answer: str, weight: float, reason: str) -> None:
            answer = (answer or "").upper()
            if answer in reasons:
                scores[answer] += weight
                reasons[answer].append(f"{reason} ({weight:+.2f})")

        # Candidate stage still matters, but it should not dominate when the
        # evidence layer has already detected a conflict.
        add(candidate_answer, 1.0, "candidate final stage")

        aggregation = evidence_result.get("aggregation", {})
        critical_errors = aggregation.get("critical_errors", [])
        evidence_conflict = aggregation.get("decision") != "accept" or bool(critical_errors)

        # Probe evidence: audit probes that contradict Base are strong signals.
        for probe in evidence_result.get("probe_results", []):
            suggested = (probe.get("suggested_answer") or "").upper()
            stance = probe.get("stance", "uncertain")
            probe_type = probe.get("probe_type", "")
            if suggested not in reasons:
                continue

            if suggested != base_answer:
                add(suggested, 3.0, f"{probe_type} suggests non-base answer")
                if base_answer:
                    add(base_answer, -2.0, f"{probe_type} contradicts base")
            elif stance == "support":
                add(suggested, 1.0, f"{probe_type} supports base")
            elif stance == "uncertain":
                add(suggested, 0.25, f"{probe_type} weak/uncertain support")

        # Prefix consistency: initial votes are weak; regeneration predictions
        # are stronger because they test whether the path remains stable.
        if prefix_result:
            meta = prefix_result.get("metadata", {})
            for pred in meta.get("all_predictions", []):
                add(pred, 0.4, "prefix initial path")
            for pred, weight in (meta.get("weighted_votes", {}) or {}).items():
                add(pred, float(weight), "prefix weighted vote")
            for regen_group in meta.get("regen_predictions", []):
                for pred in regen_group:
                    add(pred, 1.0, "prefix regeneration")

        # Step verifier: use path scores, but cap each path's effect so a
        # verifier that rewards plausible-looking wrong reasoning cannot erase
        # earlier arithmetic/option evidence by itself.
        if verifier_result:
            meta = verifier_result.get("metadata", {})
            for pred, score in zip(meta.get("all_predictions", []), meta.get("path_scores", [])):
                try:
                    weight = min(float(score) / 10.0, 1.0)
                except (TypeError, ValueError):
                    weight = 0.5
                add(pred, weight, "step verifier path")

        # Debate votes are usually late-stage evidence, but still aggregate
        # them instead of blindly trusting the summarized final answer.
        if debate_result:
            meta = debate_result.get("metadata", {})
            for pred, count in (meta.get("vote_counts", {}) or {}).items():
                add(pred, float(count), "debate vote")

        # Penalize the original Base answer when the evidence layer found a
        # critical contradiction. This is the key fix for confident-wrong cases.
        if evidence_conflict and base_answer:
            add(base_answer, -1.0, "evidence conflict penalty against base")

        ranked = scores.most_common()
        final_answer = ranked[0][0] if ranked else candidate_answer
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1] if ranked else 0.0

        # Avoid over-aggressive overrides on tiny margins.
        if final_answer != candidate_answer and margin < 1.0:
            final_answer = candidate_answer
            reason = "kept candidate because arbiter margin was too small"
        else:
            reason = "selected highest aggregated evidence score"

        return {
            "final_answer": final_answer,
            "candidate_answer": candidate_answer,
            "base_answer": base_answer,
            "scores": dict(scores),
            "ranked": ranked,
            "margin": margin,
            "reason": reason,
            "answer_reasons": reasons,
            "evidence_conflict": evidence_conflict,
        }

    def _trace_entry(
        self,
        stage: str,
        result: Dict[str, Any],
        analysis: Dict[str, Any],
        lint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "stage": stage,
            "prediction": result.get("prediction", ""),
            "analysis": analysis,
            "output_length": len(result.get("output", "") or ""),
            "output_preview": (result.get("output", "") or "")[:500],
        }
        if lint is not None:
            entry["lint"] = lint
        return entry

    def _log_stage_result(
        self,
        stage_name: str,
        result: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> None:
        prediction = result.get("prediction", "") or "EMPTY"
        reliability = analysis.get("reliability", 0.0)
        failure_type = analysis.get("failure_type", "ok")
        print(
            f"    [Evidence] {stage_name} -> pred={prediction} "
            f"reliability={reliability:.2f} failure={failure_type}"
        )

    def _finalize(
        self,
        result: Dict[str, Any],
        trace: List[Dict[str, Any]],
        activated: List[str],
        stop_reason: str,
        evidence_result: Optional[Dict[str, Any]] = None,
        arbiter_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = [
            "=== Evidence-Calibrated Adaptive CoT Harness ===",
            f"Final stage: {activated[-1] if activated else 'unknown'}",
            f"Stop reason: {stop_reason}",
            f"Activated strategies: {', '.join(activated)}",
            f"Final Answer: {result.get('prediction', '')}",
            "",
            "--- Harness trace ---",
        ]
        for item in trace:
            analysis = item.get("analysis", {})
            summary.append(
                f"{item['stage']}: pred={item.get('prediction', '')} "
                f"reliability={analysis.get('reliability', 0.0):.2f} "
                f"failure={analysis.get('failure_type', 'ok')}"
            )

        if evidence_result:
            aggregation = evidence_result.get("aggregation", {})
            summary.extend([
                "",
                "--- Evidence calibration ---",
                f"Risk types: {evidence_result.get('risk_types', [])}",
                f"Selected probes: {evidence_result.get('selected_probes', [])}",
                (
                    f"Evidence score: {aggregation.get('support_score', 0.0):.2f} "
                    f"decision={aggregation.get('decision', '')}"
                ),
            ])
            for probe in evidence_result.get("probe_results", []):
                summary.append(
                    f"{probe['probe_type']}: stance={probe['stance']} "
                    f"suggested={probe.get('suggested_answer') or 'EMPTY'} "
                    f"issue={probe.get('issue', '')}"
                )

        if arbiter_result:
            summary.extend([
                "",
                "--- Evidence-aware final arbiter ---",
                f"Candidate answer: {arbiter_result.get('candidate_answer', '')}",
                f"Final answer: {arbiter_result.get('final_answer', '')}",
                f"Scores: {arbiter_result.get('scores', {})}",
                f"Reason: {arbiter_result.get('reason', '')}",
            ])

        summary.extend(["", "--- Final strategy output ---", result.get("output", "")])

        metadata = dict(result.get("metadata", {}))
        metadata.update({
            "harness_trace": trace,
            "activated_strategies": activated,
            "final_stage": activated[-1] if activated else "",
            "stop_reason": stop_reason,
        })
        if evidence_result:
            metadata["evidence_calibration"] = evidence_result
        if arbiter_result:
            metadata["final_arbiter"] = arbiter_result

        return {
            "prediction": result.get("prediction", ""),
            "output": "\n".join(summary),
            "metadata": metadata,
        }
