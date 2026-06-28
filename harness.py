"""
Harness: Unified experiment entry point for COT reasoning project.

Five-subsystem design:
- Instructions: Prompt templates loaded from prompts/
- Tools: Model interfaces (models/) and eval utilities (eval/)
- Environment: Task environments (tasks/)
- State: Experiment configuration and runtime tracking
- Feedback: Metrics computation and result logging
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from eval.metrics import compute_metrics
from models import OpenAIModel
try:
    from models import DebertaStepVerifier
except ImportError:
    DebertaStepVerifier = None
from retrieval import SimpleKeywordRetriever
from strategies import (
    BaseCOTStrategy,
    SelfConsistencyStrategy,
    StepAwareVerifierStrategy,
    RAGCOTStrategy,
    MultiAgentDebateStrategy,
    PrefixConsistencyStrategy,
    FewShotCOTStrategy,
    EvidenceCalibratedHarnessStrategy,
)
from tasks import AQuATask

# ---------------------------------------------------------------------------
# Global stdout/stderr encoding fix for Windows terminals that default to
# non-UTF-8 code pages. Must run before any print() or tqdm.write() to
# prevent UnicodeEncodeError from corrupting experiment samples.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        else:
            import io
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
    except Exception:
        pass  # If even this fails, fall through to _safe_write below


def load_strategy(strategy_name: str, model, task, **kwargs):
    """Load a strategy by name. Handles special dependencies like retriever."""
    registry = {
        "base_cot": BaseCOTStrategy,
        "self_consistency": SelfConsistencyStrategy,
        "step_verifier": StepAwareVerifierStrategy,
        "rag_cot": RAGCOTStrategy,
        "multi_agent_debate": MultiAgentDebateStrategy,
        "prefix_consistency": PrefixConsistencyStrategy,
        "few_shot_cot": FewShotCOTStrategy,
        "evidence_calibrated_harness": EvidenceCalibratedHarnessStrategy,
    }
    if strategy_name not in registry:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(registry.keys())}")

    # Inject retriever for RAG+COT
    if strategy_name == "rag_cot" and "retriever" not in kwargs:
        kwargs["retriever"] = SimpleKeywordRetriever(
            knowledge_path=kwargs.pop("knowledge_path", "data/knowledge_base.json"),
            top_k=kwargs.pop("top_k", 3),
        )

    return registry[strategy_name](model=model, task=task, **kwargs)


def load_task(task_name: str, **kwargs):
    """Load a task by name."""
    registry = {
        "aqua": AQuATask,
    }
    if task_name not in registry:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(registry.keys())}")
    return registry[task_name](**kwargs)


def _safe_write(msg: str):
    """Write to stdout safely on Windows with fallback for encoding errors."""
    try:
        tqdm.write(msg)
    except UnicodeEncodeError:
        # Fallback: encode with replacement for non-UTF-8 terminals
        tqdm.write(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))


def _is_fatal_api_error(exc: Exception) -> bool:
    """Return True for API errors that make continuing the run useless."""
    text = str(exc).lower()
    fatal_markers = [
        "401",
        "authentication",
        "invalid api key",
        "api key",
        "unauthorized",
    ]
    return any(marker in text for marker in fatal_markers)


def run_experiment(
    strategy_name: str,
    task_name: str,
    model_name: str,
    dataset_split: str = "test",
    n_samples: Optional[int] = None,
    output_dir: str = "experiments/runs",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **strategy_kwargs
) -> str:
    """
    Run a single experiment and save results.

    Returns:
        run_id string.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Starting Experiment {run_id} ===")
    print(f"Strategy: {strategy_name} | Task: {task_name} | Model: {model_name}")

    # 1. Environment — load task
    task = load_task(task_name)
    examples = task.load_data(split=dataset_split)
    if n_samples is not None:
        examples = examples[:n_samples]
    else:
        # Default limit to 50 samples for actual experiments
        examples = examples[:50]
    print(f"Loaded {len(examples)} examples from {task_name}/{dataset_split}")

    # 2. Tools — load model
    model = OpenAIModel(model_name=model_name, api_key=api_key, base_url=base_url)

    # 3. Instructions — load strategy
    strategy = load_strategy(strategy_name, model=model, task=task, **strategy_kwargs)

    # 4. State — run inference
    results: List[Dict[str, Any]] = []
    start_time = time.time()
    correct_so_far = 0

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name} | Samples: {len(examples)}")
    print(f"{'='*60}\n")

    for idx, example in enumerate(tqdm(examples, desc="Progress", unit="sample")):
        try:
            result = strategy.run(
                example,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(result)
            # Live accuracy update
            if result["prediction"].upper() == example.get("correct", "").upper():
                correct_so_far += 1
            elapsed_so_far = time.time() - start_time
            avg_time = elapsed_so_far / (idx + 1)
            eta = avg_time * (len(examples) - idx - 1)
            _safe_write(
                f"  [{idx+1}/{len(examples)}] "
                f"Pred={result['prediction']:<2} "
                f"Gold={example.get('correct', ''):<2} "
                f"LiveAcc={correct_so_far/(idx+1):.2%} "
                f"ETA={eta:.0f}s"
            )
        except Exception as e:
            _safe_write(f"  [{idx+1}/{len(examples)}] ERROR: {e}")
            results.append({
                "prediction": "",
                "output": f"ERROR: {e}",
                "metadata": {},
            })
            if _is_fatal_api_error(e):
                _safe_write(
                    "Fatal API authentication error detected. "
                    "Stopping this run early; please set a valid OPENAI_API_KEY or pass --api_key."
                )
                break

    elapsed = time.time() - start_time

    # 5. Feedback — compute metrics and save
    evaluated_examples = examples[:len(results)]
    metrics = compute_metrics(results, evaluated_examples)
    metrics["elapsed_time"] = elapsed
    metrics["throughput"] = len(examples) / elapsed if elapsed > 0 else 0.0

    run_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "strategy": strategy_name,
            "task": task_name,
            "model": model_name,
            "dataset_split": dataset_split,
            "n_samples": len(examples),
            "completed_samples": len(results),
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "strategy_info": strategy.get_strategy_info(),
        "task_info": task.get_task_info(),
        "model_info": model.get_model_info(),
        "metrics": metrics,
        "results": [
            {
                "prediction": r["prediction"],
                "correct": ex.get("correct", ""),
                "output": r["output"],
                "metadata": r.get("metadata", {}),
            }
            for r, ex in zip(results, evaluated_examples)
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    run_path = os.path.join(output_dir, f"{run_id}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2)

    print(f"\n=== Experiment {run_id} Complete ===")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"Time: {elapsed:.2f}s | Results saved to: {run_path}")

    return run_id


def main():
    parser = argparse.ArgumentParser(description="COT Experiment Harness")
    parser.add_argument("--strategy", type=str, default="base_cot", help="Strategy name")
    parser.add_argument("--dataset", type=str, default="aqua", help="Dataset/task name")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to run")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--output_dir", type=str, default="experiments/runs", help="Output directory")
    parser.add_argument("--api_key", type=str, default=None, help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.deepseek.com/v1",
        help="Base URL for the API endpoint (default: DeepSeek Official)",
    )
    # Strategy-specific parameters
    parser.add_argument("--n_paths", type=int, default=5, help="Number of reasoning paths per prompt (for self_consistency / step_verifier / prefix_consistency)")
    parser.add_argument("--n_prompts", type=int, default=3, help="Number of diverse prompts (for step_verifier)")
    parser.add_argument("--harness_n_prompts", type=int, default=1, help="Number of verifier prompts inside evidence_calibrated_harness")
    parser.add_argument("--harness_n_paths", type=int, default=3, help="Number of paths per adaptive stage inside evidence_calibrated_harness")
    parser.add_argument("--truncation_ratio", type=float, default=0.5, help="CoT truncation ratio for prefix regeneration (for prefix_consistency)")
    parser.add_argument("--regen_count", type=int, default=3, help="Number of regenerations per prefix (for prefix_consistency)")
    parser.add_argument("--min_paths", type=int, default=3, help="Minimum initial paths before early stopping (for prefix_consistency / evidence_calibrated_harness)")
    parser.add_argument("--early_stop_agreement", type=float, default=1.0, help="Initial path agreement threshold for early stopping (for prefix_consistency / evidence_calibrated_harness)")
    parser.add_argument("--min_regen", type=int, default=2, help="Minimum regenerations before early stopping (for prefix_consistency / evidence_calibrated_harness)")
    parser.add_argument("--target_consistency", type=float, default=1.0, help="Target prefix consistency for regeneration early stopping (for prefix_consistency / evidence_calibrated_harness)")
    parser.add_argument("--weight_fn", type=str, default="linear", help="Weight function for prefix consistency voting: linear/quadratic/cubic/unanimous")
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents (for multi_agent_debate)")
    parser.add_argument("--n_rounds", type=int, default=2, help="Number of debate rounds (for multi_agent_debate)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved docs (for rag_cot)")
    parser.add_argument("--rag_hops", type=int, default=2, help="Number of retrieval hops (for rag_cot)")
    parser.add_argument("--rag_context_docs", type=int, default=5, help="Max retrieved docs to place in the final context (for rag_cot)")
    parser.add_argument("--rag_no_planner", action="store_true", help="Disable the query-planning hop for rag_cot")
    parser.add_argument("--n_shots", type=int, default=5, help="Number of few-shot examples (for few_shot_cot)")
    parser.add_argument("--local_verifier", action="store_true", help="Use local DeBERTa verifier instead of LLM verifier")
    parser.add_argument("--verifier_model_path", type=str, default="data/checkpoint", help="Path to the local verifier model checkpoint")
    parser.add_argument("--base_threshold", type=float, default=0.80, help="Acceptance threshold after Base/Few-shot/RAG stages (for evidence_calibrated_harness)")
    parser.add_argument("--vote_threshold", type=float, default=0.70, help="Acceptance threshold after vote-based stages (for evidence_calibrated_harness)")
    parser.add_argument("--verifier_threshold", type=float, default=0.70, help="Acceptance threshold after verifier stage (for evidence_calibrated_harness)")
    parser.add_argument("--max_stage", type=str, default="debate", choices=["verifier", "debate"], help="Maximum escalation stage (for evidence_calibrated_harness)")
    parser.add_argument("--max_probes", type=int, default=3, help="Maximum evidence probes after accepted Base CoT (for evidence_calibrated_harness)")
    parser.add_argument("--probe_max_tokens", type=int, default=384, help="Max tokens for each evidence probe (for evidence_calibrated_harness)")
    parser.add_argument("--probe_temperature", type=float, default=0.0, help="Temperature for evidence probes (for evidence_calibrated_harness)")
    parser.add_argument("--evidence_accept_threshold", type=float, default=3.0, help="Support score required to accept Base after evidence calibration")
    parser.add_argument("--evidence_reject_threshold", type=float, default=0.0, help="Support score at or below this triggers escalation")

    args = parser.parse_args()

    # Strategy-specific kwargs
    strategy_kwargs = {}
    if args.strategy in ("step_verifier", "evidence_calibrated_harness") and args.local_verifier:
        print(f"Loading local verifier from {args.verifier_model_path} ...")
        strategy_kwargs["local_verifier"] = DebertaStepVerifier(
            model_path=args.verifier_model_path,
        )
        print(f"  → {strategy_kwargs['local_verifier']}")
    if args.strategy in ("self_consistency", "step_verifier", "prefix_consistency"):
        strategy_kwargs["n_paths"] = args.n_paths
    if args.strategy == "step_verifier":
        strategy_kwargs["n_prompts"] = args.n_prompts
    if args.strategy == "prefix_consistency":
        strategy_kwargs["truncation_ratio"] = args.truncation_ratio
        strategy_kwargs["regen_count"] = args.regen_count
        strategy_kwargs["min_paths"] = args.min_paths
        strategy_kwargs["early_stop_agreement"] = args.early_stop_agreement
        strategy_kwargs["min_regen"] = args.min_regen
        strategy_kwargs["target_consistency"] = args.target_consistency
        strategy_kwargs["weight_fn"] = args.weight_fn
    if args.strategy == "multi_agent_debate":
        strategy_kwargs["n_agents"] = args.n_agents
        strategy_kwargs["n_rounds"] = args.n_rounds
    if args.strategy == "rag_cot":
        strategy_kwargs["top_k"] = args.top_k
        strategy_kwargs["max_hops"] = args.rag_hops
        strategy_kwargs["max_context_docs"] = args.rag_context_docs
        strategy_kwargs["use_query_planner"] = not args.rag_no_planner
    if args.strategy == "few_shot_cot":
        strategy_kwargs["n_shots"] = args.n_shots
    if args.strategy == "evidence_calibrated_harness":
        strategy_kwargs["base_threshold"] = args.base_threshold
        strategy_kwargs["vote_threshold"] = args.vote_threshold
        strategy_kwargs["verifier_threshold"] = args.verifier_threshold
        strategy_kwargs["max_stage"] = args.max_stage
        strategy_kwargs["n_paths"] = args.harness_n_paths
        strategy_kwargs["regen_count"] = args.regen_count
        strategy_kwargs["min_paths"] = args.min_paths
        strategy_kwargs["early_stop_agreement"] = args.early_stop_agreement
        strategy_kwargs["min_regen"] = args.min_regen
        strategy_kwargs["target_consistency"] = args.target_consistency
        strategy_kwargs["n_prompts"] = args.harness_n_prompts
        strategy_kwargs["n_agents"] = args.n_agents
        strategy_kwargs["n_rounds"] = args.n_rounds
        strategy_kwargs["n_shots"] = args.n_shots
        strategy_kwargs["top_k"] = args.top_k
        strategy_kwargs["max_probes"] = args.max_probes
        strategy_kwargs["probe_max_tokens"] = args.probe_max_tokens
        strategy_kwargs["probe_temperature"] = args.probe_temperature
        strategy_kwargs["evidence_accept_threshold"] = args.evidence_accept_threshold
        strategy_kwargs["evidence_reject_threshold"] = args.evidence_reject_threshold

    run_experiment(
        strategy_name=args.strategy,
        task_name=args.dataset,
        model_name=args.model,
        dataset_split=args.split,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        base_url=args.base_url,
        **strategy_kwargs,
    )


if __name__ == "__main__":
    main()
