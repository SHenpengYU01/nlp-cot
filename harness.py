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

    elapsed = time.time() - start_time

    # 5. Feedback — compute metrics and save
    metrics = compute_metrics(results, examples)
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
            }
            for r, ex in zip(results, examples)
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
    parser.add_argument("--truncation_ratio", type=float, default=0.5, help="CoT truncation ratio for prefix regeneration (for prefix_consistency)")
    parser.add_argument("--regen_count", type=int, default=3, help="Number of regenerations per prefix (for prefix_consistency)")
    parser.add_argument("--weight_fn", type=str, default="linear", help="Weight function for prefix consistency voting: linear/quadratic/cubic/unanimous")
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents (for multi_agent_debate)")
    parser.add_argument("--n_rounds", type=int, default=2, help="Number of debate rounds (for multi_agent_debate)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved docs (for rag_cot)")
    parser.add_argument("--rag_hops", type=int, default=2, help="Number of retrieval hops (for rag_cot)")
    parser.add_argument("--rag_context_docs", type=int, default=5, help="Max retrieved docs to place in the final context (for rag_cot)")
    parser.add_argument("--rag_no_planner", action="store_true", help="Disable the query-planning hop for rag_cot")
    parser.add_argument("--n_shots", type=int, default=5, help="Number of few-shot examples (for few_shot_cot)")
    parser.add_argument("--local_verifier", action="store_true", help="Use local DeBERTa verifier instead of LLM verifier")
    parser.add_argument("--verifier_model_path", type=str, default="k1r1same/aqua-verifier", help="HuggingFace model name or local path to the verifier model")

    args = parser.parse_args()

    # Strategy-specific kwargs
    strategy_kwargs = {}
    if args.strategy == "step_verifier" and args.local_verifier:
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
