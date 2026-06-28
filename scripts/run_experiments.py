#!/usr/bin/env python3
"""
Run all 6 COT strategies on 100 AQuA samples.

Usage:
    python scripts/run_experiments.py          # 串行运行，默认 100 条（推荐，避免API限流）
    python scripts/run_experiments.py --n_samples 50   # 指定 50 条
    python scripts/run_experiments.py --parallel       # 并行运行（快但可能限流）
    python scripts/run_experiments.py --strategy base_cot  # 只跑单个策略

注意：
    - step_verifier 非常慢（约 5min/条），100 条约需 8~10 小时
    - 建议首次测试先用 --strategy base_cot --n_samples 5 验证连通性
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ==================== 配置 ====================
# 从环境变量读取 API Key；推送代码前请勿硬编码密钥
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-v4-flash"
N_SAMPLES = 100
OUTPUT_DIR = "experiments/runs"

# 策略列表：(名称, 额外参数)
EXPERIMENTS = [
    ("base_cot", []),
    ("rag_cot", ["--top_k", "3"]),
    ("self_consistency", ["--n_paths", "7"]),
    ("prefix_consistency", ["--n_paths", "3", "--truncation_ratio", "0.5", "--regen_count", "3", "--weight_fn", "linear"]),
    ("multi_agent_debate", ["--n_agents", "5", "--n_rounds", "3"]),
    ("step_verifier", ["--n_paths", "3"]),
    ("evidence_calibrated_harness", []),
]


def build_cmd(strategy_name, extra_args, n_samples=N_SAMPLES):
    """构建 harness.py 命令。"""
    return [
        sys.executable, "harness.py",
        "--strategy", strategy_name,
        "--dataset", "aqua",
        "--n_samples", str(n_samples),
        "--api_key", API_KEY,
        "--base_url", BASE_URL,
        "--model", MODEL,
        "--output_dir", OUTPUT_DIR,
    ] + extra_args


def run_single(strategy_name, extra_args, verbose=True, n_samples=N_SAMPLES):
    """运行单个策略实验，返回结果摘要。"""
    cmd = build_cmd(strategy_name, extra_args, n_samples=n_samples)
    if verbose:
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动: {strategy_name}")
        print(f"{'='*60}")
        print(f"命令: {' '.join(cmd)}")
        print()

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        elapsed = time.time() - start
        if verbose:
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        return {
            "strategy": strategy_name,
            "status": "success",
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        if verbose:
            print(f"❌ {strategy_name} 失败!")
            print(e.stdout)
            print(e.stderr)
        return {
            "strategy": strategy_name,
            "status": "failed",
            "elapsed": elapsed,
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


def run_all_serial(experiments, n_samples=N_SAMPLES):
    """串行运行所有实验。"""
    print(f"\n{'#'*60}")
    print("# COT 大规模实验 — 串行模式")
    print(f"# 样本数: {n_samples} | 模型: {MODEL}")
    print(f"# 预计总耗时: base(~1h) + rag(~20min) + sc(~1.5h) + pc(~3h) + mad(~4h) + sv(~10h)")
    print(f"{'#'*60}\n")

    results = []
    for name, args in experiments:
        # step_verifier 警告
        if name == "step_verifier":
            print(f"⚠️ 警告: step_verifier 非常慢（约 5min/条），{n_samples} 条约需 8~10 小时。")
            print("若不想等，可按 Ctrl+C 跳过，后续单独运行。")
        res = run_single(name, args)
        results.append(res)
    return results


def run_all_parallel(experiments, max_workers=3, n_samples=N_SAMPLES):
    """并行运行实验（可能触发API限流）。"""
    print(f"\n{'#'*60}")
    print("# COT 大规模实验 — 并行模式")
    print(f"# 样本数: {n_samples} | 模型: {MODEL} | 并发数: {max_workers}")
    print(f"# ⚠️ 注意: 并行可能触发 API 限流，导致失败或更慢")
    print(f"{'#'*60}\n")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single, name, args, False): (name, args)
            for name, args in experiments
        }
        for future in as_completed(futures):
            name, _ = futures[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"strategy": name, "status": "exception", "error": str(e)}
            results.append(res)
            status_icon = "✅" if res.get("status") == "success" else "❌"
            print(f"{status_icon} {name} 完成 | 耗时: {res.get('elapsed', 0):.1f}s")
    return results


def print_summary(results):
    """打印实验汇总。"""
    print(f"\n{'='*60}")
    print("实验完成汇总")
    print(f"{'='*60}")
    for r in results:
        status = r.get("status", "unknown")
        icon = "✅" if status == "success" else "❌"
        print(f"{icon} {r['strategy']:<22} | 状态: {status:<10} | 耗时: {r.get('elapsed', 0):.1f}s")

    # 自动调用 analyze
    print(f"\n{'='*60}")
    print("正在生成对比报告...")
    print(f"{'='*60}")
    analyze_cmd = [sys.executable, "-m", "eval.analyze", "--runs_dir", OUTPUT_DIR, "--latest", str(len(EXPERIMENTS))]
    try:
        subprocess.run(analyze_cmd, check=True)
    except Exception as e:
        print(f"分析失败: {e}")
        print(f"可手动运行: {' '.join(analyze_cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Run COT experiments")
    parser.add_argument("--parallel", action="store_true", help="并行运行（可能限流）")
    parser.add_argument("--max_workers", type=int, default=3, help="并行最大进程数")
    parser.add_argument("--strategy", type=str, default=None, help="只运行指定策略，包括 base_cot, self_consistency, prefix_consistency, step_verifier, rag_cot, multi_agent_debate, evidence_calibrated_harness")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES, help="测试样本数（默认 100）")
    args = parser.parse_args()

    # 如果只跑单个策略
    if args.strategy:
        found = [(n, a) for n, a in EXPERIMENTS if n == args.strategy]
        if not found:
            print(f"未知策略: {args.strategy}")
            print(f"可用策略: {', '.join(n for n, _ in EXPERIMENTS)}")
            sys.exit(1)
        res = run_single(found[0][0], found[0][1], n_samples=args.n_samples)
        print_summary([res])
        return

    # 跑全部策略
    if args.parallel:
        results = run_all_parallel(EXPERIMENTS, max_workers=args.max_workers, n_samples=args.n_samples)
    else:
        results = run_all_serial(EXPERIMENTS, n_samples=args.n_samples)

    print_summary(results)


if __name__ == "__main__":
    main()
