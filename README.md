# COT 思维链推理实验项目

> 基于大语言模型的思维链推理（Chain-of-Thought Reasoning）进阶探索，借鉴 Harness Engineering 设计思想系统化实现与管理 CoT 策略。

---

## 📋 项目简介

本项目基于 OpenAI-compatible API（DeepSeek 官方 / deepseek-v4-flash），在 AQuA（Algebraic Word Problems）数据集上实现并对比 7 种 CoT 推理策略：

1. **Base COT** — 基础思维链推理
2. **Self-Consistency** — 多路径采样 + 路径质量加权投票
3. **Prefix Consistency** — 截断再生可靠性加权投票
4. **Step-Aware Verifier** — 步骤级验证 + 本地 DeBERTa verifier（支持 LLM 和本地双模式）
5. **Few-Shot CoT** — 随机采样示例的少样本思维链
6. **RAG + COT** — 检索增强思维链
7. **Multi-Agent Debate** — 多 Agent 协作辩论推理

**核心设计**：借鉴 [Harness Engineering](https://github.com/walkinglabs/learn-harness-engineering) 的五子系统思想（Instructions / Tools / Environment / State / Feedback），将每种 CoT 策略映射到 Harness 子系统，系统化地设计、实现与评估推理框架。

---

## 🏗️ 项目结构

```
.
├── data/
│   ├── AQuA/                   # 🔗 git submodule → deepmind/AQuA
│   ├── checkpoint/             # 本地 DeBERTa verifier 权重（config.json + model.safetensors）
│   └── knowledge_base.json     # RAG 检索知识库
├── prompts/                    # CoT 策略 Prompt 模板
│   ├── base_cot.txt
│   ├── few_shot_cot.txt        # Few-Shot CoT 模板（{few_shot_examples}）
│   ├── rag_cot.txt
│   └── step_verifier.txt
├── strategies/                 # 7 种 CoT 策略实现
│   ├── base.py                 # 策略基类（含 Harness 子系统声明）
│   ├── base_cot.py
│   ├── self_consistency.py
│   ├── prefix_consistency.py   # 截断再生可靠性加权投票
│   ├── step_verifier.py        # 步骤级验证 + 本地 DeBERTa verifier
│   ├── few_shot_cot.py         # Few-Shot CoT（随机采样示例 + LLM 生成推理链）
│   ├── rag_cot.py
│   └── multi_agent_debate.py
├── models/                     # LLM 接口封装 + 本地验证器
│   ├── base.py
│   ├── openai_api.py           # OpenAI-compatible API 封装
│   └── deberta_verifier.py     # DeBERTa-v3 步验证器
├── deberta_model.py            # 自定义 DeBERTaV2ForTokenClassification
├── eval/                       # 评估指标与分析工具
│   ├── metrics.py              # 准确率、推理步数、Token 估算
│   └── analyze.py              # 多实验对比分析
├── experiments/                # 实验记录与报告
│   ├── runs/                   # 原始运行结果（JSON）
│   ├── report_small_sample_20260610.md
│   └── report_50_sample_20260611.md
├── retrieval/                  # 检索模块
│   └── simple_retriever.py     # 基于关键词的 TF-IDF 检索
├── scripts/                    # 运行脚本
│   ├── init.sh                 # 环境验证脚本
│   ├── run_all.sh              # 批量测试 Bash 脚本
│   └── run_experiments.py      # 批量测试 Python 脚本
├── tests/                      # 功能验证测试
│   ├── verify_feat004.py
│   ├── verify_feat005.py
│   ├── verify_feat006.py
│   ├── verify_feat007.py
│   ├── verify_feat008.py
│   ├── verify_feat009.py
│   ├── verify_feat010.py
│   └── verify_feat011.py
├── docs/                       # 项目文档
│   ├── CLAUDE.md               # 开发规范与指令
│   ├── progress.md             # 开发进度日志
│   ├── session-handoff.md      # 多会话交接记录
│   └── 选题说明.md
├── harness.py                  # 🚀 实验管理主入口
├── harness_report.py           # Harness 子系统覆盖矩阵报告
├── requirements.txt
└── feature_list.json           # 功能状态追踪
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
# 带 submodule 克隆（推荐）
git clone --recurse-submodules git@github.com:SHenpengYU01/nlp-cot.git
cd nlp-cot

# 如果已克隆但忘了 --recurse-submodules
git submodule update --init --recursive
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖：`openai>=1.0.0`、`tqdm`、`numpy`

### 3. 环境验证

```bash
bash scripts/init.sh
```

验证内容包括：Python 环境、目录结构、核心文件完整性、模块导入、各策略干跑测试、功能状态列表。

### 4. 配置 API Key

```bash
# Linux / macOS / Git Bash
export OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"
```

---

## 🧪 运行实验

### 单策略快速测试

```bash
# 基础 COT（默认 5 条样本）
python harness.py --strategy base_cot --dataset aqua

# RAG + COT（两跳检索 + query planning）
python harness.py --strategy rag_cot --dataset aqua --top_k 3 --rag_hops 2

# Self-Consistency（7 条推理路径 + 质量加权投票 + 提前停止）
python harness.py --strategy self_consistency --dataset aqua --n_paths 7

# Prefix Consistency（3 条路径 + 截断 50% 再生 3 次 + 加权投票）
python harness.py --strategy prefix_consistency --dataset aqua --n_paths 3 --truncation_ratio 0.5 --regen_count 3

# Multi-Agent Debate（5 个 Agent × 3 轮辩论 + 交叉评审 + 收敛检测）
python harness.py --strategy multi_agent_debate --dataset aqua --n_agents 5 --n_rounds 3

# Step-Aware Verifier + 本地 DeBERTa（3 prompts × 5 路径）
python harness.py --strategy step_verifier --dataset aqua --n_prompts 3 --n_paths 5 --local_verifier

# 纯 LLM Verifier（无本地模型）
python harness.py --strategy step_verifier --dataset aqua --n_paths 3

# Few-Shot CoT（5 个随机示例）
python harness.py --strategy few_shot_cot --dataset aqua --n_shots 5
```

### 批量测试（50 条样本）

```bash
# 方式一：Python 脚本（串行，推荐）
python scripts/run_experiments.py

# 只跑单个策略
python scripts/run_experiments.py --strategy base_cot

# 并行运行（可能触发 API 限流）
python scripts/run_experiments.py --parallel --max_workers 3

# 方式二：Bash 脚本
bash scripts/run_all.sh
```

### 自定义参数

```bash
python harness.py \
  --strategy base_cot \
  --dataset aqua \
  --n_samples 100 \
  --model deepseek-v4-flash \
  --api_key "$OPENAI_API_KEY" \
  --base_url "https://api.deepseek.com/v1" \
  --temperature 0.7 \
  --max_tokens 1024 \
  --output_dir experiments/runs
```

---

## 📊 实验结果分析

### 对比多次实验

```bash
# 对比最近 5 次实验
python -m eval.analyze --runs_dir experiments/runs --latest 5

# 对比指定实验
python -m eval.analyze --runs_dir experiments/runs --run_ids 20260610_224644 20260610_225338
```

### Harness 子系统覆盖矩阵

```bash
python harness_report.py
```

输出每种策略对 Harness 五子系统（Instructions / Tools / Environment / State / Feedback）的覆盖情况。

---

## 🧠 七种 CoT 策略详解

### 1. Base COT

- **原理**：在 Prompt 中加入 "Let's think step by step"，引导模型逐步推理
- **参数**：无额外参数
- **Harness 覆盖**：Instructions + Environment（2/5）
- **特点**：最简单、最快、性价比最高

### 2. Self-Consistency

- **原理**：对同一问题采样多条推理路径，通过**路径质量加权投票**确定最终答案。每条路径根据推理步数、数学符号密度、明确答案格式、长度合理性、答案一致性等维度评分，避免低质量路径干扰投票。支持**提前停止**（当领先者无法被超越时自动结束采样）和**空答案重试**机制
- **参数**：`--n_paths N`（推荐 7；策略内部默认 `min_paths=3`、`early_stop=True`、`retry_on_empty=True`）
- **Harness 覆盖**：Instructions + Environment + State（3/5）
- **特点**：轻量级路径质量评估显著提升投票可靠性。100 样本准确率达到 **94%**，与 Prefix Consistency 持平，且输出 token 仅 238.6，速度和成本远低于 Multi-Agent Debate

### 3. Prefix Consistency

- **原理**：对每条 CoT 推理链在中间截断（如 50%），用前缀重新生成后续内容。正确推理链的原始答案在再生中复现率更高，用这个复现率作为权重进行加权多数投票
- **参数**：`--n_paths N`（初始路径数，默认 5），`--truncation_ratio R`（截断比例，默认 0.5），`--regen_count K`（每前缀再生次数，默认 3），`--weight_fn F`（权重函数：`linear`/`quadratic`/`cubic`/`unanimous`，默认 `linear`）
- **Harness 覆盖**：Instructions + Environment + State + Feedback（4/5）
- **特点**：无需 logprob 或自评 prompt，用轻量级的再生一致性作为可靠性信号。100 样本准确率达到 **93.0%**，平均输出 token 仅 **159.9**，是所有高准确率策略中输出 token 最低的；但因前缀再生需要更多 API 调用，墙钟时间约 64.4s/题

### 4. Step-Aware Verifier

- **原理**：生成多条推理路径后，对每条路径调用 Verifier 模型打分，最终通过**加权投票**聚合答案（每条路径的 Verifier 得分作为投票权重）。
- **双模式支持**：
  - **LLM Verifier**（默认）：用 LLM 对路径的每一步打分，耗时长但通用
  - **本地 DeBERTa Verifier**（`--local_verifier`）：加载自定义训练的 DeBERTa-v3-large Token 分类模型，取 `[CLS]` 位置 `SOLUTION-CORRECT` 的 softmax 概率作为路径分。推理在本地 CUDA 上运行，**零 API 费用**，速度大幅提升
- **多 Prompt 多样生成**：支持 `--n_prompts` 个不同 prompt（各含随机 few-shot 示例），每个 prompt 生成 `--n_paths` 条路径
- **参数**：`--n_paths N`（每 prompt 路径数，默认 5），`--n_prompts N`（不同 prompt 数，默认 3），`--local_verifier`（启用本地 DeBERTa verifier），`--verifier_model_path PATH`（模型路径，默认 `data/checkpoint/`）
- **Harness 覆盖**：Instructions + Tools + Environment + State + Feedback（5/5）
- **特点**：完整覆盖 Harness 五子系统。本地 verifier 模式下，50 样本测试从 **206.6s/题**（LLM verifier）降至 **52.3s/题**（本地 verifier），快 4 倍。100 样本（deepseek-v4-flash LLM verifier）准确率 **94%**

### 5. Few-Shot CoT

- **原理**：在 prompt 中加入随机采样的 AQuA 示例作为少样本示范，引导 LLM 参照示例格式推理。每次 `run()` 重新随机采样，保证每道题示例不同
- **参数**：`--n_shots N`（示例个数，默认 5）
- **Harness 覆盖**：Instructions + Environment（2/5）
- **特点**：通过多样例示范提升推理一致性；示例由 LLM 动态生成，无需人工标注

### 6. RAG + COT

- **原理**：采用 IRCoT 风格的两跳检索流程，先用题目和知识库做首轮召回，再根据已检索证据生成 follow-up query 进行二跳检索，最后将去重后的证据块注入 Prompt 辅助推理
- **参数**：`--top_k K`（每跳检索文档数，默认 3），`--rag_hops N`（检索轮数，默认 2），`--rag_no_planner`（关闭 query planning）
- **Harness 覆盖**：Instructions + Tools + Environment + State（4/5）
- **特点**：相比一次性拼接检索结果，这一版更接近题目要求的“检索与思维链交替执行”。它通过 topic-aware seed query + query planning 降低无关知识干扰，适合知识密集型多步题

### 7. Multi-Agent Debate

- **原理**：5 个 LLM Agent 分别扮演不同角色（分析师 / 批判者 / 直觉者 / 验证者 / 综合者），通过 **ThreadPoolExecutor 并行执行**、**交叉评审**（每轮各 Agent 审阅其他 Agent 的答案并指出漏洞）、**收敛检测**（若所有 Agent 答案一致则提前停止）和 **多数投票** 决定最终答案
- **参数**：`--n_agents N`（Agent 数，默认 5），`--n_rounds R`（辩论轮数，默认 3）
- **Harness 覆盖**：Instructions + Environment + State + Feedback（4/5）
- **特点**：多角色并行协作显著提升了推理稳定性。100 样本准确率达到 **95.0%**，为所有策略中最高；50 样本准确率 **94.0%**。但因 5 Agent × 多轮调用，实际 API 调用量约为 15 次/题，成本较高

---

## 🔧 Harness Engineering 设计思想

本项目借鉴 [walkinglabs/learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering) 的五子系统模型：

| 子系统 | CoT 中的体现 | 激活策略 |
|---|---|---|
| **Instructions** | Prompt 模板设计（推理格式、角色指令、评分标准） | 全部 |
| **Tools** | 外部工具调用（检索器、Verifier 元推理工具） | rag_cot, step_verifier |
| **Environment** | AQuA 任务环境（数据加载、答案提取、评估指标） | 全部 |
| **State** | 运行时状态管理（多路径历史、检索上下文、辩论记录） | self_consistency, prefix_consistency, rag_cot, multi_agent_debate, step_verifier |
| **Feedback** | 反馈闭环（步骤级验证打分、多 Agent 互评纠错、前缀再生一致性） | prefix_consistency, multi_agent_debate, step_verifier |

**核心洞察**：子系统覆盖数 ≠ 准确率，但**高质量的 Feedback 机制**能显著提升性能。升级后的 `multi_agent_debate`（4/5）通过 5 Agent 并行 + 交叉评审 + 收敛检测，在 100 样本上达到 **95.0%**，为所有策略最高；`self_consistency`（3/5）和 `step_verifier`（5/5）均达到 **94.0%**。这说明**高质量局部评估**可以来自不同机制：Self-Consistency 通过本地路径质量评分提升 State 聚合质量；Prefix Consistency 通过前缀再生一致性引入轻量 Feedback；Multi-Agent Debate 则通过多角色互评和收敛检测实现最强推理稳定性。`rag_cot` 已升级为两跳 IRCoT 风格检索，但效果仍受知识库规模与覆盖面影响，说明 Tools 子系统的质量比覆盖本身更关键。

---

## 📈 100 样本实验结果（AQuA test 前 100 条）

| 策略 | 准确率 | 平均输出 Token | 平均输入 Token | 平均推理步数 | 综合性价比 |
|---|---|---|---|---|---|
| **multi_agent_debate** | **95.0%** ⭐ | 72.2* | — | 2.0 | ⭐⭐⭐⭐ **最高准确率，多 Agent 协作** |
| **self_consistency** | **94.0%** ⭐ | 238.6 | 130.0 | 8.0 | ⭐⭐⭐⭐⭐ **高准确率 + 低 token 消耗** |
| **step_verifier (LLM)** | **94.0%** ⭐ | 563.7 | — | 21.6 | ⭐⭐⭐ **准确率高但 token 消耗大** |
| **prefix_consistency** | **93.0%** ⭐ | **159.9** | 130.0 | 6.7 | ⭐⭐⭐⭐⭐ **最高准确率中输出 token 最低** |
| **rag_cot** | **92.0%** | 197.9 | 238.6 | 6.0 | ⭐⭐⭐⭐ 检索器升级后潜力较大 |
| base_cot | 91.0% | 187.6 | 130.0 | 5.8 | ⭐⭐⭐⭐⭐ **最佳基础性价比** |
| few_shot_cot | — | — | — | — | ⭐⭐⭐ 待基准测试 |

> *注：multi_agent_debate 的 output 字段仅记录投票摘要（非完整 Agent 输出），实际 API 调用约 5 agents × 最多 3 rounds = 15 次/题。

> 注：multi_agent_debate 与 step_verifier 的输入 token 因多轮交互统计方式不同，当前记录为 0。

## 📈 50 样本实验结果（AQuA test 前 50 条）

| 策略 | 准确率 | 平均耗时/条 | 平均 Token | 综合性价比 |
|---|---|---|---|---|
| **multi_agent_debate** | **94.0%** ⭐ | 18.9s | 70.4* | ⭐⭐⭐⭐ **多 Agent 并行，最高准确率之一** |
| **self_consistency** | **94.0%** ⭐ | 25.6s | 232.5 | ⭐⭐⭐⭐⭐ **最高准确率中最快** |
| **prefix_consistency** | **94.0%** ⭐ | 64.4s | 160.9 | ⭐⭐⭐⭐ **Feedback 可靠性 + 低输出 token** |
| **step_verifier (本地 DeBERTa)** | **94.0%** ⭐ | **52.3s** | — | ⭐⭐⭐⭐ **本地验证，零 API 费用** |
| base_cot | 92.0% | 5.2s | 174.4 | ⭐⭐⭐⭐⭐ **最佳基础性价比** |
| step_verifier (LLM) | 92.0% | **206.6s** | 387.9 | ⭐ 极慢，收益有限 |
| few_shot_cot | — | — | — | ⭐⭐⭐ 待基准测试 |
| rag_cot | **78.0%** ▼ | 4.2s | 151.3 | ⭐⭐ 检索噪声损害性能 |

> *注：multi_agent_debate 的 output 字段仅记录投票摘要，实际 API 调用约 5 agents × 最多 3 rounds = 15 次/题。

> 详细分析见 [`experiments/report_100_sample_20260615.md`](experiments/report_100_sample_20260615.md)  
> 历史 50 样本分析见 [`experiments/report_50_sample_20260611.md`](experiments/report_50_sample_20260611.md)

---

## 📝 核心文件说明

| 文件 | 说明 |
|---|---|
| `harness.py` | 实验主入口，注册全部策略，支持命令行参数，实时显示进度与准确率 |
| `scripts/run_experiments.py` | 批量测试 Python 脚本，支持串行/并行/单策略模式 |
| `scripts/run_all.sh` | 批量测试 Bash 脚本，顺序执行全部策略 |
| `eval/metrics.py` | 评估指标：准确率、推理步数、Token 估算、多实验对比 |
| `eval/analyze.py` | CLI 分析工具，对比多次实验并打印格式化表格 |
| `harness_report.py` | 生成 Harness 子系统覆盖矩阵 |
| `scripts/init.sh` | 环境验证脚本，检查依赖、模块导入、各策略干跑 |
| `feature_list.json` | 11 项功能的状态追踪（全部已完成） |
| `docs/progress.md` | 开发进度日志 |

---

## ⚠️ 已知问题与注意事项

1. **API 限流**：当前端点不支持 `n > 1` 的批量生成，`self_consistency`、`prefix_consistency` 和 `step_verifier` 已改为循环调用 `n=1`
2. **step_verifier LLM 模式极慢**：50 样本约需 3 小时（206.6s/题），建议使用 `--local_verifier` 本地模式（52.3s/题，快 4 倍）
3. **DeBERTa verifier 分数偏低**：当前分数范围 0.0~2.0（vs 理想 0~10），原因在于训练数据（紧凑数学符号）与 LLM 输出（英文叙述）的风格不匹配。相对排序仍然有效——分数高的路径在风格上更接近训练数据
4. **prefix_consistency 速度**：50 样本实测约 53.7 分钟（64.4s/题，3 路径 × 3 再生），墙钟时间高于增强版 Self-Consistency，但输出 token 更低
5. **rag_cot 检索噪声**：当前 keyword-based 检索可能引入无关知识，建议在高质量知识库上使用
6. **本地模型路径**：`data/checkpoint/` 目录需要存放完整的 DeBERTa checkpoint（`model.safetensors` + `config.json`），tokenizer 自动从 `microsoft/deberta-v3-large` 加载

---

## 📚 参考文献

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models: https://arxiv.org/abs/2201.11903
- Self-Consistency Improves Chain of Thought Reasoning in Language Models: https://arxiv.org/abs/2203.11171
- Prefix Consistency (PC-WMV): https://arxiv.org/abs/2605.07654
- RAG + COT (IRCoT): https://arxiv.org/abs/2212.09095
- Multi-Agent Debate: https://arxiv.org/abs/2305.14325
- Step-Aware Verifier: https://arxiv.org/abs/2310.15123
- Harness Engineering: https://github.com/walkinglabs/learn-harness-engineering
- AQuA Dataset: https://github.com/deepmind/AQuA

---

## 🤝 贡献与使用

本项目为思维链推理的实验性实现，欢迎基于本项目进行扩展：

- 接入更多数据集（GSM8K、MATH 等）
- 优化 RAG 检索器（使用 Embedding-based 语义检索）
- 改进本地 DeBERTa verifier（续训更多 epoch、混入 LLM 风格数据）
- 尝试更多 CoT 变体（Tree-of-Thought、Program-Aided LLM 等）
- 探索 Prefix Consistency 的变体（不同截断比例、基于句子的截断、多截断点融合）

---

> **Co-Authored-By**: Claude <noreply@anthropic.com>
