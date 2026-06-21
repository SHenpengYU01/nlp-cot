# Session Progress Log

## Current State

**Last Updated:** 2026-06-16
**Session ID:** 100-sample-exp
**Active Feature:** 100-sample benchmark experiments (feat-004 ~ feat-008)

## Status

### What's Done

- [x] 阅读选题说明，理解作业要求（基础COT + BONUS: 借鉴Harness设计思想实现CoT）
- [x] 调研 learn-harness-engineering 仓库，理解五子系统模型
- [x] 安装 harness-creator skill 并运行脚本生成基础 harness
- [x] 定制 CLAUDE.md — 项目描述、工作流、验证命令
- [x] 定制 feature_list.json — COT 实验的10项功能清单
- [x] 定制 init.sh — 环境验证脚本
- [x] 创建项目代码目录结构（data/, prompts/, strategies/, tasks/, models/, eval/, experiments/runs/）
- [x] 编写 harness.py 主入口框架（五子系统：Instructions/Tools/Environment/State/Feedback）
- [x] 封装 LLM 模型接口（models/base.py + models/openai_api.py，默认兼容 DeepSeek 官方 OpenAI-compatible 接口）
- [x] 实现 AQuA 数据集加载（tasks/aqua_task.py）
- [x] 实现基础 COT 策略（strategies/base_cot.py + prompts/base_cot.txt）
- [x] 实现 Self-Consistency 策略（strategies/self_consistency.py）
- [x] 实现 Step-Aware Verifier 策略（strategies/step_verifier.py + prompts/step_verifier.txt）
- [x] 实现检索增强 COT / RAG+COT（strategies/rag_cot.py + retrieval/ + data/knowledge_base.json + prompts/rag_cot.txt）
  - [x] 升级为 IRCoT 风格两跳检索：topic-aware seed query + query planning + 去重证据拼接
- [x] 实现 Multi-Agent Debate 策略（strategies/multi_agent_debate.py）
- [x] 实现评估指标与实验记录（eval/metrics.py + eval/analyze.py）
- [x] harness.py 注册全部策略并支持策略参数：n_paths、n_agents、n_rounds、top_k
- [x] 实现 BONUS: Harness Engineering 思想融合
  - [x] BaseStrategy 增加 `harness_subsystems()` 五子系统声明
  - [x] 各策略显式声明 Instructions / Tools / Environment / State / Feedback 覆盖情况
  - [x] harness_report.py 生成 Harness 子系统覆盖矩阵
  - [x] verify_feat010.py 验证五子系统声明与覆盖演进
- [x] 编写 dry-run 验证脚本：verify_feat004.py ~ verify_feat010.py
- [x] `./init.sh` 验证通过（含 feat-004 ~ feat-010 干跑验证）
- [x] 100 样本真实 API 实验（AQuA test 前 100 条）
  - [x] base_cot：91/100 = 91.0%，avg_out_tokens=187.6
  - [x] self_consistency：94/100 = 94.0%，avg_out_tokens=238.6
  - [x] rag_cot：92/100 = 92.0%，avg_out_tokens=197.9
  - [x] multi_agent_debate：95/100 = 95.0%，avg_out_tokens=72.2（5 Agent并行+交叉评审+收敛检测，output仅记录投票摘要）
  - [x] prefix_consistency：93/100 = 93.0%，avg_out_tokens=159.9
  - [x] step_verifier：94/100 = 94.0%，avg_out_tokens=563.7

### What's In Progress

- 无

### What's Next

1. 补全剩余策略的 100 样本基准测试：
   - few_shot_cot：`python harness.py --strategy few_shot_cot --dataset aqua --n_samples 100 --n_shots 5`
   - step_verifier（LLM）：`python harness.py --strategy step_verifier --dataset aqua --n_samples 100 --n_paths 3`
   - step_verifier（本地 DeBERTa，100 样本）
2. 对实验结果运行对比分析：`python eval/analyze.py --runs_dir experiments/runs --latest 5`
3. 根据实验结果撰写报告：比较各 CoT 策略准确率、推理步数、token消耗与 Harness 子系统覆盖关系

## Blockers / Risks

- [ ] **模型访问**：需要 OPENAI_API_KEY 才能运行实际实验（可通过 `--api_key` 或环境变量传入）
- [x] **AQuA数据**：已存在于 data/AQuA/，格式为 JSON Lines（train.json, dev.json, test.json）
- [x] **BONUS方向**：已修正为“借鉴 Harness Engineering 五子系统设计思想系统化实现 CoT”，并在策略代码中体现

## Decisions Made

- **BONUS理解**：不是将 CoT 迁移到 Agentic Task，而是借鉴 Harness Engineering 的系统设计思想来实现 CoT
- **Harness框架**：采用 walkinglabs 五子系统模型（Instructions/Tools/Environment/State/Feedback）
- **实验管理**：统一入口 harness.py，配置驱动，结果自动记录到 experiments/runs/
- **评估方法**：Controlled variable exclusion test（固定模型，每次只变一个策略），并结合 Harness 子系统覆盖矩阵解释策略差异
- **模型调用方式**：OpenAI-compatible API（默认模型 deepseek-v4-flash，默认 base_url 为 DeepSeek 官方 `https://api.deepseek.com/v1`）

## Files Modified This Session

- `CLAUDE.md` — 项目指令文件，修正 BONUS 描述
- `feature_list.json` — 功能清单，feat-001 ~ feat-010 全部完成
- `progress.md` — 会话进度日志（本文件）
- `session-handoff.md` — 会话交接模板，修正 BONUS 描述
- `init.sh` — 环境验证脚本，加入 feat-004 ~ feat-010 干跑验证
- `harness.py` — 实验管理主入口，注册全部策略与策略参数
- `requirements.txt` — Python 依赖
- `models/base.py` — LLM 抽象基类
- `models/openai_api.py` — OpenAI API 封装
- `models/__init__.py` — 模型包入口
- `tasks/base.py` — 任务抽象基类
- `tasks/aqua_task.py` — AQuA 数据集加载与评估
- `tasks/__init__.py` — 任务包入口
- `strategies/base.py` — 策略抽象基类，新增五子系统声明接口
- `strategies/base_cot.py` — 基础 COT 策略
- `strategies/self_consistency.py` — Self-Consistency 策略
- `strategies/step_verifier.py` — Step-Aware Verifier 策略
- `strategies/rag_cot.py` — RAG+COT 策略
- `strategies/multi_agent_debate.py` — Multi-Agent Debate 策略
- `strategies/__init__.py` — 策略包入口
- `retrieval/base.py` — 检索器抽象接口
- `retrieval/simple_retriever.py` — 简单关键词检索器
- `retrieval/__init__.py` — 检索包入口
- `eval/metrics.py` — 评估指标（准确率、推理步数、token估算、对比分析）
- `eval/analyze.py` — 实验结果对比工具
- `prompts/base_cot.txt` — 基础 COT prompt 模板
- `prompts/step_verifier.txt` — verifier prompt 模板
- `prompts/rag_cot.txt` — RAG+COT prompt 模板
- `data/knowledge_base.json` — 简单数学知识库
- `harness_report.py` — Harness 五子系统覆盖矩阵报告
- `verify_feat004.py` ~ `verify_feat010.py` — 各功能 dry-run 验证脚本

## Evidence of Completion

- [x] Harness文件完整性：`./init.sh` 通过验证
- [x] 目录结构：`data/ prompts/ strategies/ tasks/ models/ eval/ experiments/runs/ retrieval/` 已创建
- [x] 功能清单：10项功能定义清晰，依赖关系合理，状态均为 completed
- [x] 模块导入验证：python 可成功导入所有核心模块
- [x] harness.py 可执行：`python harness.py --help` 正常输出
- [x] feat-004 干跑验证：`python verify_feat004.py` 通过，端到端流程正确
- [x] feat-005 干跑验证：`python verify_feat005.py` 通过，多数投票逻辑正确
- [x] feat-006 干跑验证：`python verify_feat006.py` 通过，步骤级验证逻辑正确
- [x] feat-007 干跑验证：`python verify_feat007.py` 通过，检索增强逻辑正确
- [x] feat-008 干跑验证：`python verify_feat008.py` 通过，多Agent辩论逻辑正确
- [x] feat-009 干跑验证：`python verify_feat009.py` 通过，指标与实验记录逻辑正确
- [x] feat-010 干跑验证：`python verify_feat010.py` 通过，Harness子系统声明与覆盖矩阵正确
- [x] 完整验证：`bash init.sh` 通过

## Notes for Next Session

- 当前代码层面 10 个功能均已完成并通过 dry-run 验证。
- 下一步重点不是继续加功能，而是运行真实 API 实验并收集结果。
- 如果真实 API 成本或速度有限，建议每个策略先跑 `--n_samples 20`，确认流程后再扩大样本量。
