# Agentic Search RL

这是一个面向多轮信息检索的 Search Agent 后训练项目。项目借鉴并适配 **SmartSearch** 的查询级过程监督与偏好优化思路，在 **Champion 风格的交互式 Search 运行时**上组织 SFT → DPO → GRPO 流水线。

项目关注的核心问题是：让训练所使用的动作与检索证据，和 Agent 实际执行的轨迹保持一致，而不是重新从模型原始文本中猜测工具行为。

## 设计原则

- **以实际执行为依据**：真正执行的 Search query 与 observation 是 canonical 数据的权威来源。
- **支持多查询 Search**：一条动作可以包含多个 query，并保留每个 query 与对应检索结果的映射。
- **按角色构造训练目标**：监督 assistant 生成的内容；环境 observation 只作为上下文，不作为预测目标。
- **因果一致的查询改写**：修改 query 后重新检索并重新 rollout，不复用分支点之后的旧轨迹后缀。
- **统一运行时，多阶段策略**：Base 与各阶段单一 adapter 均通过同一个 PolicyBackend 接入 Champion。

## 总体架构

下图展示项目的目标训练架构。

```text
Teacher 轨迹
    │ 权威 action / observation 重建
    ▼
Canonical 轨迹 ──→ HF / PEFT SFT
                         │
                 SFT 策略 Agent rollout
                         │
                    查询级过程 Judge
                         │
                Query refinement + 真实分支
                         │
                  Canonical 偏好数据
                         ▼
              LLaMA-Factory-compatible DPO
                         │
                  Base + DPO adapter
                         ▼
           veRL 在线强化学习 ↔ Champion Search 环境
                         │
                        GRPO
```

### 训练流水线

| 阶段 | 主要职责 | 训练技术栈 |
|---|---|---|
| SFT | 学习 canonical reasoning、Search action 与最终答案 | HF / PEFT |
| 过程监督 | 诊断 query 意图和检索证据质量 | 冻结的查询级 Judge |
| 偏好构造 | 比较真实原始轨迹与 counterfactual 轨迹 | SmartSearch 风格排序 |
| DPO | 优化 canonical chosen/rejected continuation | 现代 LLaMA-Factory 兼容路径 |
| GRPO | 在交互式 Search 环境中进行组相对策略优化 | 现代上游 veRL |

历史 correctness baseline 与正式训练后端接口分开保留。具体命令及其真实支持范围见[入口说明](docs/ENTRYPOINTS.md)。

## Agent 框架与运行环境

本项目的 Agent 运行时建立在 [Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition](https://github.com/yiming-qing/Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition) 的框架思路之上。上游框架提供 FastAPI 入口、ReAct 循环、工具调用解析、超时与 token 保护；本项目在此基础上冻结 Visit，仅保留 Search 主线，并将 Search 改造成适配训练和评测的多查询 Champion 运行时。

典型调用链为：

```text
POST / → agent.py → agent_loop.py::react_agent(question)
       → System Prompt + 当前日期 → ReAct 循环
       → Search action 解析 → retriever → observation → 继续生成
       → final answer
```

训练与评测在项目虚拟环境中运行。检索侧使用 E5 向量编码与 CPU FAISS 索引；Search 的实际 query 和 observation 以离线 retriever 返回结果为准。项目在虚拟环境中进行检索， Visit 已冻结，本项目不依赖 Visit/Jina 抓取链路。

## Champion Search 环境

策略生成 action 后，Champion 负责解析动作、调用 retriever 执行 Search、追加真实 observation，并让模型继续生成。一条 Search action 可以包含 `[q1, q2, …]`；即使策略最终看到的是合并后的 observation，评分与审计仍保留每个 query 到自身检索结果的映射。

## Canonical 数据表示

| 数据类型 | 内容 | 必须保持的约束 |
|---|---|---|
| SFT 样本 | Canonical messages 与 provenance | 不通过 legacy `response` 重新构造 |
| 偏好样本 | 共享 prompt、chosen/rejected continuation 与 provenance | 同一问题，且两侧都来自真实执行轨迹 |
| 在线轨迹 | 每轮 input IDs、generated IDs 与 token origin | 环境 token 只作为上下文，不参与策略损失 |
| Checkpoint | Base identity、单阶段 adapter 与 lineage | 不隐式堆叠 SFT+DPO+GRPO adapter |

完整契约与源码职责见[架构说明](docs/ARCHITECTURE.md)。

## 仓库结构

```text
src/agentic_search_rl/
  runtime/                  # Champion 兼容的公共运行时 API
  data/                     # Canonical SFT、偏好数据和在线轨迹
  rewards/                  # 过程评分接口
  evaluation/               # 现有答案评估指标
  training/
    sft/                    # HF / PEFT 模型与数据构造
    dpo/                    # LLaMA-Factory canonical bridge
      baselines/            # 历史 TRL correctness baseline
    grpo/                   # 在线轨迹与 reward 集成边界
      baselines/            # TRL forward oracle
configs/                    # 模型身份、实验配置与可移植示例
scripts/
  data/                     # Teacher 生成、raw990 冻结与 canonical 重建入口
  process/                  # Process Judge V3 审计与评分入口
  sft/                      # SFT 数据构造与 preflight
  dpo/                      # DPO 数据构造与 preflight
  grpo/                     # GRPO correctness preflight
tests/                      # 离线回归测试
docs/                       # 主线架构与入口文档
```

旧实现 package 继续保留在薄兼容层之后，从而兼容现有 import，并保持冻结源码的 identity 不变。

## 快速开始

使用 Python 3.10+，并进入对应阶段的项目环境。查看入口和计算答案指标无需下载模型：

```bash
PYTHONPATH=src python -m agentic_search_rl --help
PYTHONPATH=src python -m agentic_search_rl entrypoints
PYTHONPATH=src python -m agentic_search_rl score-answer --prediction "Paris" --gold "Paris"
```

执行模型与数据 preflight 时，需要显式提供本地资产路径：

```bash
export ASRL_MODEL_PATH=/path/to/local/base
export ASRL_TOKENIZER_PATH=/path/to/exact/tokenizer
export ASRL_SFT_DATASET=/path/to/canonical_sft.jsonl
PYTHONPATH=src python -m agentic_search_rl sft-preflight --config configs/examples/sft.local.json
```

该命令只构造 dataset、model 和 optimizer state，不进行训练。示例中的资源参数仅用于展示配置方法，不代表任何硬件承载保证。运行模型前请先阅读[配置说明](configs/README.md)与[主线入口](docs/ENTRYPOINTS.md)。

当前公开数据：990 条主池位于 data/splits/sft_pool_990_seed20260904.jsonl，canonical SFT 候选位于 data/canonical/sft_candidates_632.jsonl（实际 632 条；未发现独立 660 条文件）。

## 可复现性

每次实验应记录 base/tokenizer revision、adapter identity、dataset hash、split 排除规则、seed、运行时依赖、generation 参数和 checkpoint lineage。Canonical 轨迹不允许静默截断；最终评估数据必须与 post-training 数据及 Judge calibration 数据严格隔离。

模型、索引、凭据、checkpoint 和大规模轨迹属于本地资产，不提交为源码依赖。需要提交的是可审计的配置、数据清单、统计和可重现脚本，详见[架构说明](docs/ARCHITECTURE.md)。

## 致谢与许可证

本项目借鉴 [SmartSearch](https://github.com/RUC-NLPIR/SmartSearch) 的部分PRM思路，并适配 [Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition](https://github.com/yiming-qing/Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition) 的交互式 Agent 语义，结合其guard机制改进了PRM机制。项目自身的 multi-query canonicalization、role-aware 训练桥接和 counterfactual 分支执行，与上游工作在架构文档中明确区分。

HF/PEFT、LLaMA-Factory、veRL 与 FlashRAG 为项目提供了训练和检索生态支持。第三方 attribution 与保留的许可证文本见[第三方声明](THIRD_PARTY_NOTICES.md)。仓库整体许可证需要由项目所有者在正式发布前确定；上游许可证义务始终有效。
