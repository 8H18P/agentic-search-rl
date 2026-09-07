# Agentic Search RL

这是一个面向多轮信息检索的 Search Agent 后训练项目。项目借鉴并适配 **SmartSearch** 的查询级过程监督与偏好优化思路，在 **Champion 风格的交互式 Search 运行时**上组织 SFT → DPO → GRPO 流水线。

项目关注的核心问题是：解决多轮 Search Agent 在长链路任务中最终奖励稀疏、错误搜索难以归因以及中间步骤信用分配困难的问题。当前实现使用冻结的查询级 LLM Judge（不是独立训练的本地 PRM checkpoint）评估每一步 Query 的搜索意图与检索结果有效性，并将其作为 process reward 用于偏好优化与强化学习。

## 设计原则

- **查询级过程信用分配**：不只依赖最终答案奖励，而是将长链路 Search 轨迹拆解到 Query 级别，对中间搜索决策进行独立评价，缓解多轮 Agent 中最终奖励稀疏、错误步骤难以归因的问题。
- **过程奖励驱动搜索质量优化**：通过 PRM 分别评估 Query 的搜索意图与真实检索结果有效性，形成细粒度 Process Reward，使训练目标从“最终答对”进一步扩展到“中间搜索步骤是否有效”。
- **SFT 学习能力，DPO 优化 Query，GRPO 优化长期策略**：SFT 负责建立基本的多轮 Search 与工具调用能力；DPO 利用 Query-level preference pairs 提升查询准确性与相关性；GRPO 将过程奖励引入在线 rollout，进一步优化跨多轮的搜索、证据利用与停止策略。
- **提升有效搜索而非单纯增加搜索次数**：训练目标不仅关注最终回答质量，同时降低重复 Query、无收益 Search 和重复 Observation，使模型在更少的 Search actions 下获得更高的有效检索率与 Search Efficiency。
- **因果一致的 Query Refinement**：对于低质量 Query，在分支点进行改写后必须重新执行真实检索并重新 rollout，旧 Query 对应的 observation 与后续轨迹不再复用，保证偏好数据和过程奖励真实反映 Query 修改带来的因果效果。
- **真实交互轨迹作为训练依据**：所有过程监督、偏好构造和强化学习奖励均建立在 Agent 实际执行的 Search query 与返回 evidence 上，避免仅根据模型文本表面形式判断搜索行为，使训练信号与真实环境交互保持一致。

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
               Canonical native DPO
                         │
                  Base + DPO adapter
                         ▼
            TRL GRPO 训练 ↔ Champion Search 环境
                         │
                        GRPO
```

### 训练流水线

| 阶段 | 主要职责 | 训练技术栈 |
|---|---|---|
| SFT | 学习 canonical reasoning、Search action 与最终答案 | HF / PEFT |
| 过程监督 | 诊断 query 意图和检索证据质量 | 冻结的查询级 Judge |
| 偏好构造 | 比较真实原始轨迹与 counterfactual 轨迹 | SmartSearch 风格排序 |
| DPO | 优化 canonical chosen/rejected continuation | 项目原生 role-aware DPO 训练器 |
| GRPO | 在交互式 Search 环境中进行组相对策略优化 | TRL objective + Champion 分段回放 |

历史 correctness baseline 与正式训练后端接口分开保留。具体命令及其真实支持范围见[入口说明](docs/ENTRYPOINTS.md)。

## Agent 框架与运行环境

本项目的 Agent 运行时建立在 [Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition](https://github.com/yiming-qing/Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition) （本项目称为champion agent）的框架思路之上。上游框架提供 FastAPI 入口、ReAct 循环、工具调用解析、超时与 token 保护；本项目在此基础上冻结 Visit，仅保留 Search 主线，并将 Search 改造成适配训练和评测的多查询 Champion 运行时。在PRM阶段，本项目将 Champion 中 Guard 的机制融入了奖励函数，作为合法性检验。

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
src/
  agentic_search_rl/        # 公共 facade：runtime/data/rewards/evaluation/training 命名空间
  canonical_sft/            # SFT 与 DPO 的 canonical 数据、loss 与正式训练入口
  champion_runtime/         # Champion Search 环境与 HF/PEFT policy backend
  online_grpo/              # 在线轨迹 capture、过程奖励与 TRL GRPO 参数更新
  rl_agent/                 # 答案评估与 policy/protocol 适配
configs/
  examples/                 # 可移植的 SFT/DPO/GRPO 示例配置
  process_judge/v3_frozen/  # 冻结的 V3 Process Judge prompt/schema/manifest
scripts/
  agent/                    # Teacher rollout 生成
  config/                   # 模型身份冻结工具
  data/                     # Teacher 生成、raw990 冻结与 canonical 重建入口
  prm/                      # Process Judge V3 评分入口
  sft/                      # SFT 数据构造与训练入口
  dpo/                      # DPO 训练入口
  grpo/                     # GRPO preflight/train 入口
  grpo_runtime/             # GRPO 共享运行引擎
  evaluation/               # 答案指标与四阶段统一评测
tests/                      # 离线回归测试
docs/                       # 主线架构与入口文档
```

`agentic_search_rl` 是面向公开命令的薄命名空间；具体实现以 `canonical_sft`、`champion_runtime`、`online_grpo`、`rl_agent` 为准，保持模块 identity 与冻结源码不变。

## 快速开始

使用 Python 3.10+，并进入对应阶段的项目环境。查看入口和计算答案指标无需下载模型：

```bash
PYTHONPATH=src python -m agentic_search_rl --help
PYTHONPATH=src python -m agentic_search_rl entrypoints
PYTHONPATH=src python -m agentic_search_rl score-answer --prediction "Paris" --gold "Paris"
```

正式执行模型训练时，需要显式提供本地资产路径：

```bash
export ASRL_MODEL_PATH=/path/to/local/base
export ASRL_TOKENIZER_PATH=/path/to/exact/tokenizer
export ASRL_SFT_DATASET=/path/to/canonical_sft.jsonl
PYTHONPATH=src python -m agentic_search_rl sft-train --config configs/examples/sft.local.json
```

只构造 dataset、model 和 optimizer 而不更新参数的命令是 `sft-preflight`。正式 DPO 使用 `scripts/dpo/train.sh`，正式 GRPO 使用 `scripts/grpo/train.sh`。示例中的资源参数仅用于展示配置方法，不代表任何硬件承载保证。运行模型前请先阅读[配置说明](configs/README.md)与[主线入口](docs/ENTRYPOINTS.md)。

仓库不提交私有或受许可证约束的大规模轨迹与 split；`data/` 默认被 `.gitignore` 排除。公开仓库提供生成与冻结脚本，使用者须自行提供合法原始数据，并把生成文件 hash 写入 run manifest。不得把本地存在但 Git 未跟踪的数据描述为公开数据。

## 可复现性

每次运行应记录 base/tokenizer revision、adapter identity、dataset hash、split 排除规则、seed、运行时依赖、generation 参数和 checkpoint lineage。Canonical 轨迹不允许静默截断；最终评估数据必须与 post-training 数据及 Judge calibration 数据严格隔离。

模型、索引、凭据、checkpoint 和大规模轨迹属于本地资产，不提交为源码依赖。需要提交的是可审计的配置、数据清单、统计和可重现脚本，详见[架构说明](docs/ARCHITECTURE.md)。

## 运行与产物

仓库提供 SFT、DPO、在线 GRPO、checkpoint/reload、更新后 rollout 与四阶段统一评测的完整代码入口。各入口在运行时生成训练日志、参数更新审计、checkpoint/reload 清单与四阶段统一评测报告，并写入配置指定的输出目录。

## 致谢与许可证

本项目借鉴 [SmartSearch](https://github.com/RUC-NLPIR/SmartSearch) 的部分PRM思路，并适配 [Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition](https://github.com/yiming-qing/Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition) 的交互式 Agent 语义，结合其guard机制改进了PRM机制。项目自身的 multi-query canonicalization、role-aware 训练桥接和 counterfactual 分支执行，与上游工作在架构文档中明确区分。

HF/PEFT、TRL 与 FlashRAG 为项目提供了训练和检索生态支持。第三方 attribution 与保留的许可证文本见[第三方声明](THIRD_PARTY_NOTICES.md)。仓库整体许可证需要由项目所有者在正式发布前确定；上游许可证义务始终有效。
