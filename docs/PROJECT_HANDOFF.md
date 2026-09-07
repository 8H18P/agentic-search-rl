# 项目当前状态交接

## 1. 项目定位

本项目是一个面向多轮信息检索的 Agent 后训练工程，当前主线为：

```text
Canonical 数据
→ Champion Search Agent
→ SFT
→ DPO
→ GRPO / veRL（实现入口已预留）
→ 统一评测
```

项目借鉴 SmartSearch 的查询级过程监督、查询改写和偏好优化思路，但 Agent 运行时基于 Champion 风格的交互式 Search 环境。

## 2. 当前代码状态

当前 Git 主分支已推送至 GitHub：

```text
repository: https://github.com/8H18P/agentic-search-rl
branch: main
HEAD: 25bfbb1
```

工作区应保持干净。README 和主线文档使用中文，Git 提交身份为：

```text
user.name  = Zachary Chan
user.email = chenweijie2025@163.com
```

## 3. Agent 与检索运行时

Agent 框架参考：

https://github.com/yiming-qing/Research-Agent---1st-place-in-Alibaba-Cloud-Data-AI-Competition

当前调用链：

```text
POST /
  → agent.py
  → agent_loop.py::react_agent(question)
  → System Prompt + 当前日期
  → ReAct 循环
  → Search action 解析
  → offline retriever
  → observation
  → 继续生成
  → final answer
```

运行约束：

- 训练和评测在项目虚拟环境中运行；
- 检索使用 E5 向量编码与 CPU FAISS；
- Search 是当前唯一启用的工具主线；
- Visit 已冻结，不作为训练和评测依赖；
- Champion multi-query action 保留 query 与 result 的权威映射；
- 实际 Search query 和 observation 以 retriever 返回结果为准。

## 4. 当前数据资产

核心数据必须保留，不得删除：

```text
data/splits/sft_pool_990_seed20260904.jsonl
```

- 990 条主数据；
- canonical SFT candidate：

```text
data/canonical/sft_candidates_632.jsonl
```

- 当前核验为 632 条，不将其伪造为 660 条；
- 数据生成、canonical 重建和筛选脚本位于 `scripts/data/`；
- validation400 不属于当前主线训练数据。

数据原则：

```text
canonical authoritative data
→ canonical messages
→ exact tokenizer
→ role-aware labels
```

## 5. 训练主线与入口

脚本按任务分目录组织：

```text
scripts/
├── agent/
├── data/
├── sft/
├── dpo/
├── grpo_verl/
├── prm/
└── evaluation/
```

主要入口：

```bash
scripts/data/generate_teacher.sh
scripts/data/freeze_raw990.sh
scripts/data/build_canonical_candidates.sh
scripts/sft/preflight.sh
scripts/sft/train.sh
scripts/dpo/preflight.sh
scripts/grpo_verl/preflight.sh
scripts/prm/audit_v3.sh
scripts/prm/score_v3.sh
scripts/evaluation/evaluate.sh
```

训练阶段说明：

- SFT：Qwen3.5-4B + LoRA，使用 canonical 数据和 role-aware supervision；
- DPO：使用真实 original/counterfactual preference candidate，保持 policy/reference 语义一致；
- GRPO / veRL：保留主线目录和 preflight 入口，正式训练需在满足算力与运行时条件后执行；
- PRM：Process Judge V3 用于 query-level Intent / Retrieval 过程评分；
- Evaluation：统一使用 Champion runtime、现有 parser、retriever 和 outcome evaluator。

## 6. 模型与配置

模型相关配置位于：

```text
configs/model/
```

默认模型身份：

```text
Qwen/Qwen3.5-4B
```

实际部署可通过环境变量指定本地模型、tokenizer、SFT/DPO adapter 路径，不应把机器私有路径或密钥写入代码、配置和文档。

## 7. 文档入口

```text
docs/ARCHITECTURE.md
docs/ENTRYPOINTS.md
docs/DATA_PIPELINE.md
docs/TRAINING.md
docs/EVALUATION.md
docs/PRM.md
docs/REPOSITORY_CLEANUP_REPORT.md
```

README 是项目总入口；上述文档分别说明架构、脚本入口、数据管线、训练、评测和过程评分。

## 8. 环境与安全约束

安装依赖：

```bash
pip install -r requirements.txt
```

建议在项目虚拟环境中执行所有训练和评测命令。W&B、DashScope、检索服务等凭据只从环境变量或本地 login 获取，不提交到 Git，不写入日志和 artifact。

不要把以下内容重新加入公开仓库：

- 历史 smoke / Mini E2E 运行数据；
- baseline 对比日志；
- 临时审计记录；
- 完整 raw trajectory 或敏感问题内容；
- 模型权重、adapter 大文件和 API key。

## 9. 后续使用原则

后续开发只沿当前主线进行：

1. 先确认虚拟环境、模型身份和 retriever 状态；
2. 使用 `scripts/` 下对应阶段的入口；
3. 保持 canonical 数据、Champion runtime 和统一评测契约；
4. 训练产物放在外部 artifact 路径，不污染公开代码仓库；
5. 新增实验文档只保留可复现、可审计且属于主线的内容。
