# 项目架构

## 数据到策略

```text
Teacher 实际执行事件
  → raw990 主池
  → canonical action/observation 重建
  → SFT（LoRA 或 Full FT）
  → Champion runtime rollout
  → Process Judge V3 / PRM
  → canonical DPO
  → GRPO/veRL 在线优化
  → 统一评测
```

## 模块职责

| 模块 | 位置 | 职责 |
|---|---|---|
| Agent runtime | `src/champion_runtime/` | 解析 Search action、调用 retriever、追加 observation、控制轮次 |
| 数据构建 | `scripts/agent/`、`scripts/data/` | 生成 Teacher 轨迹、冻结统计、canonical 重建 |
| SFT | `src/canonical_sft/` | tokenizer、角色掩码、稀疏 CE、LoRA/Full FT |
| DPO | `src/canonical_sft/dpo.py`、`trl_dpo.py`、`llamafactory_dpo.py` | chosen/rejected 表示和训练桥接 |
| GRPO | `src/online_grpo/`、`scripts/grpo_verl/` | 在线轨迹、过程奖励和 veRL 接口 |
| PRM | `scripts/prm/`、`src/agentic_search_rl/rewards/` | 冻结 V3 的 Intent/Retrieval query 评分 |
| 评测 | `src/agentic_search_rl/evaluation/`、`src/rl_agent/evaluator.py` | EM/F1、协议和轨迹指标 |

## 关键不变量

1. 实际执行的 action 与对应 tool response 是 canonical 唯一来源。
2. multi-query action 保留 query 到 result 的位置映射。
3. 训练中 observation 作为上下文，角色掩码只监督 assistant 生成内容。
4. 分支修改后必须重新检索并重新生成后缀，不能复用旧 suffix。
5. 每个阶段使用 Base + 当前单一 adapter，路径和身份写入 manifest。

模型、索引、checkpoint 和运行日志均通过配置指向外部资产，不随源码发布。
