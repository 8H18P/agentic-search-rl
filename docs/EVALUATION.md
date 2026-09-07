# 评测主线

评测复用 `Champion runtime`、统一 `PolicyBackend`、离线 retriever 和 `src/rl_agent/evaluator.py`。策略、问题 split、generation budget 与 retriever 必须在 manifest 中明确记录；完整 trajectory trace 应保留在仓库外供人工核查。

公共指标包括答案 EM/F1、Search action/query 数、tool-call 有效率、invalid action、正常终止、force answer、循环或预算终止。validation400 只在明确的正式评测命令中使用，不参与训练数据构建。

```bash
bash scripts/evaluation/evaluate.sh --prediction "Paris" --gold "Paris"
```

该入口只执行无模型的答案指标示例；大规模 rollout 与结果写盘由调用方提供 manifest 和输出目录。
