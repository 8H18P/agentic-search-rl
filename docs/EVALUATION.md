# 评测主线

评测复用 `Champion runtime`、统一 `PolicyBackend`、离线 retriever 和 `src/rl_agent/evaluator.py`。策略、问题 split、generation budget 与 retriever 必须在 manifest 中明确记录；完整 trajectory trace 应保留在仓库外供人工核查。

公共指标包括答案 EM/F1、Search action/query 数、tool-call 有效率、invalid action、正常终止、force answer、循环或预算终止。validation400 只在明确的正式评测命令中使用，不参与训练数据构建。

```bash
bash scripts/evaluation/evaluate.sh --prediction "Paris" --gold "Paris"
```

该入口只执行无模型的答案指标示例；大规模 rollout 与结果写盘由调用方提供 manifest 和输出目录。

四阶段 rollout 完成后，使用严格聚合入口生成统一 JSON 与中文 Markdown 报告：

```bash
bash scripts/evaluation/compare_stages.sh \
  --stage Base=/path/to/base.jsonl \
  --stage SFT=/path/to/sft.jsonl \
  --stage DPO=/path/to/dpo.jsonl \
  --stage GRPO=/path/to/grpo.jsonl \
  --output-json /path/to/stage_comparison.json \
  --output-markdown /path/to/STAGE_COMPARISON.md
```

逐题 JSONL 必须包含 `question_id`、EM/F1、task success、Search action/query 数、valid/repeated query 数、retrieval success 数、process scores、trajectory reward 和 termination reason。任何必需字段缺失或四阶段问题集合不一致都会直接失败，聚合器不会用默认值伪造结果。
