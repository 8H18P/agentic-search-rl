# 主线入口

所有命令从仓库根目录执行。脚本只负责设置 `PYTHONPATH` 和转发参数，资源路径通过参数或环境变量提供。

## Agent 与数据

```bash
bash scripts/data/generate_teacher.sh --manifest <questions.jsonl> --run-id <id>
bash scripts/data/freeze_raw990.sh ...
bash scripts/data/build_canonical_candidates.sh ...
```

Teacher runner 位于 `scripts/agent/`；canonical 构建和正式 SFT 序列化位于 `scripts/data/`。

## SFT

```bash
bash scripts/sft/preflight.sh configs/examples/sft.local.json
bash scripts/sft/train.sh <sft-config.json>
```

`mode` 可选 `lora` 或 `full`。训练代码共享同一数据和 loss 路径。

## DPO

```bash
bash scripts/dpo/preflight.sh <bridge-args>
```

该入口验证 canonical bridge；实际训练由 `src/canonical_sft/trl_dpo.py` 或 LLaMA-Factory 环境显式启动。

## GRPO / veRL

```bash
bash scripts/grpo_verl/preflight.sh <preflight-args>
```

这是 veRL/在线轨迹的诊断入口，不会在导入或 `--help` 时启动训练。

## PRM 与评测

```bash
bash scripts/prm/audit_v3.sh --repo . --artifact-dir <dir> --output <report>
bash scripts/prm/score_v3.sh <judge-args>
bash scripts/evaluation/evaluate.sh --prediction "Paris" --gold "Paris"
```

V3 冻结 prompt/schema 位于 `configs/process_judge/v3_frozen/`，不得改变其语义。评测结果和完整轨迹写入仓库外目录。
