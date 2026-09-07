# 训练主线

## SFT

入口：`bash scripts/sft/train.sh <config.json>`。`canonical_sft.runner` 统一支持 `mode=lora` 与 `mode=full`，共享 tokenizer、canonical collator、role-aware labels、稀疏 supervised causal CE 和 optimizer 构造。LoRA 通过配置指定 rank、alpha、target modules；模型和数据路径不得写死在源码。

## DPO

canonical chosen/rejected continuation 由 `src/canonical_sft/dpo.py` 表示。正式入口是：

```bash
bash scripts/dpo/train.sh configs/examples/dpo.local.json
```

`src/canonical_sft/dpo_runner.py` 显式加载两个互不共享参数对象的 Base + SFT adapter：policy 可训练，reference 全冻结。入口执行 sigmoid DPO loss、`backward()`、梯度审计、`optimizer.step()`、checkpoint、optimizer state、clean reload，并输出 `grpo_handoff.json`。配置 `eval_dataset_path` 后会在独立偏好集上计算 loss 与 reward accuracy；未配置时相应字段标记为 training-pairs diagnostic。

## GRPO

`src/online_grpo/` 负责 Champion 轨迹、冻结 reference、过程奖励和实际参数更新。只读 preflight 与正式训练明确分开：

```bash
bash scripts/grpo/preflight.sh <grpo-config.json>
bash scripts/grpo/train.sh <grpo-config.json>
```

正式入口在同一 policy version 下先完成整个 group rollout，再快照 old logprob，执行 GRPO `backward()`、梯度裁剪和 `optimizer.step()`；每次更新后都会用新 policy 重新进入 Champion Search 环境，并记录 reward、KL、policy loss、grad norm、参数 hash、checkpoint 与 clean reload。

所有阶段都从 Base + 单阶段 adapter 恢复，不隐式叠加 SFT、DPO、GRPO adapter。训练配置、seed、数据 hash 和模型 revision 应写入运行 manifest。GRPO 跨进程续训需要同时设置 `training.resume_from_checkpoint` 并传入 `--resume`：入口先从原始 DPO adapter 构造冻结 reference，再把 checkpoint policy 恢复到 default adapter，并恢复 optimizer state；reference 不会随 checkpoint 漂移。
