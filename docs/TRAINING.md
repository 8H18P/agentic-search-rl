# 训练主线

## SFT

入口：`bash scripts/sft/train.sh <config.json>`。`canonical_sft.runner` 统一支持 `mode=lora` 与 `mode=full`，共享 tokenizer、canonical collator、role-aware labels、稀疏 supervised causal CE 和 optimizer 构造。LoRA 通过配置指定 rank、alpha、target modules；模型和数据路径不得写死在源码。

## DPO

canonical chosen/rejected continuation 由 `src/canonical_sft/dpo.py` 表示，`src/canonical_sft/trl_dpo.py` 和 `llamafactory_dpo.py` 提供 TRL/现代 LLaMA-Factory bridge。`scripts/dpo/preflight.sh` 只做数据与 reference/policy 一致性检查；正式训练需在具备对应依赖的环境中显式启动。

## GRPO / veRL

`src/online_grpo/` 负责 Champion 轨迹与过程奖励接口，`scripts/grpo_verl/preflight.sh` 是当前公开入口。veRL 依赖和多卡资源单独安装配置；本仓库不在导入或 `--help` 时启动训练。

所有阶段都从 Base + 单阶段 adapter 恢复，不隐式叠加 SFT、DPO、GRPO adapter。训练配置、seed、数据 hash 和模型 revision 应写入运行 manifest。
