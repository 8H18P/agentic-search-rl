# 配置归属

- `examples/`：可移植的公开示例，路径由显式环境变量提供。
- `process_judge/v3_frozen/`：冻结的 V3 Process Judge 的 prompt、输出 schema 与 manifest，语义不得改动。

模型身份 manifest 由 `scripts/config/freeze_model_identity.py` 在运行时生成，不随源码发布。示例中的机器路径和规模属于历史输入，不是公开默认值，不要盲目复制到其他主机。

`examples/sft.local.json` 由公开 SFT preflight wrapper 读取。该 wrapper 会展开 `${ASRL_MODEL_PATH}`、`${ASRL_TOKENIZER_PATH}` 和 `${ASRL_SFT_DATASET}`，未设置变量时立即失败。Model/tokenizer 仍只允许本地加载。同一流程中可将 mode 设为 `lora` 或 `full`。长度、batch、梯度累积、epoch、max steps 和 LoRA 参数属于可调超参数，不代表某张显卡一定能承载训练。

示例数值用于说明配置结构；实际运行由显式模型身份、设备和 manifest 固化。默认 attention backend 为 SDPA。

`examples/dpo.local.json` 是项目原生 DPO 入口的可移植示例；`examples/grpo.local.json` 对应在线 GRPO。各阶段运行时环境保持隔离，GRPO 核心版本见 `requirements-grpo-lock.txt`。

`examples/grpo.local.json` 通过环境变量绑定 DPO checkpoint、问题池、排除集、retriever 与 Judge。模型身份文件不要手写 hash，可使用：

```bash
python scripts/config/freeze_model_identity.py \
  --model-path "$ASRL_MODEL_PATH" \
  --tokenizer-path "$ASRL_TOKENIZER_PATH" \
  --model-id Qwen/Qwen3.5-4B \
  --revision <精确 revision> \
  --output /path/to/model_identity.json
```

密钥必须来自环境变量或本地登录，绝不能来自配置。W&B 通过现有 `report_to` 选择启用；不得将 API key 写入 JSON 配置。
