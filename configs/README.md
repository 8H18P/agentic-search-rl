# 配置归属

- `model/`：现有 base/tokenizer 身份和运行时契约。
- `runtime/`：各阶段专用的依赖与源码身份。
- `smartsearch_aligned/`：与上游对齐的配置证据。
- `mini_e2e/`：历史上针对特定资源的实验配置。
- `examples/`：可移植的公开示例，路径由显式环境变量提供。

现有实验/冻结配置保持不变，以保留可复现性。其中的机器路径和规模是历史输入，不是公开默认值。它们仍需发布审查；不要盲目复制到其他主机。

`examples/sft.local.json` 由公开 SFT preflight wrapper 读取。该 wrapper 会展开 `${ASRL_MODEL_PATH}`、`${ASRL_TOKENIZER_PATH}` 和 `${ASRL_SFT_DATASET}`，未设置变量时立即失败。Model/tokenizer 仍只允许本地加载。同一流程中可将 mode 设为 `lora` 或 `full`。长度、batch、梯度累积、epoch、max steps 和 LoRA 参数属于实验配置，不代表某张显卡一定能承载训练。

示例数值仅用于说明，未经 benchmark。现有 model loader 仍在源码中选择 SDPA；修改该行为是独立技术债务，不会被本示例隐藏。

没有新增 LF/veRL 训练示例冒充已通过运行时 gate。只能通过已验证 bridge 和后续阶段专用工作进行配置。各阶段运行时环境刻意保持隔离，而不是使用一份未验证的大而全 requirements 文件。

密钥必须来自环境变量或本地登录，绝不能来自配置。W&B 通过现有 `report_to` 选择启用；不得将 API key 写入 JSON 配置。
