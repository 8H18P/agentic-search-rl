# 第三方声明

## ChampionAgent

Champion 兼容运行时源自上游 ChampionAgent 实现。
已保留 MIT 声明：[ChampionAgent-MIT.txt](third_party/licenses/ChampionAgent-MIT.txt)。
Copyright (c) 2026 Yiming Han。已与本地上游 LICENSE 核对。

## SmartSearch

查询级监督、查询改写与偏好优化的概念和源码参考基础：
[RUC-NLPIR/SmartSearch](https://github.com/RUC-NLPIR/SmartSearch)，已审计源码版本
`5e9e7ec0d6af34f45fad145d4bff442529e7e6ff`。
已保留 MIT 声明：[SmartSearch-MIT.txt](third_party/licenses/SmartSearch-MIT.txt)。
Copyright (c) 2025 SmartSearch。不得将上游实验声称为本项目结果。

## 训练与检索依赖

HF Transformers/PEFT、现代 LLaMA-Factory、veRL、TRL 和 FlashRAG 均为外部依赖。
各软件包保留自身许可条款。公开包发现规则不包含嵌套 checkout。
根目录中的 SmartSearch MIT 声明不会取代其 vendored 依赖（例如 LLaMA-Factory）内部的许可声明。

本次重构期间，在指定源码 checkout 中未找到 FlashRAG 许可证文本。
在精确版本和许可条款核实之前，不得再分发 FlashRAG 源码或 vendor bundle。
依赖再分发以及数据集/模型许可需在最终发布前复审。

## 项目许可证

本次重构未选定新的项目级许可证，所有者必须在发布前作出决定。
这不会免除 Champion 衍生代码或其他上游材料的许可义务。
已包含的上游声明不构成对仓库所有资产的统一授权。
