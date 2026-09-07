# 数据构建与筛选

## 权威来源

主数据池通过配置绑定到 `data/splits/` 下的 JSONL，使用者从有权使用的 Teacher rollout 通过 `scripts/data/` 构建。每条记录由实际执行事件构成；Search action 必须来自 parser 成功的执行事件，observation 必须来自对应的工具响应。不得从 legacy `response` 或原始文本动作重新猜测数据。

## SFT candidate

Canonical SFT candidate 由主池依次经过 outcome、canonical reconstruction、结构、冗余与长度规则生成。脚本不会调用模型或 retriever 修改原始轨迹：

```bash
bash scripts/data/freeze_raw990.sh ...
bash scripts/data/build_canonical_candidates.sh ...
python scripts/data/build_formal_sft_dataset.py \
  --candidates /path/to/sft_candidates.jsonl ...
```

筛选是确定性的，使用 tokenizer 长度、canonical action/observation 对齐、结构完整性和 SmartSearch 风格结果冗余约束。任何长度超限必须由显式 cutoff 策略处理；默认训练 collator 禁止静默截断。

## 其他 split

训练、Judge calibration 与评测 split 分别通过配置绑定并保持隔离；运行 manifest 记录实际路径、排除关系和 hash。
