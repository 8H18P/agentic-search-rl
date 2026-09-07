# 数据构建与筛选

## 权威来源

主数据池是 `data/splits/sft_pool_990_seed20260904.jsonl`。每条记录由 Teacher rollout 的实际执行事件构成；Search action 必须来自 parser 成功的执行事件，observation 必须来自对应的工具响应。不得从 legacy `response` 或原始文本动作重新猜测数据。

## SFT candidate

`data/canonical/sft_candidates_632.jsonl` 是当前仓库实际保留的候选文件，共 632 条。其生成漏斗为：990 条主池 → outcome 通过 → canonical reconstruction → 结构有效 → 冗余规则通过 → 16K 长度规则，最后得到 632 条。脚本不会调用模型或 retriever 修改原始轨迹：

```bash
bash scripts/data/freeze_raw990.sh ...
bash scripts/data/build_canonical_candidates.sh ...
python scripts/data/build_formal_sft_dataset.py \
  --candidates data/canonical/sft_candidates_632.jsonl ...
```

筛选是确定性的，使用 tokenizer 长度、canonical action/observation 对齐、结构完整性和 SmartSearch 风格结果冗余约束。任何长度超限必须由显式 cutoff 策略处理；默认训练 collator 禁止静默截断。

## 其他 split

`remaining890_seed20260904.jsonl` 是从 990 主池拆出的剩余问题，`validation_400.jsonl` 用于独立评测，Process Judge V3 的 held-out 文件位于 `data/splits/process_judge_v3_heldout/`。训练、Judge calibration 与评测 split 必须保持隔离。
