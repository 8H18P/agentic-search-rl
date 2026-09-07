# Process Judge / PRM

当前过程奖励使用冻结的 Process Judge V3：分别对每个 authoritative query/result pair 计算 Intent 与 Retrieval，按冻结规则得到 `process_score`。实现位于 `scripts/prm/`，运行时奖励桥接位于 `src/online_grpo/reward.py`。

```bash
bash scripts/prm/score_v3.sh <judge-args>
```

V3 prompt、输出 schema 和 manifest 位于 `configs/process_judge/v3_frozen/`。本项目中的 PRM 指冻结 LLM Judge 提供的 query-level process reward 接口。query-level 结果保留 trajectory/action/query 索引与输入 hash，评分产物写入仓库外目录。
