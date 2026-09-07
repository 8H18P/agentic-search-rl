# Process Judge / PRM

当前过程奖励使用冻结的 Process Judge V3：分别对每个 authoritative query/result pair 计算 Intent 与 Retrieval，按冻结规则得到 `process_score`。实现位于 `scripts/prm/`，运行时奖励桥接位于 `src/online_grpo/reward.py`。

```bash
bash scripts/prm/audit_v3.sh --repo . --artifact-dir <audit-dir> --output <report>
bash scripts/prm/score_v3.sh <judge-args>
```

V3 prompt、输出 schema 和 manifest 位于 `configs/process_judge/v3_frozen/`，不得在清理或发布时改写其语义。query-level 结果必须保留 trajectory/action/query 索引与输入 hash；正式规模可在后续用冻结标签训练本地轻量 PRM，本仓库当前不自动训练 PRM。
