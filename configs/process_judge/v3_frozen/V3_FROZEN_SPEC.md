# V3 Query-Level Process Judge Frozen Specification

## 1. Freeze purpose

V3 targeted calibration is complete. This snapshot freezes the exact input-isolated Judge implementation before final held-out validation. Once held-out validation begins, its results must not be used to modify V3 in place. Any later prompt, parser, schema, input-boundary, or scoring change requires an explicitly versioned V4.

## 2. Architecture

The Intent Judge receives only the original question, gold answer, authoritative history before the current Search action, current action reasoning, and current executed query. It cannot receive the current result, sibling results, future information, teacher final answer, or raw textual action.

The Retrieval Judge receives only the current executed query and its authoritative query-specific result section. It cannot receive the original question, gold answer, history, reasoning, sibling queries/results, future information, teacher final answer, or raw textual action.

The model does not emit a combined process answer. Composition is deterministic:

```python
process_score = int(intent_score == 1 and retrieval_score == 1)
```

Efficiency/redundancy is an independent sidecar diagnostic. It cannot change Intent, Retrieval, or process score.

## 3. SmartSearch boundary

SmartSearch's original `process_reward.py` uses one joint LLM Judge and defines usefulness around a necessary/actionable query whose result answers that query intent, with strict entity matching. Two-channel Intent/Retrieval input isolation is a Champion-specific engineering adaptation used to prevent long-history, multi-query, original-question, and downstream-constraint contamination. This specification does not claim that upstream SmartSearch uses two independent LLM calls.

## 4. Calibration evidence

Targeted calibration used qwen3.5-flash with temperature 0, thinking disabled, and 384 maximum output tokens.

- Retrieval: 37/37 API/parse success.
- Intent: 50/50 API/parse success.
- Known unambiguous retrieval regressions: 6/6.
- Post-hoc retrieval adjudication: 35/36, with one ambiguous item excluded.
- Observed downstream contamination: 0.
- Natural positives: 20/20; controlled synthetic negatives: 20/20; reasoning-ablated positives: 10/10.

These targeted-set results cannot be extrapolated directly to the full 14,279-query distribution. V3 is ready only for one fresh, frozen held-out validation; it is not yet approved for the full Judge run.

## 5. Immutable-after-validation rule

After final held-out validation starts:

- V3 prompts may not be modified in place.
- V3 parsers or schemas may not be modified in place.
- Validation human labels may not be changed.
- Any post-hoc adjudication must be stored separately and must preserve the pre-validation labels.
- V3 results may select or reject this frozen version, but may not tune it.

The authoritative executable definitions are the source files and symbols recorded in `V3_FREEZE_MANIFEST.json`. The prompt text files here are human-readable placeholder renderings of those executed prompt functions.
