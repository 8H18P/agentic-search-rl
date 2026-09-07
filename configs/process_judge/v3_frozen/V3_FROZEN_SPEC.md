# V3 Query-Level Process Judge Frozen Specification

## 1. Freeze purpose

This snapshot defines the exact input-isolated Judge implementation. Any later prompt, parser, schema, input-boundary, or scoring change requires an explicitly versioned successor.

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

## 4. Versioning rule

- V3 prompts may not be modified in place.
- V3 parsers or schemas may not be modified in place.
- Evaluation data and generated scores are stored outside the source repository.
- Any behavior change must receive a new version and manifest.

The authoritative executable definitions are the source files and symbols recorded in `V3_FREEZE_MANIFEST.json`. The prompt text files here are human-readable placeholder renderings of those executed prompt functions.
