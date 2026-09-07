"""Alias-aware post-hoc evaluator for canonical baseline splits."""
from __future__ import annotations
import collections, json, re
from pathlib import Path

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")

def normalize(text):
    text = str(text or "").lower().strip()
    text = _PUNCT.sub("", text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())

def aliases(value):
    if isinstance(value, str): return [value]
    if isinstance(value, list): return [str(x) for x in value if str(x).strip()]
    return []

def canonicalize_final_answer(prediction):
    """Use the last complete LaTex \boxed{...} payload when present."""
    text = str(prediction or "").strip()
    candidates = []
    start = 0
    while True:
        marker = text.find("\\boxed", start)
        if marker < 0:
            break
        brace = text.find("{", marker + len("\\boxed"))
        if brace < 0:
            start = marker + len("\\boxed")
            continue
        depth = 0
        for pos in range(brace, len(text)):
            if text[pos] == "{": depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    value = text[brace + 1:pos].strip()
                    if value:
                        candidates.append(value)
                    start = pos + 1
                    break
        else:
            start = brace + 1
    return candidates[-1] if candidates else text

def em_f1(prediction, golds):
    pred = normalize(canonicalize_final_answer(prediction))
    pred_tokens = pred.split()
    best_em = best_f1 = 0.0
    for gold in aliases(golds):
        truth = normalize(gold); truth_tokens = truth.split()
        best_em = max(best_em, float(pred == truth))
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        count = sum(common.values())
        if not pred_tokens and not truth_tokens: score = 1.0
        elif not pred_tokens or not truth_tokens: score = 0.0
        else:
            precision, recall = count / len(pred_tokens), count / len(truth_tokens)
            score = 0.0 if not count else 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, score)
    return best_em, best_f1

def evaluate(results_path: Path, split_path: Path, output_path: Path):
    split = {row["id"]: row for row in (json.loads(x) for x in split_path.read_text(encoding="utf-8").splitlines() if x.strip())}
    rows = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line); canonical = split.get(item["question_id"])
        if canonical is None: raise KeyError(f"missing canonical id {item['question_id']}")
        golds = aliases(canonical.get("golden_answers"))
        item["evaluation_prediction"] = canonicalize_final_answer(item.get("prediction", ""))
        em, f1 = em_f1(item["evaluation_prediction"], golds)
        # Historical records used [] for a missing engine.  Preserve intent as
        # an explicit value rather than a nested empty-list pseudo-engine.
        item["requested_engines"] = [
            engine if engine else "unspecified"
            for engine in item.get("requested_engines", [])
        ]
        item.update({"golden_answers": golds, "answer_em": em, "answer_f1": f1})
        item.pop("gold_answer", None); rows.append(item)
    output_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    n = len(rows); searched = [r for r in rows if r.get("num_queries", 0) > 0]
    terms = collections.Counter(r.get("termination_reason") for r in rows)
    summary = {"num_questions":n,"EM":sum(r["answer_em"] for r in rows)/n,"F1":sum(r["answer_f1"] for r in rows)/n,"search_rate":len(searched)/n,"no_search_rate":1-len(searched)/n,"avg_queries_all":sum(r.get("num_queries",0) for r in rows)/n,"avg_queries_when_searching":sum(r.get("num_queries",0) for r in searched)/len(searched) if searched else 0,"answer_before_search_rate":sum(bool(r.get("answer_before_search")) for r in rows)/n,"strict_protocol_valid_rate":sum(bool(r.get("strict_protocol_valid")) for r in rows)/n,"logical_action_valid_rate":sum(bool(r.get("logical_action_valid")) for r in rows)/n,"max_rounds_failure_rate":terms["max_rounds"]/n,"budget_hit_rate":sum(bool(r.get("budget_hit")) for r in rows)/n,"duplicate_query_rate":sum(bool(r.get("duplicate_query_count")) for r in rows)/n,"termination_reason_distribution":dict(terms)}
    return rows, summary
