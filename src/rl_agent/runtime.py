"""Stage-1 offline Search Agent runtime; it never mutates policy actions."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class RuntimeConfig:
    name: str = "offline_rl"
    retriever_url: str = "http://127.0.0.1:8766"
    top_k: int = 5
    max_query_budget: int = 8
    max_rounds: int = 10
    snippet_chars: int = 400
    guard_mode: str = "MONITOR"
    visit_enabled: bool = False


def offline_rl_config(**overrides: Any) -> RuntimeConfig:
    config = RuntimeConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@dataclass
class Action:
    kind: str
    query: str = ""
    engine: str | None = None
    answer: str = ""
    strict_protocol_valid: bool = False
    logical_action_valid: bool = False
    raw: str = ""
    reason: str = ""
    proposed_tool: str = ""


def parse_champion_action(raw: str, allow_native_search: bool = True) -> Action:
    """Strict champion JSON plus non-strict compatibility forms."""
    raw = raw.strip()
    payload = None
    strict = False
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.S)
    if match:
        try:
            payload, strict = json.loads(match.group(1)), True
        except json.JSONDecodeError:
            pass
    if payload is None:
        match = re.search(r'\{\s*"name"\s*:\s*"(?:search|visit)".*?\}', raw, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    if isinstance(payload, dict):
        name = str(payload.get("name", "")).lower()
        args = payload.get("arguments", {}) or {}
        if name == "search":
            queries = args.get("query", args.get("queries", []))
            engines = args.get("engine", [])
            if isinstance(queries, str): queries = [queries]
            if isinstance(engines, str): engines = [engines]
            if isinstance(queries, list) and len(queries) == 1 and str(queries[0]).strip():
                engine = str(engines[0]) if isinstance(engines, list) and engines else None
                return Action("search", str(queries[0]), engine, strict_protocol_valid=strict, logical_action_valid=True, raw=raw)
            return Action("invalid", strict_protocol_valid=strict, raw=raw)
        if name == "visit":
            return Action("forbidden_policy_tool", strict_protocol_valid=False, logical_action_valid=False, raw=raw, reason="forbidden_policy_tool", proposed_tool="visit")
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.S)
    if answer:
        return Action("answer", answer=answer.group(1).strip(), strict_protocol_valid=True, logical_action_valid=True, raw=raw)
    compat = re.search(r"(?:^|\n)\s*\(?\s*search\s*(\{.*?\})\s*\)?\s*$", raw, re.I | re.S)
    if compat:
        try:
            args = json.loads(compat.group(1))
            query = args.get("query", "")
            if isinstance(query, list):
                query = query[0] if len(query) == 1 else ""
            engine = args.get("engine")
            if isinstance(engine, list):
                engine = engine[0] if engine else None
            if isinstance(query, str) and query.strip():
                return Action("search", query.strip(), str(engine) if engine else None, logical_action_valid=True, raw=raw)
        except json.JSONDecodeError:
            pass
    # Explicit text forms emitted by Qwen; accept only a quoted argument, not
    # arbitrary prose that merely mentions searching or answering.
    text_search = re.search(r"(?:^|\n|\()\s*search\s*(?:query|exact\s+query)?\s*[:=]\s*[\"']([^\"']+)[\"']", raw, re.I)
    if text_search:
        return Action("search", text_search.group(1).strip(), logical_action_valid=True, raw=raw)
    text_answer = re.search(r"(?:^|\n|\()\s*answer\s*:\s*[\"']([^\"']+)[\"']", raw, re.I)
    if text_answer:
        return Action("answer", answer=text_answer.group(1).strip(), logical_action_valid=True, raw=raw)
    search = re.search(r"<search>\s*(.*?)\s*</search>", raw, re.S) if allow_native_search else None
    if search:
        return Action("search", search.group(1).strip(), None, logical_action_valid=True, raw=raw)
    return Action("invalid", raw=raw)


class OfflineResearchEnv:
    """8766 adapter: query is causal; engine is recorded but non-causal."""
    def __init__(self, url: str, top_k: int = 5, snippet_chars: int = 400):
        self.url, self.top_k, self.snippet_chars = url.rstrip("/"), top_k, snippet_chars

    def search(self, queries: list[str], engines: list[str] | None) -> dict[str, Any]:
        requested = list(engines or [])
        if len(queries) != 1:
            return {"success": False, "observation": "Invalid Search: offline_rl requires exactly one query.", "requested_engine": requested, "executed_backend": "offline_e5", "engine_effective": False, "invalid_action": True, "hits": []}
        query = queries[0]
        import urllib.request
        body = json.dumps({"query": query, "top_n": self.top_k, "return_score": True}).encode()
        request = urllib.request.Request(self.url + "/search", data=body, headers={"Content-Type": "application/json"})
        docs, scores = json.load(urllib.request.urlopen(request, timeout=90))
        hits, lines = [], []
        for rank, (doc, score) in enumerate(zip(docs, scores), 1):
            title, _, body = doc["contents"].partition("\n")
            snippet = body[:self.snippet_chars].replace("\n", " ").strip()
            ident = str(doc["id"])
            lines.append(f"Title: {title}\nURL: local://wiki/{ident}\nSnippet: {snippet}")
            hits.append({"passage_id": ident, "score": float(score), "title": title, "full_contents": doc["contents"], "snippet": snippet})
        return {"success": True, "observation": "\n\n".join(lines), "requested_engine": requested, "executed_backend": "offline_e5", "engine_effective": False, "executed_query": query, "hits": hits}

    def visit_disabled(self) -> dict[str, Any]:
        return {"success": False, "observation": "Visit is disabled in offline_rl Stage-1.", "visit_disabled": True, "executed_backend": "offline_e5", "engine_effective": False, "hits": []}


class MonitorGuard:
    """Diagnostics-only guard.  Its returned action is always byte-for-byte unchanged."""
    def __init__(self): self.queries: list[str] = []; self.observations: list[str] = []; self.stagnant_steps = 0
    def before(self, action: Action) -> dict[str, Any]:
        duplicate = action.query in self.queries if action.kind == "search" else False
        return {"mode": "MONITOR", "duplicate_query": duplicate, "query_similarity": float(duplicate), "proposed_query": action.query, "proposed_engine": action.engine, "executed_query": action.query, "executed_engine": action.engine}
    def after(self, observation: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
        duplicate_observation = observation in self.observations
        self.stagnant_steps = self.stagnant_steps + 1 if duplicate_observation else 0
        self.observations.append(observation)
        diagnostics.update({"duplicate_observation": duplicate_observation, "observation_similarity": float(duplicate_observation), "evidence_gain": not duplicate_observation, "stagnant_steps": self.stagnant_steps, "terminate_suggested": self.stagnant_steps >= 3})
        return diagnostics
    def commit(self, action: Action) -> None:
        if action.kind == "search": self.queries.append(action.query)


class LocalQwenPolicy:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True, low_cpu_mem_usage=True).to(self.device).eval()
    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=512, temperature=None, top_p=None, top_k=None, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()


SYSTEM = """You are a research agent. Use only this Champion protocol. Before answering, issue a search when evidence is needed: <tool_call>{\"name\":\"search\",\"arguments\":{\"query\":[\"exact query\"],\"engine\":[\"google\"]}}</tool_call>. After tool results, answer with <answer>concise answer</answer>. Exactly one query per Search."""


def run_question(policy: LocalQwenPolicy, question: str, config: RuntimeConfig, trajectory_path: Path | None = None) -> dict[str, Any]:
    env, guard = OfflineResearchEnv(config.retriever_url, config.top_k, config.snippet_chars), MonitorGuard()
    messages = [{"role":"system","content":SYSTEM}, {"role":"user","content":question}]
    records, budget, final, term, searches = [], 0, "", "max_rounds", 0
    started = time.perf_counter()
    for round_idx in range(1, config.max_rounds + 1):
        raw = policy.generate(messages); action = parse_champion_action(raw, allow_native_search=getattr(config, "allow_native_search", True))
        # The base instruct model sometimes emits a plain final sentence after
        # grounded tool evidence. Keep it non-strict, but end the episode rather
        # than turning an already-grounded answer into a spurious invalid loop.
        if action.kind == "invalid" and searches and raw.strip() and "search" not in raw.lower():
            action = Action("answer", answer=raw.strip(), logical_action_valid=True, raw=raw)
        record: dict[str, Any] = {"round":round_idx,"raw_model_output":raw,"parsed_action":asdict(action),"strict_protocol_valid":action.strict_protocol_valid,"logical_action_valid":action.logical_action_valid,"query_budget_used":budget}
        if action.kind == "answer": final, term = action.answer, "answer"; record["final_answer"] = final; records.append(record); break
        diag = guard.before(action); record["guard_diagnostics"] = diag
        if action.kind == "search":
            if budget >= config.max_query_budget: result = {"success":False,"observation":"Query budget exhausted.","hits":[]}; term = "query_budget"
            else: budget += 1; searches += 1; result = env.search([action.query], [action.engine] if action.engine else [])
            guard.commit(action)
        elif action.kind == "forbidden_policy_tool": result = {"success":False,"observation":"Forbidden policy tool: visit is not allowed.","forbidden_policy_tool":True,"reason":"forbidden_policy_tool","hits":[]}
        else: result = {"success":False,"observation":"Invalid action. Use Champion Search or Answer.","hits":[]}
        record.update({k:v for k,v in result.items() if k != "full_contents"}); record["guard_diagnostics"] = guard.after(result["observation"], diag); record["query_budget_used"] = budget; records.append(record)
        messages.extend([{"role":"assistant","content":raw},{"role":"tool","content":result["observation"]}])
        if term == "query_budget": break
    if trajectory_path:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records)+"\n", encoding="utf-8")
        trajectory_path.with_suffix(".txt").write_text("\n\n".join(f"ROUND {r['round']}\n{r['raw_model_output']}\n{r.get('observation','')}" for r in records), encoding="utf-8")
    return {"question":question,"prediction":final,"num_queries":budget,"num_rounds":len(records),"strict_protocol_valid":all(r["strict_protocol_valid"] for r in records),"logical_action_valid":all(r["logical_action_valid"] for r in records),"duplicate_query_count":sum(bool(r.get("guard_diagnostics",{}).get("duplicate_query")) for r in records),"budget_hit":term=="query_budget","answer_before_search":bool(final and not searches),"requested_engines":[(r.get("requested_engine") or "unspecified") for r in records if "requested_engine" in r],"executed_backends":[r.get("executed_backend") for r in records if "executed_backend" in r],"engine_effective":False,"termination_reason":term,"latency":time.perf_counter()-started,"records":records}
