"""Native SmartSearch rollout.

Template and stop strings follow src/smartsearch/src/re_call/inference/re_call.py
(ReCall.init_prompt, cat_tool_results and run), rather than the Champion runtime.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

from rl_agent.protocol.smartsearch import first_action, parse
from rl_agent.runtime import MonitorGuard, OfflineResearchEnv

SYSTEM = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""


class _StopOnText:
    def __init__(self, tokenizer, prompt_length, stop_strings):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_strings = tuple(stop_strings)

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=False)
        return any(stop in text for stop in self.stop_strings)


class SmartSearchPolicy:
    """HF implementation of the official ReCall raw Qwen prompt and stop policy."""
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
        self.torch, self.StoppingCriteria, self.StoppingCriteriaList = torch, StoppingCriteria, StoppingCriteriaList
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
            local_files_only=True, low_cpu_mem_usage=True,
        ).to(self.device).eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        criterion = self._criterion(inputs.input_ids.shape[-1])
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=8192,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.StoppingCriteriaList([criterion]),
            )
        return self.tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    def _criterion(self, prompt_length):
        outer = self
        class Criterion(outer.StoppingCriteria):
            def __init__(self):
                self.impl = _StopOnText(outer.tokenizer, prompt_length, ("</search>", "</answer>"))
            def __call__(self, input_ids, scores, **kwargs):
                return self.impl(input_ids, scores, **kwargs)
        return Criterion()


def init_prompt(question: str) -> str:
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>"


def run_question(policy, question, config, trajectory_path: Path | None = None):
    env, guard = OfflineResearchEnv(config.retriever_url, config.top_k, config.snippet_chars), MonitorGuard()
    prompt, records, budget, searches, final, term = init_prompt(question), [], 0, 0, "", "max_rounds"
    started = time.perf_counter()
    for round_idx in range(1, config.max_rounds + 1):
        generated = policy.generate(prompt)
        raw = first_action("<think>" + generated)
        action = parse(raw)
        record = {"round": round_idx, "raw_model_output": raw, "parsed_action": asdict(action), "strict_protocol_valid": action.strict_protocol_valid, "logical_action_valid": action.logical_action_valid, "query_budget_used": budget}
        if action.kind == "answer":
            final, term = action.value, "answer"
            record["final_answer"] = final
            records.append(record)
            break
        diag = guard.before(type("Action", (), {"kind": action.kind, "query": action.value, "engine": None})())
        if action.kind == "search":
            if budget >= config.max_query_budget:
                result, term = {"success": False, "observation": "Query budget exhausted.", "hits": []}, "query_budget"
            else:
                budget += 1; searches += 1
                result = env.search([action.value], [])
                guard.commit(type("Action", (), {"kind": "search", "query": action.value})())
        else:
            result = {"success": False, "observation": "Invalid SmartSearch action. Use <search> or <answer>.", "hits": []}
        record.update(result)
        record["guard_diagnostics"] = guard.after(result["observation"], diag)
        record["query_budget_used"] = budget
        records.append(record)
        prompt += generated
        prompt += f"<result>\n{result['observation']}\n</result>\n"
        if term == "query_budget":
            break
    if trajectory_path:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")
    return {"question": question, "prediction": final, "num_queries": budget, "num_rounds": len(records), "strict_protocol_valid": all(x["strict_protocol_valid"] for x in records), "logical_action_valid": all(x["logical_action_valid"] for x in records), "duplicate_query_count": sum(bool(x.get("guard_diagnostics", {}).get("duplicate_query")) for x in records), "budget_hit": term == "query_budget", "answer_before_search": bool(final and not searches), "requested_engines": [], "executed_backends": [x.get("executed_backend") for x in records if x.get("executed_backend")], "engine_effective": False, "termination_reason": term, "latency": time.perf_counter() - started, "records": records}
