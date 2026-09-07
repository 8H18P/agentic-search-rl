"""Minimal DashScope OpenAI-compatible policy backend for teacher rollouts."""
from __future__ import annotations
import copy
import os
import re
import requests
import time

def normalize_tool_call_lexically(text: str):
    rules = []
    def repl_open(m):
        rules.append("tool_call_missing_gt")
        return "<tool_call>"
    def repl_close(m):
        rules.append("tool_call_close_missing_gt")
        return "</tool_call>"
    text2 = re.sub(r"<tool_call[ \t]*>", "<tool_call>", text)
    if text2 != text:
        rules.append("tool_call_tag_whitespace")
    before_close = text2
    text2 = re.sub(r"</tool_call[ \t]*>", "</tool_call>", text2)
    if text2 != before_close:
        rules.append("tool_call_close_tag_whitespace")
    before_missing = text2
    text2 = re.sub(r"<tool_call(?=[ \t]*\n)", repl_open, text2)
    text2 = re.sub(r"</tool_call(?=[ \t]*\n)", repl_close, text2)
    if text2 != before_missing and not rules:
        rules.append("tool_call_missing_gt")
    return text2, list(dict.fromkeys(rules))

class DashScopePolicy:
    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 temperature: float = 0.4, max_tokens: int = 8192):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.responses: list[dict] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        request_start = time.time()
        request_snapshot = {
            "timestamp": request_start,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": None,
            "extra_body": {"enable_thinking": True},
            "messages_snapshot": copy.deepcopy(messages),
        }
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "enable_thinking": True}
        response = None
        last_exc = None
        retry_count = 0
        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
                    json=payload, timeout=(15, 180),
                )
                response.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= 2:
                    raise
                retry_count += 1
                time.sleep(2 ** attempt)
        request_end = time.time()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content") or ""
        raw_content = message.get("content") or ""
        normalized, normalization_rules = normalize_tool_call_lexically(raw_content)
        metadata = {
            "request_snapshot": request_snapshot,
            "requested_model": self.model,
            "returned_model": data.get("model", self.model),
            "reasoning_content": reasoning,
            "content": raw_content,
            "raw_api_content": raw_content,
            "normalized_content": normalized,
            "normalization_rule_id": normalization_rules[0] if normalization_rules else None,
            "normalization_applied": bool(normalization_rules),
            "normalization_rule_ids": normalization_rules,
            "finish_reason": choice.get("finish_reason"),
            "usage": copy.deepcopy(data.get("usage")),
            "request_start_time": request_start,
            "request_end_time": request_end,
            "latency_seconds": request_end - request_start,
            "attempt_index": retry_count + 1,
            "api_retry_count": retry_count,
            "http_status": response.status_code if response is not None else None,
        }
        self.responses.append(copy.deepcopy(metadata))
        if reasoning and "<think>" not in normalized.lower():
            return "<think>\n" + reasoning + "\n</think>\n" + normalized
        return normalized
