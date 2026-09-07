"""Canonical, execution-grounded DPO dataset and role-aware collator.

The DPO schema keeps a shared message prefix separate from the two
counterfactual continuations.  Only assistant messages in a continuation are
eligible for policy loss; user/tool-response messages remain context.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


_BLOCK = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)


class CanonicalDPODataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.open(encoding="utf-8") if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[int], list[tuple[int, int]]]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    return text, list(encoded["input_ids"]), list(encoded.get("offset_mapping", []))


def _encode_with_role_mask(tokenizer: Any, prompt: list[dict[str, str]], continuation: list[dict[str, str]]) -> dict[str, Any]:
    messages = prompt + continuation
    text, ids, offsets = _tokenize_messages(tokenizer, messages)
    prompt_count = len(prompt)
    labels = [-100] * len(ids)
    assistant_tokens = 0
    tool_response_tokens = 0
    blocks = list(_BLOCK.finditer(text))
    if len(blocks) != len(messages):
        raise ValueError(f"chat-template block count mismatch: blocks={len(blocks)} messages={len(messages)}")
    for message_index, (message, block) in enumerate(zip(messages, blocks)):
        body_start, body_end = block.span(2)
        overlaps = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < body_end and end > body_start
        ]
        if message_index >= prompt_count and message.get("role") == "assistant":
            for index in overlaps:
                labels[index] = ids[index]
            assistant_tokens += len(overlaps)
        if message_index >= prompt_count and message.get("role") == "user" and "<tool_response>" in str(message.get("content", "")):
            tool_response_tokens += sum(labels[index] != -100 for index in overlaps)
    if assistant_tokens <= 0:
        raise ValueError("continuation has no supervised assistant tokens")
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": [1] * len(ids),
        "text": text,
        "prompt_tokens": len(_tokenize_messages(tokenizer, prompt)[1]),
        "assistant_tokens": assistant_tokens,
        "tool_response_tokens": tool_response_tokens,
    }


class DPORoleAwareCollator:
    """Collate chosen/rejected branches without truncation."""

    def __init__(self, tokenizer: Any, max_length: int = 16384):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        chosen = []
        rejected = []
        for row in batch:
            prompt = row["prompt"]
            c = _encode_with_role_mask(self.tokenizer, prompt, row["chosen"])
            r = _encode_with_role_mask(self.tokenizer, prompt, row["rejected"])
            if len(c["input_ids"]) > self.max_length or len(r["input_ids"]) > self.max_length:
                raise ValueError(
                    "silent truncation blocked: "
                    f"{row.get('pair_id', '<unknown>')} chosen={len(c['input_ids'])} "
                    f"rejected={len(r['input_ids'])} max={self.max_length}"
                )
            chosen.append(c)
            rejected.append(r)

        def pad(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            width = max(len(record["input_ids"]) for record in records)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            input_ids = torch.full((len(records), width), int(pad_id), dtype=torch.long)
            labels = torch.full((len(records), width), -100, dtype=torch.long)
            attention = torch.zeros((len(records), width), dtype=torch.long)
            for index, record in enumerate(records):
                length = len(record["input_ids"])
                input_ids[index, :length] = torch.tensor(record["input_ids"], dtype=torch.long)
                labels[index, :length] = torch.tensor(record["labels"], dtype=torch.long)
                attention[index, :length] = 1
            return input_ids, attention, labels

        c_ids, c_attention, c_labels = pad(chosen)
        r_ids, r_attention, r_labels = pad(rejected)
        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_attention,
            "chosen_labels": c_labels,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_attention,
            "rejected_labels": r_labels,
            "pair_ids": [row.get("pair_id") for row in batch],
            "chosen_lengths": [len(record["input_ids"]) for record in chosen],
            "rejected_lengths": [len(record["input_ids"]) for record in rejected],
            "prompt_lengths": [record["prompt_tokens"] for record in chosen],
            "chosen_supervised_tokens": [record["assistant_tokens"] for record in chosen],
            "rejected_supervised_tokens": [record["assistant_tokens"] for record in rejected],
            "chosen_tool_response_tokens": [record["tool_response_tokens"] for record in chosen],
            "rejected_tool_response_tokens": [record["tool_response_tokens"] for record in rejected],
        }


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum policy log-probabilities over unmasked continuation labels."""
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    safe_labels = shifted_labels.clamp_min(0)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * shifted_labels.ne(-100)).sum(dim=-1)
