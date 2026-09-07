"""Token-exact per-generation capture, independent of training stage."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility promised by pyproject.toml
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self):
            return self.value

import torch


class TokenOrigin(StrEnum):
    POLICY_GENERATED = "POLICY_GENERATED"
    ENVIRONMENT_OBSERVATION = "ENVIRONMENT_OBSERVATION"
    RUNTIME_INJECTED = "RUNTIME_INJECTED"
    EXTERNAL_REFINER = "EXTERNAL_REFINER"
    PROMPT = "PROMPT"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require_exact(expected, actual, label="replay_prefix"):
    if expected != actual:
        pos = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
        raise ValueError(json.dumps({"gate": label, "first_difference": pos,
            "expected_length": len(expected), "actual_length": len(actual),
            "difference_type": "length" if pos == min(len(expected), len(actual)) else "token_id",
            "expected_id": expected[pos] if pos < len(expected) else None,
            "actual_id": actual[pos] if pos < len(actual) else None}))


def input_origins(tokenizer, text, messages, actual_ids):
    encoded = tokenizer(text, return_offsets_mapping=True)
    require_exact(actual_ids, encoded["input_ids"], "capture_tokenizer_offsets")
    blocks = list(re.finditer(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", text, re.S))
    if len(blocks) != len(messages):
        raise ValueError("message/token provenance block count mismatch")
    spans = []
    for i, (block, msg) in enumerate(zip(blocks, messages)):
        start, end = block.span(2)
        kind = TokenOrigin.PROMPT if i < 2 else TokenOrigin.RUNTIME_INJECTED
        if msg.get("role") == "assistant":
            kind = TokenOrigin.POLICY_GENERATED
        if msg.get("token_origin") == TokenOrigin.EXTERNAL_REFINER:
            kind = TokenOrigin.EXTERNAL_REFINER
        spans.append((start, end, str(kind)))
        if msg.get("role") in ("user", "tool") and "<tool_response>" in block.group(2):
            a = text.index("<tool_response>", start, end)
            b = text.find("</tool_response>", a, end)
            if b < 0:
                raise ValueError("unclosed authoritative observation")
            spans.append((a, b + len("</tool_response>"), str(TokenOrigin.ENVIRONMENT_OBSERVATION)))
    kinds = []
    for a, b in encoded["offset_mapping"]:
        matches = [kind for start, end, kind in spans if a >= start and b <= end and b > a]
        kinds.append(matches[-1] if matches else str(TokenOrigin.RUNTIME_INJECTED))
    return kinds


class RolloutCapture:
    """Sink for HFPolicyBackend plus observer; used with sequential Champion runs."""

    def __init__(self, tokenizer, observer, trajectory_id, group_id, question_id, seed, max_seq_length):
        self.tokenizer, self.observer = tokenizer, observer
        self.trajectory_id, self.group_id, self.question_id = trajectory_id, group_id, question_id
        self.seed, self.max_seq_length = seed, max_seq_length
        self.records, self.events = [], []
        self.round_idx = 0
        self.pending = None

    def record_event(self, event_type, payload=None):
        payload = copy.deepcopy(payload or {})
        if event_type == "round_request":
            self.round_idx = payload.get("round_idx", self.round_idx)
        if event_type == "policy_output_processed" and self.records:
            row = self.records[-1]
            row["runtime_accepted_text"] = payload["runtime_accepted_text"]
            row["transformation_type"] += payload.get("transformation_type", [])
            row["transformation_applied"] = bool(row["transformation_type"])
            row["generation_path"] = payload["generation_path"]
            row["accepted_text_observed"] = True
        self.events.append({"event_type": event_type, "payload": payload})
        self.observer.record_event(event_type, payload)

    def before_generation(self, batch, prompt, messages, kwargs, model):
        ids = batch["input_ids"][0].detach().cpu().tolist()
        if len(ids) + kwargs["max_new_tokens"] > self.max_seq_length:
            raise ValueError("resource budget exceeded before generation; no truncation performed")
        if not kwargs["do_sample"]:
            raise ValueError("online GRPO capture requires stochastic generation")
        settings = model.generation_config.to_dict()
        settings.update(kwargs)
        # Pure temperature sampling keeps the replay likelihood well-defined.
        if settings.get("top_k", 0) != 0 or settings.get("top_p", 1.0) != 1.0 or settings.get("repetition_penalty", 1.0) != 1.0:
            raise ValueError("online capture supports temperature sampling without extra logits processors")
        self.pending = {
            "trajectory_id": self.trajectory_id, "group_id": self.group_id, "question_id": self.question_id,
            "generation_idx": len(self.records), "round_idx": self.round_idx, "seed": self.seed,
            "input_ids": ids, "input_length": len(ids), "actual_input_sha256": digest(ids),
            "input_token_origins": input_origins(self.tokenizer, prompt, messages, ids),
            "generation_temperature": settings.get("temperature", 1.0),
            "do_sample": settings["do_sample"], "top_p": settings.get("top_p", 1.0),
            "top_k": settings.get("top_k", 0), "max_new_tokens": settings["max_new_tokens"],
            "eos_token_id": settings.get("eos_token_id"),
            "cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda_rng_sha256": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest() if torch.cuda.is_available() else None,
            "transformation_type": [], "accepted_text_observed": False,
        }

    def after_generation(self, generated, raw_text, accepted_text):
        row = self.pending
        ids = generated.detach().cpu().tolist()
        row.update(generated_ids=ids, generated_length=len(ids), actual_generated_sha256=digest(ids),
                   raw_generated_text=raw_text, runtime_accepted_text=accepted_text)
        if raw_text != accepted_text:
            row["transformation_type"].append("strip")
        row["transformation_applied"] = bool(row["transformation_type"])
        row["generation_budget_reached"] = len(ids) == row["max_new_tokens"]
        self.records.append(row)
        self.pending = None


def replay_segment(record):
    """State is copied from actual sampling IDs, never from canonical text."""
    state = list(record["input_ids"])
    actions = list(record["generated_ids"])
    if not actions:
        raise ValueError("zero sampled action tokens")
    return {"state_ids": state, "action_ids": actions,
            "input_ids": state + actions,
            "attention_mask": [1] * (len(state) + len(actions)),
            "loss_mask": [0] * len(state) + [1] * len(actions),
            "token_origins": record["input_token_origins"] + [str(TokenOrigin.POLICY_GENERATED)] * len(actions),
            "temperature": record["generation_temperature"], "capture": record}


def audit_segment(segment):
    record = segment["capture"]
    n = len(record["input_ids"])
    require_exact(record["input_ids"], segment["state_ids"], "objective_state")
    require_exact(record["generated_ids"], segment["action_ids"], "objective_action")
    require_exact(record["input_ids"], segment["input_ids"][:n])
    require_exact(record["generated_ids"], segment["input_ids"][n:], "sampled_action")
    if digest(record["input_ids"]) != record["actual_input_sha256"] or digest(record["generated_ids"]) != record["actual_generated_sha256"]:
        raise ValueError("capture identity changed")
    if segment["loss_mask"] != [0]*n + [1]*record["generated_length"]:
        raise ValueError("role/environment loss mask mismatch")
    if segment["attention_mask"] != [1]*len(segment["input_ids"]):
        raise ValueError("environment context removed from attention")
    return {"trajectory_id": record["trajectory_id"], "generation_idx": record["generation_idx"],
            "round_idx": record["round_idx"], "exact_prefix_match": True, "exact_action_match": True,
            "environment_mask_pass": True, "state_tokens": n, "policy_loss_tokens": record["generated_length"],
            "state_origin_counts": dict(Counter(record["input_token_origins"]))}


def audit_flat(records):
    issues = []
    for previous, following in zip(records, records[1:]):
        expected = previous["input_ids"] + previous["generated_ids"]
        try:
            require_exact(expected, following["input_ids"][:len(expected)], "flat_transcript_prefix")
        except ValueError as exc:
            issues.append({"generation_idx": following["generation_idx"], "difference": json.loads(str(exc))})
    return {"flat_transcript_replay_supported": not issues, "transition_count": max(0, len(records)-1), "failures": issues}
