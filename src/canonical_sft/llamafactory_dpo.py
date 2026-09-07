"""Canonical data bridge to pinned LLaMA-Factory; preflight only.

No LLaMA-Factory dataset processor/template is used. Training stays gated until
real-model TRL/LLaMA-Factory parity and checkpoint handoff have passed.
"""
from __future__ import annotations

import torch

from .trl_dpo import CanonicalTRLDataCollator


class CanonicalLlamaFactoryCollator:
    """Same tokens and role mask as TRL, with LF's concatenated labels contract."""

    def __init__(self, tokenizer, max_length: int):
        self.canonical = CanonicalTRLDataCollator(tokenizer, max_length)

    def __call__(self, rows):
        encoded = self.canonical(rows)
        labels = encoded["input_ids"].clone()
        labels.masked_fill_(encoded["completion_mask"].eq(0), -100)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }


def validate_explicit_reference(model, ref_model):
    """Refuse LF's disable_adapter fallback and accidentally shared parameters."""
    if ref_model is None or ref_model is model:
        raise ValueError("provide a distinct frozen Base + initial SFT adapter reference")
    for name, candidate in (("policy", model), ("reference", ref_model)):
        configs = getattr(candidate, "peft_config", {})
        if len(configs) != 1:
            raise ValueError(f"{name} must have exactly one PEFT adapter, no stacking")
    if any(p.requires_grad for p in ref_model.parameters()):
        raise ValueError("reference must be completely frozen")
    policy_ids = {id(p) for p in model.parameters()}
    if any(id(p) in policy_ids for p in ref_model.parameters()):
        raise ValueError("explicit reference must not share parameter objects with policy")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("only the resumed policy LoRA parameters may be trainable")
    # This entry is initial-state preflight, not a resume-after-update loader.
    policy_adapter = {n: p for n, p in model.named_parameters() if "lora_" in n}
    ref_adapter = {n: p for n, p in ref_model.named_parameters() if "lora_" in n}
    if policy_adapter.keys() != ref_adapter.keys():
        raise ValueError("initial policy/reference adapter keys differ")
    for name, parameter in policy_adapter.items():
        left = parameter.detach().to(device="cpu", dtype=torch.float32)
        right = ref_adapter[name].detach().to(device="cpu", dtype=torch.float32)
        if not torch.equal(left, right):
            raise ValueError(f"initial policy/reference adapter weights differ: {name}")


def build_preflight_trainer_class():
    """Use upstream LF loss/forward, never implement another DPO objective."""
    from llamafactory.train.dpo.trainer import CustomDPOTrainer

    class CanonicalLlamaFactoryPreflightTrainer(CustomDPOTrainer):
        def __init__(self, *, model, ref_model, **kwargs):
            validate_explicit_reference(model, ref_model)
            super().__init__(model=model, ref_model=ref_model, **kwargs)

        def train(self, *args, **kwargs):
            raise RuntimeError("preflight-only bridge: real-model parity gate not yet closed")

    return CanonicalLlamaFactoryPreflightTrainer
