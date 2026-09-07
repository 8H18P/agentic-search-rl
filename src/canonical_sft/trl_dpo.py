"""Thin TRL adapter for canonical, role-aware DPO samples.

This module deliberately leaves canonical message reconstruction to
``canonical_sft.dpo``.  It only converts that audited representation into the
tensor contract expected by TRL's mature DPO objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .dpo import DPORoleAwareCollator


@dataclass
class DPOSmokeConfig:
    model_path: str
    tokenizer_path: str
    sft_adapter_path: str
    train_dataset_path: str
    output_dir: str
    model_id: str
    model_revision: str
    selected_pair_ids: list[str] = field(default_factory=list)
    selection_reason: str = "all_pairs"
    max_seq_length: int = 16384
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-7
    max_steps: int = 4
    beta: float = 0.1
    loss_type: str = "sigmoid"
    optim: str = "adamw_torch_fused"
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    use_liger_kernel: bool = True
    activation_offloading: bool = False
    torch_empty_cache_steps: int | None = 1
    cuda_allocator_conf: str = "expandable_segments:True"
    dtype: str = "bfloat16"
    attention_backend: str = "sdpa"
    report_to: list[str] = field(default_factory=list)
    seed: int = 20260905

    def validate(self) -> None:
        for value, label in (
            (self.max_seq_length, "max_seq_length"),
            (self.per_device_train_batch_size, "per_device_train_batch_size"),
            (self.gradient_accumulation_steps, "gradient_accumulation_steps"),
            (self.max_steps, "max_steps"),
        ):
            if int(value) <= 0:
                raise ValueError(f"{label} must be positive")
        if self.beta <= 0 or self.learning_rate <= 0:
            raise ValueError("beta and learning_rate must be positive")
        if self.loss_type != "sigmoid":
            raise ValueError("this smoke entrypoint requires standard sigmoid DPO")
        for path, label in (
            (self.model_path, "model_path"),
            (self.tokenizer_path, "tokenizer_path"),
            (self.sft_adapter_path, "sft_adapter_path"),
            (self.train_dataset_path, "train_dataset_path"),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")


class CanonicalTRLDataCollator:
    """Adapt role-aware chosen/rejected tensors to TRL's concatenated batch.

    Output order is all chosen rows followed by all rejected rows.  The
    completion mask is derived from the already-audited role-aware labels, so
    shared prompt and tool observations remain conditioning-only.
    """

    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.role_collator = DPORoleAwareCollator(tokenizer, max_length=max_length)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.role_collator(rows)
        chosen_ids = encoded["chosen_input_ids"]
        chosen_attention = encoded["chosen_attention_mask"]
        chosen_labels = encoded["chosen_labels"]
        rejected_ids = encoded["rejected_input_ids"]
        rejected_attention = encoded["rejected_attention_mask"]
        rejected_labels = encoded["rejected_labels"]

        width = max(chosen_ids.shape[1], rejected_ids.shape[1])
        if width > self.max_length:
            raise ValueError(f"silent truncation blocked: batch width={width} max={self.max_length}")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        def repad(tensor: torch.Tensor, fill: int) -> torch.Tensor:
            if tensor.shape[1] == width:
                return tensor
            output = torch.full((tensor.shape[0], width), fill, dtype=tensor.dtype)
            output[:, : tensor.shape[1]] = tensor
            return output

        chosen_ids = repad(chosen_ids, int(pad_id))
        rejected_ids = repad(rejected_ids, int(pad_id))
        chosen_attention = repad(chosen_attention, 0)
        rejected_attention = repad(rejected_attention, 0)
        chosen_labels = repad(chosen_labels, -100)
        rejected_labels = repad(rejected_labels, -100)

        completion_mask = torch.cat(
            (chosen_labels.ne(-100), rejected_labels.ne(-100)), dim=0
        ).long()
        attention_mask = torch.cat((chosen_attention, rejected_attention), dim=0)
        input_ids = torch.cat((chosen_ids, rejected_ids), dim=0)
        if (completion_mask > attention_mask).any():
            raise ValueError("completion mask includes padded tokens")
        if (completion_mask.sum(dim=1) == 0).any():
            raise ValueError("a DPO branch has zero supervised assistant tokens")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }


def build_canonical_trl_trainer_class():
    """Create a DPOTrainer subclass without importing TRL at module import."""
    from torch.utils.data import SequentialSampler
    from trl import DPOTrainer

    class CanonicalDPOTrainer(DPOTrainer):
        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            # Canonical role-aware tokenization is performed by the collator.
            return dataset

        def _get_train_sampler(self, train_dataset=None):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            return SequentialSampler(dataset)

        def log(self, logs, start_time=None):
            if torch.cuda.is_available():
                logs["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            return super().log(logs, start_time)

    return CanonicalDPOTrainer
