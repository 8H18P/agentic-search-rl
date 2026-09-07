"""Formal canonical DPO training entrypoint.

This runner deliberately owns the small, stable DPO objective instead of
depending on a version-specific Trainer wrapper.  Canonical tokenization and
role masks remain authoritative: prompt and environment observation tokens are
conditioning-only on both branches.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agentic_search_rl.training.sft import read_config
from .dpo import CanonicalDPODataset, sequence_logps
from .trl_dpo import CanonicalTRLDataCollator


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_hash(named_parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters):
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class DPOTrainConfig:
    model_path: str
    tokenizer_path: str
    sft_adapter_path: str
    train_dataset_path: str
    output_dir: str
    model_id: str = "local-model"
    model_revision: str = "local"
    selected_pair_ids: list[str] = field(default_factory=list)
    max_seq_length: int = 16384
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    max_steps: int = -1
    num_train_epochs: float = 1.0
    beta: float = 0.1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    attention_backend: str = "sdpa"
    device: str = "cuda"
    seed: int = 20260905
    resume_from_checkpoint: str | None = None
    eval_dataset_path: str | None = None
    eval_max_batches: int = 0

    def validate(self) -> None:
        if self.max_seq_length <= 0 or self.per_device_train_batch_size <= 0:
            raise ValueError("sequence length and batch size must be positive")
        if self.gradient_accumulation_steps <= 0 or self.beta <= 0 or self.learning_rate <= 0:
            raise ValueError("gradient accumulation, beta and learning rate must be positive")
        if self.max_steps == 0 or self.num_train_epochs <= 0:
            raise ValueError("training must resolve to at least one optimizer step")
        for path, label in (
            (self.model_path, "model_path"),
            (self.tokenizer_path, "tokenizer_path"),
            (self.sft_adapter_path, "sft_adapter_path"),
            (self.train_dataset_path, "train_dataset_path"),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.eval_dataset_path and not Path(self.eval_dataset_path).exists():
            raise FileNotFoundError(f"eval_dataset_path does not exist: {self.eval_dataset_path}")


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta: float):
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = float(beta) * (policy_margin - reference_margin)
    losses = -F.logsigmoid(logits)
    return losses.mean(), logits


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in ("input_ids", "attention_mask", "completion_mask")
    }


def _branch_logps(model, inputs):
    output = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
    )
    labels = inputs["input_ids"].masked_fill(inputs["completion_mask"].eq(0), -100)
    values = sequence_logps(output.logits, labels)
    half = values.shape[0] // 2
    if half == 0 or values.shape[0] != 2 * half:
        raise ValueError("DPO batch must contain chosen rows followed by rejected rows")
    return values[:half], values[half:]


def _load_base(config: DPOTrainConfig):
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, config.dtype) if config.dtype else None
    return AutoModelForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=config.attention_backend,
        low_cpu_mem_usage=True,
    )


def _load_policy(config: DPOTrainConfig, adapter_path: str | Path, trainable: bool):
    from peft import PeftModel

    return PeftModel.from_pretrained(
        _load_base(config), adapter_path, local_files_only=True, is_trainable=trainable,
        autocast_adapter_dtype=False,
    )


def _load_pair(config: DPOTrainConfig, adapter_path: str | Path):
    policy = _load_policy(config, adapter_path, True)
    reference = _load_policy(config, config.sft_adapter_path, False)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference.eval()
    if config.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.enable_input_require_grads()
    policy.config.use_cache = False
    reference.config.use_cache = False
    return policy, reference


def _selected_dataset(config: DPOTrainConfig):
    dataset = CanonicalDPODataset(config.train_dataset_path)
    if not config.selected_pair_ids:
        return dataset
    wanted = set(config.selected_pair_ids)
    rows = [row for row in dataset.rows if row.get("pair_id") in wanted]
    observed = {row.get("pair_id") for row in rows}
    missing = sorted(wanted - observed)
    if missing:
        raise ValueError(f"selected_pair_ids missing from dataset: {missing}")
    dataset.rows = rows
    return dataset


def _evaluate(model, reference, loader, device, beta, max_batches=0):
    model.eval()
    losses, correct, count = [], 0, 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            inputs = _model_inputs(batch, device)
            pc, pr = _branch_logps(model, inputs)
            rc, rr = _branch_logps(reference, inputs)
            loss, logits = dpo_loss(pc, pr, rc, rr, beta)
            losses.append(float(loss))
            correct += int((logits > 0).sum())
            count += int(logits.numel())
    model.train()
    return {"loss": sum(losses) / len(losses), "reward_accuracy": correct / count, "pairs": count}


def run(config_path: str | Path) -> dict[str, Any]:
    raw = read_config(config_path)
    config = DPOTrainConfig(**raw)
    config.validate()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DPO config requests CUDA, but CUDA is unavailable")

    from transformers import AutoTokenizer

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path, local_files_only=True, trust_remote_code=True
    )
    dataset = _selected_dataset(config)
    if not len(dataset):
        raise ValueError("empty DPO dataset")
    loader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=False,
        collate_fn=CanonicalTRLDataCollator(tokenizer, config.max_seq_length),
    )
    eval_dataset = CanonicalDPODataset(config.eval_dataset_path) if config.eval_dataset_path else dataset
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=False,
        collate_fn=CanonicalTRLDataCollator(tokenizer, config.max_seq_length),
    )
    adapter_source = config.resume_from_checkpoint or config.sft_adapter_path
    policy, reference = _load_pair(config, adapter_source)
    policy.to(device).train()
    reference.to(device)
    trainable = [(name, value) for name, value in policy.named_parameters() if value.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("only one resumed policy LoRA adapter may be trainable")
    if any(value.requires_grad for value in reference.parameters()):
        raise RuntimeError("reference policy must be completely frozen")
    before_hash = _parameter_hash(trainable)
    reference_adapter = [(name, value) for name, value in reference.named_parameters() if "lora_" in name]
    if not config.resume_from_checkpoint and before_hash != _parameter_hash(reference_adapter):
        raise RuntimeError("initial policy and frozen reference adapters are not identical")
    reference_hash = _parameter_hash(reference.named_parameters())
    optimizer = torch.optim.AdamW(
        [value for _, value in trainable], lr=config.learning_rate, weight_decay=config.weight_decay
    )
    completed_steps = 0
    if config.resume_from_checkpoint:
        state_path = Path(config.resume_from_checkpoint) / "trainer_state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        completed_steps = int(state["optimizer_steps"])

    before_metrics = _evaluate(
        policy, reference, eval_loader, device, config.beta, config.eval_max_batches
    )
    formal_micro_steps = math.ceil(len(loader) * config.num_train_epochs)
    target_steps = config.max_steps if config.max_steps > 0 else math.ceil(
        formal_micro_steps / config.gradient_accumulation_steps
    )
    total_micro_steps = target_steps * config.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    metrics, micro_step = [], 0
    while micro_step < total_micro_steps:
        for batch in loader:
            inputs = _model_inputs(batch, device)
            with torch.no_grad():
                reference_chosen, reference_rejected = _branch_logps(reference, inputs)
            policy_chosen, policy_rejected = _branch_logps(policy, inputs)
            loss, logits = dpo_loss(
                policy_chosen, policy_rejected, reference_chosen, reference_rejected, config.beta
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite DPO loss")
            (loss / config.gradient_accumulation_steps).backward()
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [value for _, value in trainable], config.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite DPO gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                record = {
                    "optimizer_step": completed_steps,
                    "loss": float(loss.detach()),
                    "reward_margin": float(logits.detach().mean()),
                    "reward_accuracy": float((logits.detach() > 0).float().mean()),
                    "grad_norm": float(grad_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                metrics.append(record)
                print(json.dumps(record), flush=True)
            if micro_step >= total_micro_steps:
                break

    after_hash = _parameter_hash(trainable)
    if before_hash == after_hash:
        raise RuntimeError("optimizer steps completed but trainable parameter hash did not change")
    if _parameter_hash(reference.named_parameters()) != reference_hash:
        raise RuntimeError("frozen reference policy changed during DPO")
    if not optimizer.state:
        raise RuntimeError("optimizer state is empty after DPO update")

    checkpoint = output_dir / f"checkpoint-{completed_steps}"
    policy.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    torch.save(
        {"optimizer": optimizer.state_dict(), "optimizer_steps": completed_steps},
        checkpoint / "trainer_state.pt",
    )
    adapter_file = next(checkpoint.glob("adapter_model.*"))
    checkpoint_hash = _sha256(adapter_file)
    after_metrics = _evaluate(
        policy, reference, eval_loader, device, config.beta, config.eval_max_batches
    )

    del optimizer, policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = _load_policy(config, checkpoint, False)
    reloaded.to(device).eval()
    reload_hash = _parameter_hash(
        [(name, value) for name, value in reloaded.named_parameters() if "lora_" in name]
    )
    if reload_hash != after_hash:
        raise RuntimeError("clean-reloaded DPO adapter hash differs from saved policy")
    reload_metrics = _evaluate(
        reloaded, reference, eval_loader, device, config.beta, config.eval_max_batches
    )
    manifest = {
        "stage": "dpo",
        "status": "training_completed",
        "config": asdict(config),
        "config_path": str(Path(config_path).resolve()),
        "dataset_sha256": _sha256(config.train_dataset_path),
        "eval_dataset_sha256": _sha256(config.eval_dataset_path) if config.eval_dataset_path else None,
        "evaluation_source": "held_out_dataset" if config.eval_dataset_path else "training_pairs_diagnostic",
        "input_sft_adapter_sha256": _sha256(next(Path(config.sft_adapter_path).glob("adapter_model.*"))),
        "optimizer_steps": completed_steps,
        "backward_called": True,
        "optimizer_state_nonempty": True,
        "parameter_update_confirmed": True,
        "policy_before_hash": before_hash,
        "policy_after_hash": after_hash,
        "reference_frozen_confirmed": True,
        "reference_hash": reference_hash,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "clean_reload_pass": True,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "reload_metrics": reload_metrics,
        "metrics": metrics,
        "handoff": {"next_stage": "grpo", "adapter_path": str(checkpoint.resolve()), "adapter_sha256": checkpoint_hash},
    }
    (output_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "grpo_handoff.json").write_text(
        json.dumps(manifest["handoff"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
