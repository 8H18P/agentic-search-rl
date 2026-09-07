from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .training import (
    SFTConfig,
    build_dataloader,
    build_optimizer,
    load_model,
    load_tokenizer,
    parameter_state,
    prepare_model,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_batch(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in ("input_ids", "attention_mask", "labels")
    }


def sparse_supervised_loss(model, batch):
    """Exact causal CE while materializing logits only for supervised targets."""
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("sparse_supervised currently requires batch_size=1")
    targets = batch["labels"][:, 1:]
    positions = torch.nonzero(targets[0] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("zero supervised causal targets")
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        logits_to_keep=positions,
    )
    logits = output.logits
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets[:, positions].reshape(-1),
        reduction="mean",
    )


def liger_sparse_supervised_loss(model, batch):
    """The same sparse causal CE fused with the output projection for memory."""
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("liger_sparse_supervised currently requires batch_size=1")
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    targets = batch["labels"][:, 1:]
    positions = torch.nonzero(targets[0] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("zero supervised causal targets")
    core = model.base_model.model if hasattr(model, "base_model") else model
    hidden = core.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )[0]
    supervised_hidden = hidden[:, positions, :].reshape(-1, hidden.shape[-1])
    supervised_targets = targets[:, positions].reshape(-1)
    criterion = LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)
    return criterion(core.lm_head.weight, supervised_hidden, supervised_targets)


def compute_loss(model, batch, config):
    if config.loss_implementation == "standard":
        return model(**batch, use_cache=config.use_cache).loss
    if config.loss_implementation == "liger_sparse_supervised":
        return liger_sparse_supervised_loss(model, batch)
    return sparse_supervised_loss(model, batch)


def _scheduler(optimizer, config, total_steps):
    warmup_steps = math.ceil(config.warmup_ratio * total_steps)

    def multiplier(current_step):
        # Step numbers are shifted so a one-step engineering trial still makes
        # a real update while exercising the production warmup configuration.
        next_step = current_step + 1
        if warmup_steps and next_step <= warmup_steps:
            return next_step / warmup_steps
        if config.lr_scheduler_type == "constant":
            return 1.0
        progress = (next_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup_steps


def _init_wandb(config, raw_config, output_dir, parameter_state, total_steps):
    if "wandb" not in config.report_to:
        return None
    import wandb

    identity_path = Path("configs/model/qwen35_4b_identity.json")
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text()) if dataset_manifest_path.exists() else {}
    public_config = dict(raw_config)
    public_config.update(
        {
            "model_identity": identity.get("model_id", "Qwen/Qwen3.5-4B"),
            "model_revision": identity.get("resolved_revision"),
            "tokenizer_config_sha256": identity.get("tokenizer_config_sha256"),
            "tokenizer_json_sha256": identity.get("tokenizer_json_sha256"),
            "canonical_source_sha256": dataset_manifest.get("source_sha256"),
            "serialized_dataset_sha256": dataset_manifest.get("serialized_sha256"),
            "train_sample_count": dataset_manifest.get("count"),
            "truncated_sample_count": dataset_manifest.get("truncated_sample_count"),
            "effective_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
            "bf16": config.dtype == "bfloat16",
            "loss_impl": "fused_sparse_causal_ce" if config.loss_implementation == "liger_sparse_supervised" else config.loss_implementation,
            "smartsearch_faithful_labels": config.supervision_mode == "trajectory_response",
            "validation400_used": False,
            "trainable_params": parameter_state["trainable"],
            "total_params": parameter_state["total"],
            "optimizer_step_target": total_steps,
        }
    )
    try:
        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            job_type=config.wandb_job_type,
            tags=config.wandb_tags,
            config=public_config,
            mode=config.wandb_mode,
        )
    except Exception:
        if config.wandb_mode != "online":
            raise
        run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            job_type=config.wandb_job_type,
            tags=config.wandb_tags,
            config=public_config,
            mode="offline",
        )
    run_info = {
        "enabled": True,
        "project": config.wandb_project,
        "run_name": config.wandb_run_name,
        "run_id": run.id,
        "mode": getattr(run.settings, "mode", config.wandb_mode),
        "url": run.url,
        "offline_directory": run.dir,
    }
    (output_dir / "wandb_run.json").write_text(json.dumps(run_info, indent=2))
    return run


def run(config_path):
    from agentic_search_rl.training.sft import read_config

    raw_config = read_config(config_path)
    config = SFTConfig(**raw_config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config)
    loader = build_dataloader(config, tokenizer)
    model = prepare_model(load_model(config), config)
    model.train()
    state = parameter_state(model)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if config.mode == "lora" and any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("unexpected non-LoRA trainable parameter")

    device = torch.device("cuda")
    model.to(device)
    optimizer = build_optimizer(model, config)
    formal_micro_steps = int(len(loader) * config.num_train_epochs)
    derived_steps = math.ceil(formal_micro_steps / config.gradient_accumulation_steps)
    target_steps = config.max_steps if config.max_steps > 0 else derived_steps
    if target_steps < 1:
        raise ValueError("training run resolves to zero optimizer steps")
    scheduler_total_steps = config.scheduler_total_steps or target_steps
    scheduler, warmup_steps = _scheduler(optimizer, config, scheduler_total_steps)
    before = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable}
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    total_micro_target = target_steps * config.gradient_accumulation_steps if config.max_steps > 0 else formal_micro_steps
    wandb_run = _init_wandb(config, raw_config, output_dir, state, target_steps)
    metrics = []
    micro_metrics = []
    supervised_tokens_seen = truncated_samples_seen = 0
    micro_step = optimizer_step = 0
    while micro_step < total_micro_target:
        if len(loader) == 0:
            raise RuntimeError("empty dataset")
        for audit_batch in loader:
            batch = _model_batch(audit_batch, device)
            micro_step += 1
            epoch = (micro_step - 1) // len(loader) + 1
            group_start = ((micro_step - 1) // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
            accumulation_divisor = min(config.gradient_accumulation_steps, total_micro_target - group_start)
            started = time.perf_counter()
            loss = compute_loss(model, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {loss.item()}")
            (loss / accumulation_divisor).backward()
            elapsed = time.perf_counter() - started
            micro_record = {
                "micro_step": micro_step,
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "loss": float(loss.detach()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "original_tokens": int(audit_batch["original_lengths"][0]),
                "effective_tokens": int(audit_batch["effective_lengths"][0]),
                "truncated": bool(audit_batch["truncated"][0]),
                "supervised_tokens": int(audit_batch["supervised_tokens"][0]),
                "masked_prompt_tokens": int(audit_batch["masked_tokens"][0]),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "step_time_seconds": elapsed,
            }
            micro_metrics.append(micro_record)
            print(json.dumps(micro_record), flush=True)
            supervised_tokens_seen += micro_record["supervised_tokens"]
            truncated_samples_seen += int(micro_record["truncated"])
            will_optimizer_step = micro_step % config.gradient_accumulation_steps == 0 or micro_step == total_micro_target
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/micro_loss": micro_record["loss"],
                        "train/sequence_length": micro_record["effective_tokens"],
                        "train/supervised_tokens": micro_record["supervised_tokens"],
                        "train/is_truncated": int(micro_record["truncated"]),
                        "train/micro_step": micro_step,
                        "train/epoch": epoch,
                        "system/peak_vram_allocated_gb": micro_record["peak_allocated_bytes"] / 1024**3,
                        "system/peak_vram_reserved_gb": micro_record["peak_reserved_bytes"] / 1024**3,
                        "system/step_time_sec": micro_record["step_time_seconds"],
                        "progress/samples_seen": micro_step,
                        "progress/supervised_tokens_seen": supervised_tokens_seen,
                        "progress/truncated_samples_seen": truncated_samples_seen,
                    },
                    step=micro_step,
                    commit=not will_optimizer_step,
                )
            if micro_step % config.gradient_accumulation_steps and micro_step != total_micro_target:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], config.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm: {grad_norm.item()}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            record = {
                "step": optimizer_step,
                "loss": float(loss.detach()),
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "original_tokens": int(audit_batch["original_lengths"][0]),
                "effective_tokens": int(audit_batch["effective_lengths"][0]),
                "truncated": bool(audit_batch["truncated"][0]),
                "supervised_tokens": int(audit_batch["supervised_tokens"][0]),
            }
            metrics.append(record)
            print(json.dumps(record), flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": record["loss"],
                        "train/learning_rate": record["learning_rate"],
                        "train/grad_norm": record["grad_norm"],
                        "train/optimizer_step": optimizer_step,
                        "train/epoch": epoch,
                    },
                    step=micro_step,
                )
            if micro_step >= total_micro_target:
                break

    changed = any(
        not torch.equal(before[name], parameter.detach().float().cpu())
        for name, parameter in trainable
    )
    checkpoint = output_dir / "checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    manifest = {
        "engineering_smoke_only": True,
        "config": raw_config,
        "config_path": str(Path(config_path).resolve()),
        "dataset_sha256": _sha256(config.train_dataset_path),
        "parameter_state": state,
        "trainable_parameter_names": [name for name, _ in trainable],
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "parameter_update_confirmed": changed,
        "metrics": metrics,
        "micro_metrics": micro_metrics,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "scheduler_total_steps": scheduler_total_steps,
        "warmup_steps": warmup_steps,
        "validation400_used": False,
        "checkpoint": str(checkpoint.resolve()),
        "wandb": json.loads((output_dir / "wandb_run.json").read_text()) if wandb_run is not None else {"enabled": False},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in micro_metrics), encoding="utf-8"
    )

    del optimizer, scheduler, model
    torch.cuda.empty_cache()
    from peft import PeftModel

    base = load_model(config).to(device)
    reloaded = PeftModel.from_pretrained(base, checkpoint, is_trainable=False).to(device).eval()
    reload_batch = _model_batch(next(iter(loader)), device)
    with torch.no_grad():
        reload_loss = compute_loss(reloaded, reload_batch, config)
    manifest.update(
        {
            "checkpoint_save_pass": (checkpoint / "adapter_config.json").exists()
            and any(checkpoint.glob("adapter_model.*")),
            "clean_reload_pass": True,
            "reload_forward_pass": bool(torch.isfinite(reload_loss)),
            "reload_loss": float(reload_loss),
            "checkpoint_sha256": _sha256(next(checkpoint.glob("adapter_model.*"))),
        }
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "optimizer_steps": optimizer_step,
                "micro_steps": micro_step,
                "parameter_update_confirmed": changed,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "reload_loss": manifest["reload_loss"],
                "validation400_used": False,
            }
        )
        wandb_run.finish()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
