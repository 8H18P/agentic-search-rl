"""Canonical HF/PEFT construction with no duplicated training loop."""
import json
import os
import re
from pathlib import Path


def read_config(path):
    """Expand explicit environment placeholders; fail rather than guess missing paths."""
    text = Path(path).read_text(encoding="utf-8")
    def replace(match):
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Required configuration environment variable is unset: {name}")
        # Escape for the containing JSON string, including Windows paths.
        return json.dumps(os.environ[name])[1:-1]
    return json.loads(re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, text))


def preflight(config_path):
    """Construct existing canonical data/model/optimizer without forward, step, or save."""
    from canonical_sft.training import (
        SFTConfig, load_tokenizer, load_model, prepare_model,
        build_dataloader, build_optimizer, parameter_state,
    )
    config = SFTConfig(**read_config(config_path))
    tokenizer = load_tokenizer(config)
    loader = build_dataloader(config, tokenizer)
    batch = next(iter(loader))
    model = prepare_model(load_model(config), config)
    optimizer = build_optimizer(model, config)
    return {
        "mode": config.mode,
        "parameter_state": parameter_state(model),
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "optimizer": type(optimizer).__name__,
        "optimizer_steps": 0,
        "preflight_only": True,
    }
