"""Stage-agnostic Hugging Face/PEFT policy backend for Champion runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import torch


@runtime_checkable
class PolicyBackend(Protocol):
    """The only model contract consumed by Champion Agent runtime."""

    identity: str

    async def generate(
        self,
        messages: list[dict[str, str]],
        stop: list[str] | None = None,
        temperature: float = 0.4,
        max_tokens: int = 8192,
    ) -> str: ...


@dataclass(frozen=True)
class HFPolicyConfig:
    base_model_path: str
    tokenizer_path: str
    adapter_path: str | None = None
    policy_id: str = "hf_policy"
    dtype: str = "bfloat16"
    device: str = "cuda"
    attention_backend: str = "sdpa"
    local_files_only: bool = True
    trust_remote_code: bool = True
    autocast_adapter_dtype: bool = False
    freeze_parameters: bool | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HFPolicyConfig":
        return cls(**dict(value))

    @classmethod
    def from_file(cls, path: str | Path) -> "HFPolicyConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate_for_checkpoint_load(self) -> None:
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        for path, label in (
            (self.base_model_path, "base_model_path"),
            (self.tokenizer_path, "tokenizer_path"),
        ):
            if self.local_files_only and not Path(path).exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.adapter_path and self.local_files_only and not Path(self.adapter_path).exists():
            raise FileNotFoundError(f"adapter_path does not exist: {self.adapter_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HFPolicyBackend:
    """One runtime implementation for base and arbitrary PEFT checkpoints.

    Training stage is retained only in ``config.provenance``. It never selects
    code paths or changes Champion runtime semantics.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: HFPolicyConfig,
        *,
        load_origin: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.identity = config.policy_id
        self.load_origin = load_origin
        # Optional training observability; does not change the inference API.
        self.capture_sink = None
        self.parameters_frozen_for_runtime = (
            load_origin == "checkpoint"
            if config.freeze_parameters is None
            else config.freeze_parameters
        )
        if self.parameters_frozen_for_runtime:
            self.model.requires_grad_(False)
        self.model.eval()
        self._audit = self._build_audit()

    @classmethod
    def from_checkpoint(cls, config: HFPolicyConfig) -> "HFPolicyBackend":
        config.validate_for_checkpoint_load()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, config.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
            dtype=dtype,
            attn_implementation=config.attention_backend,
            low_cpu_mem_usage=True,
        )
        if config.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                config.adapter_path,
                is_trainable=False,
                local_files_only=config.local_files_only,
                autocast_adapter_dtype=config.autocast_adapter_dtype,
            )
        model = model.to(config.device)
        return cls(model, tokenizer, config, load_origin="checkpoint")

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        config: HFPolicyConfig | Mapping[str, Any] | None = None,
    ) -> "HFPolicyBackend":
        """Wrap an already-loaded model handle without disk reload or mutation."""
        if config is None:
            config = HFPolicyConfig(
                base_model_path="loaded://model",
                tokenizer_path="loaded://tokenizer",
                policy_id="loaded_model_policy",
                device=str(getattr(model, "device", "loaded")),
                provenance={"policy_stage": "online_or_loaded"},
            )
        elif not isinstance(config, HFPolicyConfig):
            config = HFPolicyConfig.from_mapping(config)
        return cls(model, tokenizer, config, load_origin="model_handle")

    def _build_audit(self) -> dict[str, Any]:
        adapter_names = sorted(getattr(self.model, "peft_config", {}).keys())
        adapter_config = None
        adapter_file = None
        if self.config.adapter_path:
            config_path = Path(self.config.adapter_path) / "adapter_config.json"
            if config_path.exists():
                adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
            for filename in ("adapter_model.safetensors", "adapter_model.bin"):
                candidate = Path(self.config.adapter_path) / filename
                if candidate.exists():
                    adapter_file = candidate
                    break
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        return {
            "policy_id": self.config.policy_id,
            "load_origin": self.load_origin,
            "base_model_path": self.config.base_model_path,
            "tokenizer_path": self.config.tokenizer_path,
            "adapter_path": self.config.adapter_path,
            "adapter_attached": bool(self.config.adapter_path),
            "adapter_names": adapter_names,
            "adapter_base_model": (adapter_config or {}).get("base_model_name_or_path"),
            "adapter_type": (adapter_config or {}).get("peft_type"),
            "adapter_rank": (adapter_config or {}).get("r"),
            "adapter_file": str(adapter_file) if adapter_file else None,
            "adapter_file_sha256": _sha256(adapter_file) if adapter_file else None,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "parameters_frozen_for_runtime": self.parameters_frozen_for_runtime,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "attention_backend": self.config.attention_backend,
            "generation": dict(self.config.generation),
            "provenance": dict(self.config.provenance),
        }

    def audit_manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._audit))

    def _generate_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        generation = dict(self.config.generation)
        configured_max = int(generation.pop("max_new_tokens", max_tokens))
        environment_cap = int(os.getenv("LOCAL_MAX_NEW_TOKENS", str(configured_max)))
        effective_max = min(int(max_tokens), configured_max, environment_cap)
        configured_sampling = generation.pop("do_sample", None)
        do_sample = temperature > 0 if configured_sampling is None else bool(configured_sampling)
        enable_thinking = bool(generation.pop("enable_thinking", False))

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        batch = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        kwargs = {
            "max_new_tokens": effective_max,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            **generation,
        }
        if do_sample:
            kwargs["temperature"] = max(float(temperature), 1e-5)
        if self.capture_sink is not None:
            self.capture_sink.before_generation(batch, prompt, messages, kwargs, self.model)
        with torch.inference_mode():
            output = self.model.generate(**batch, **kwargs)
        generated = output[0][batch["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=False)
        if self.capture_sink is not None:
            self.capture_sink.after_generation(generated, raw_text, raw_text.strip())
        return raw_text.strip()

    async def generate(
        self,
        messages: list[dict[str, str]],
        stop: list[str] | None = None,
        temperature: float = 0.4,
        max_tokens: int = 8192,
    ) -> str:
        # ``stop`` is intentionally not newly interpreted here: the legacy
        # local backend also left stopping to model EOS and Champion parsing.
        return await asyncio.to_thread(self._generate_sync, messages, temperature, max_tokens)


def load_policy_from_config(
    config: str | Path | Mapping[str, Any] | HFPolicyConfig,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> HFPolicyBackend:
    if isinstance(config, (str, Path)):
        parsed = HFPolicyConfig.from_file(config)
    elif isinstance(config, HFPolicyConfig):
        parsed = config
    else:
        parsed = HFPolicyConfig.from_mapping(config)
    if model is None and tokenizer is None:
        return HFPolicyBackend.from_checkpoint(parsed)
    if model is None or tokenizer is None:
        raise ValueError("model and tokenizer must be supplied together")
    return HFPolicyBackend.from_model(model, tokenizer, parsed)


class LocalQwenPolicy:
    """Backward-compatible facade for legacy research scripts."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        tokenizer_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        self._backend = HFPolicyBackend.from_checkpoint(
            HFPolicyConfig(
                base_model_path=model_path,
                tokenizer_path=tokenizer_path or model_path,
                adapter_path=adapter_path,
                policy_id=f"{model_path}+{adapter_path or 'no_adapter'}",
                device=device,
                generation={"max_new_tokens": 8192, "enable_thinking": False},
                provenance={"legacy_facade": True},
            )
        )
        self.model = self._backend.model
        self.tokenizer = self._backend.tokenizer
        self.device = self._backend.device
        self.identity = self._backend.identity

    def audit_manifest(self) -> dict[str, Any]:
        return self._backend.audit_manifest()

    async def generate(self, *args, **kwargs) -> str:
        return await self._backend.generate(*args, **kwargs)
