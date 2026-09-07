import asyncio

import torch

from champion_runtime import agent_loop
from champion_runtime.local_policy import HFPolicyBackend, HFPolicyConfig, PolicyBackend


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "|".join(message["content"] for message in messages)

    def __call__(self, prompt, return_tensors):
        return FakeBatch(input_ids=torch.tensor([[1, 2]], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False):
        return "FAKE"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def generate(self, input_ids, **kwargs):
        return torch.cat((input_ids, torch.tensor([[3]], dtype=torch.long)), dim=1)


def make_backend(stage):
    return HFPolicyBackend.from_model(
        FakeModel(),
        FakeTokenizer(),
        HFPolicyConfig(
            base_model_path="loaded://model",
            tokenizer_path="loaded://tokenizer",
            policy_id=f"fake_{stage}",
            device="cpu",
            generation={"max_new_tokens": 8, "do_sample": False},
            provenance={"policy_stage": stage},
        ),
    )


def test_loaded_model_handle_and_stage_agnostic_generation():
    sft = make_backend("sft")
    dpo = make_backend("dpo")
    messages = [{"role": "user", "content": "hello"}]
    assert isinstance(sft, PolicyBackend)
    assert sft.audit_manifest()["load_origin"] == "model_handle"
    assert sft.audit_manifest()["trainable_parameter_count"] == 1
    assert sft.audit_manifest()["parameters_frozen_for_runtime"] is False
    assert asyncio.run(sft.generate(messages, temperature=0.0)) == "FAKE"
    assert asyncio.run(dpo.generate(messages, temperature=0.0)) == "FAKE"


def test_champion_backend_setter_is_scoped_and_validated():
    previous = agent_loop.set_policy_backend(None)
    backend = make_backend("loaded")
    assert agent_loop.set_policy_backend(backend) is None
    assert agent_loop.get_policy_backend() is backend
    assert agent_loop.set_policy_backend(None) is backend
    assert agent_loop.get_policy_backend() is None
    agent_loop.set_policy_backend(previous)


def test_policy_stage_is_provenance_only():
    a = make_backend("sft").config
    b = make_backend("dpo").config
    assert a.generation == b.generation
    assert a.provenance["policy_stage"] != b.provenance["policy_stage"]
