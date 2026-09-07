"""Public import migration and unchanged canonical/Champion contracts."""
import importlib
import io
import json

import pytest
import torch


@pytest.mark.parametrize("public,legacy", [
    ("runtime.agent_loop", "champion_runtime.agent_loop"),
    ("runtime.policy", "champion_runtime.local_policy"),
    ("runtime.search", "champion_runtime.offline_search"),
    ("data.sft", "canonical_sft.loader"),
    ("data.preference", "canonical_sft.dpo"),
    ("data.trajectory", "online_grpo.capture"),
    ("evaluation.metrics", "rl_agent.evaluator"),
])
def test_same_module_identity(public, legacy):
    assert importlib.import_module("agentic_search_rl." + public) is importlib.import_module(legacy)


def test_search_query_order_and_observation_mapping(monkeypatch):
    from agentic_search_rl.runtime import search
    requests = []
    def fake_open(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        return io.BytesIO(json.dumps([[{"contents": payload["query"] + "\nEvidence"}], [1.0]]).encode())
    monkeypatch.setattr(search.urllib.request, "urlopen", fake_open)
    output = search.batch_search(["first", "second"])
    assert [r["query"] for r in requests] == ["first", "second"]
    assert output == "Search results for: first\n1. first\nEvidence\n\nSearch results for: second\n1. second\nEvidence"


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}


def test_sft_canonical_masks_and_no_truncation():
    from agentic_search_rl.data.sft import RoleAwareCollator
    from canonical_sft.loader import RoleAwareCollator as Legacy
    tok = CharTokenizer()
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": '<tool_call>{"name":"search","arguments":{"query":["q"]}}</tool_call>'},
        {"role": "user", "content": "<tool_response>Evidence</tool_response>"},
        {"role": "assistant", "content": "<answer>Answer</answer>"},
    ]
    row = {"messages": messages}
    public, old = RoleAwareCollator(tok, 4096)([row]), Legacy(tok, 4096)([row])
    assert all(torch.equal(public[k], old[k]) for k in public)
    supervised = "".join(chr(x) for x in public["labels"][0].tolist() if x != -100)
    assert supervised == messages[1]["content"] + messages[3]["content"]
    with pytest.raises(ValueError, match="silent truncation blocked"):
        RoleAwareCollator(tok, 2)([row])


def test_dpo_shared_prefix_mask_and_preference_direction():
    from agentic_search_rl.data.preference import _encode_with_role_mask, DPORoleAwareCollator
    tok = CharTokenizer()
    prefix = [{"role":"user","content":"Question"}, {"role":"assistant","content":"Old reasoning"}]
    chosen = [{"role":"user","content":"<tool_response>Evidence</tool_response>"},
              {"role":"assistant","content":"Preferred"}]
    rejected = [{"role":"assistant","content":"Rejected"}]
    result = _encode_with_role_mask(tok, prefix, chosen)
    assert "".join(chr(x) for x in result["labels"] if x != -100) == "Preferred"
    assert result["tool_response_tokens"] == 0
    assert _encode_with_role_mask(tok, prefix, rejected)["text"].endswith("Rejected<|im_end|>\n")
    with pytest.raises(ValueError, match="silent truncation blocked"):
        DPORoleAwareCollator(tok, 2)([{"prompt":prefix, "chosen":chosen, "rejected":rejected}])


def test_public_config_is_explicit(tmp_path, monkeypatch):
    from agentic_search_rl.training.sft import read_config
    path = tmp_path / "config.json"
    path.write_text('{"model_path":"${ASRL_TEST_MODEL}"}')
    monkeypatch.delenv("ASRL_TEST_MODEL", raising=False)
    with pytest.raises(ValueError, match="ASRL_TEST_MODEL"):
        read_config(path)
    monkeypatch.setenv("ASRL_TEST_MODEL", '/tmp/model "quoted"')
    assert read_config(path)["model_path"] == '/tmp/model "quoted"'


def test_lightweight_cli(capsys):
    from agentic_search_rl.__main__ import main
    main(["score-answer", "--prediction", "The answer", "--gold", "answer"])
    assert json.loads(capsys.readouterr().out) == {"answer_em":1.0, "answer_f1":1.0}


def test_sft_preflight_delegates_without_step(tmp_path, monkeypatch):
    from agentic_search_rl.training.sft import preflight
    from canonical_sft import training
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_path":"local-base", "tokenizer_path":"local-tokenizer",
                                 "train_dataset_path":"canonical.jsonl", "mode":"full"}))
    model = object()
    monkeypatch.setattr(training, "load_tokenizer", lambda config: object())
    monkeypatch.setattr(training, "load_model", lambda config: model)
    monkeypatch.setattr(training, "prepare_model", lambda value, config: value)
    monkeypatch.setattr(training, "build_dataloader", lambda config, tok: [{"labels":torch.tensor([[1]])}])
    monkeypatch.setattr(training, "build_optimizer", lambda value, config: object())
    monkeypatch.setattr(training, "parameter_state", lambda value: {"total":1,"trainable":1,"frozen":0})
    result = preflight(config)
    assert result["mode"] == "full"
    assert result["optimizer_steps"] == 0
