import pytest

from canonical_sft.loader import RoleAwareCollator


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nreason/action<|im_end|>\n<|im_start|>user\n<tool_response>observation</tool_response><|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": list(range(1, len(text) + 1))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_default_blocks_truncation():
    with pytest.raises(ValueError, match="silent truncation blocked"):
        RoleAwareCollator(TinyTokenizer(), 5)([{"messages": [{"role": "assistant", "content": "x"}]}])


def test_explicit_cutoff_is_aligned_and_auditable():
    batch = RoleAwareCollator(TinyTokenizer(), 150, "cutoff")(
        [{"messages": [{"role": "assistant", "content": "x"}]}]
    )
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["effective_lengths"].item() == 150
    assert batch["original_lengths"].item() > 150
    assert batch["truncated"].item() is True
    assert batch["supervised_tokens"].item() > 0


def test_trajectory_response_masks_prompt_and_supervises_observation():
    tokenizer = TinyTokenizer()
    batch = RoleAwareCollator(tokenizer, 1000, "error", "trajectory_response")(
        [{"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}, {"role": "assistant", "content": "reason/action"}, {"role": "user", "content": "<tool_response>observation</tool_response>"}]}]
    )
    text = tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=False)
    labels = batch["labels"][0].tolist()
    prompt_position = text.index("question")
    observation_position = text.index("observation")
    assistant_position = text.index("reason/action")
    assert labels[prompt_position] == -100
    assert labels[assistant_position] != -100
    assert labels[observation_position] != -100
