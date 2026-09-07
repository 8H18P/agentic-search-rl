import unittest
from unittest.mock import patch
import torch
from canonical_sft.llamafactory_dpo import CanonicalLlamaFactoryCollator, validate_explicit_reference


class Stub(torch.nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(2), requires_grad=trainable)
        self.base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.peft_config = {"default": object()}


class BridgeTest(unittest.TestCase):
    def test_labels_preserve_only_completion(self):
        encoded = {"input_ids": torch.tensor([[1, 2, 3], [1, 4, 0]]),
                   "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                   "completion_mask": torch.tensor([[0, 1, 0], [0, 1, 0]])}
        with patch("canonical_sft.llamafactory_dpo.CanonicalTRLDataCollator") as mock:
            mock.return_value.return_value = encoded
            result = CanonicalLlamaFactoryCollator(None, 8)([{}])
        self.assertEqual(result["labels"].tolist(), [[-100, 2, -100], [-100, 4, -100]])
        self.assertEqual(encoded["input_ids"].tolist(), [[1, 2, 3], [1, 4, 0]])

    def test_explicit_frozen_reference(self):
        validate_explicit_reference(Stub(True), Stub(False))

    def test_naked_base_fallback_blocked(self):
        with self.assertRaises(ValueError):
            validate_explicit_reference(Stub(True), None)

    def test_reference_trainable_blocked(self):
        with self.assertRaises(ValueError):
            validate_explicit_reference(Stub(True), Stub(True))

    def test_shared_model_blocked(self):
        model = Stub(True)
        with self.assertRaises(ValueError):
            validate_explicit_reference(model, model)

    def test_base_trainable_blocked(self):
        model = Stub(True)
        model.base.requires_grad_(True)
        with self.assertRaises(ValueError):
            validate_explicit_reference(model, Stub())

    def test_stacking_blocked(self):
        model = Stub(True)
        model.peft_config["extra"] = object()
        with self.assertRaises(ValueError):
            validate_explicit_reference(model, Stub())

    def test_wrong_initial_adapter_blocked(self):
        reference = Stub()
        with torch.no_grad():
            reference.lora_A.add_(1)
        with self.assertRaises(ValueError):
            validate_explicit_reference(Stub(True), reference)


if __name__ == "__main__":
    unittest.main()
