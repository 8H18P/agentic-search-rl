import pytest

torch = pytest.importorskip("torch")

from canonical_sft.dpo_runner import dpo_loss


def test_dpo_objective_rewards_larger_policy_margin():
    ref_chosen = torch.tensor([2.0])
    ref_rejected = torch.tensor([1.0])
    neutral, neutral_logits = dpo_loss(
        torch.tensor([2.0]), torch.tensor([1.0]), ref_chosen, ref_rejected, 0.1
    )
    improved, improved_logits = dpo_loss(
        torch.tensor([3.0]), torch.tensor([1.0]), ref_chosen, ref_rejected, 0.1
    )
    assert neutral_logits.item() == 0.0
    assert improved_logits.item() > 0.0
    assert improved.item() < neutral.item()


def test_dpo_objective_backpropagates_only_policy_values():
    chosen = torch.tensor([2.0], requires_grad=True)
    rejected = torch.tensor([1.0], requires_grad=True)
    reference_chosen = torch.tensor([2.0])
    reference_rejected = torch.tensor([1.0])
    loss, _ = dpo_loss(chosen, rejected, reference_chosen, reference_rejected, 0.1)
    loss.backward()
    assert chosen.grad is not None and rejected.grad is not None
    assert reference_chosen.grad is None and reference_rejected.grad is None
