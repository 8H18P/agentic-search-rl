"""Small turn-segment adapter reusing the installed TRL GRPO objective."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math

import torch
try:
    from trl import GRPOTrainer
    from trl.trainer.utils import nanstd
except (ImportError, ModuleNotFoundError):
    class GRPOTrainer:  # permits audit helpers in a non-GRPO environment
        def __init__(self, *args, **kwargs):
            raise RuntimeError("the formal GRPO entry requires the pinned TRL environment")

    def nanstd(tensor, dim=0):
        return tensor.std(dim=dim, correction=0)

from .capture import audit_segment


def parameter_groups(model):
    groups = {"default": [], "ref": [], "base": []}
    for name, p in model.named_parameters():
        group = "default" if ".default." in name else "ref" if ".ref." in name else "base"
        groups[group].append((name, p))
    return groups


def parameter_hash(named):
    h = hashlib.sha256()
    for name, p in sorted(named):
        name = name.replace(".default.", ".adapter.").replace(".ref.", ".adapter.")
        h.update(name.encode()); h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def enforce_default_ownership(model):
    model.set_adapter("default")
    groups = parameter_groups(model)
    for kind, entries in groups.items():
        for _, p in entries:
            p.requires_grad_(kind == "default")
    if not groups["default"] or not groups["ref"]:
        raise ValueError("missing DPO/default/reference adapter")


@contextmanager
def reference_scope(model):
    """PEFT set_adapter toggles requires_grad; explicitly freeze reference inference."""
    model.set_adapter("ref")
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            yield
    finally:
        enforce_default_ownership(model)


def ownership_audit(model, optimizer):
    groups = parameter_groups(model)
    ids = {k: {id(p) for _, p in values} for k, values in groups.items()}
    opt = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = ids["default"]
    flags = {k: all(p.requires_grad == (k == "default") for _, p in v) for k, v in groups.items()}
    if opt != expected or opt & ids["ref"] or opt & ids["base"] or not all(flags.values()):
        raise ValueError("optimizer/default/ref/base ownership violation")
    return {"optimizer_owns_default_only": True, "default_trainable": flags["default"],
            "ref_frozen": flags["ref"], "base_frozen": flags["base"],
            "parameter_ids": {k: sorted(v) for k,v in ids.items()},
            "optimizer_parameter_ids": sorted(opt),
            "named_parameter_ids": {k: [{"name": n,"id": id(p),"numel": p.numel(),"requires_grad": p.requires_grad} for n,p in vals] for k,vals in groups.items()},
            "process_local_ids_only": True}


def group_advantages(rewards):
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("invalid group rewards")
    tensor = torch.tensor(rewards, dtype=torch.float32)
    mean, std = tensor.mean(), nanstd(tensor, dim=0)
    advantages = (tensor-mean)/(std+1e-4)
    return advantages.tolist(), {"reward_per_rollout": rewards, "group_mean": mean.item(), "group_std": std.item(),
        "advantage_per_rollout": advantages.tolist(), "zero_variance_group": bool(std == 0),
        "implementation": "TRL group scaling with nanstd and epsilon=1e-4"}


class ChampionSegmentGRPOTrainer(GRPOTrainer):
    """Use real captured states; stock TRL owns the per-token clipped objective.

    Each segment is weighted by its sampled token share within its trajectory.
    Thus a longer sequence of turns does not gain extra trajectory weight.
    """

    def _generate_and_score_completions(self, inputs):
        raise RuntimeError("Champion training uses captured turn-segment replay through the GRPO runner")

    @contextmanager
    def segment_temperature(self, value):
        previous = self.temperature
        self.temperature = float(value)
        try:
            yield
        finally:
            self.temperature = previous

    def segment_inputs(self, segment, advantage):
        audit_segment(segment)
        device = self.accelerator.device
        state = torch.tensor([segment["state_ids"]], dtype=torch.long, device=device)
        actions = torch.tensor([segment["action_ids"]], dtype=torch.long, device=device)
        return {"prompt_ids": state, "prompt_mask": torch.ones_like(state),
                "completion_ids": actions, "completion_mask": torch.ones_like(actions),
                "tool_mask": torch.ones_like(actions),
                "advantages": torch.tensor([advantage], dtype=torch.float32, device=device)}

    def segment_logps(self, model, inputs):
        ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        mask = torch.ones_like(ids)
        return self._get_per_token_logps_and_entropies(model, ids, mask, inputs["completion_ids"].shape[1], batch_size=1)[0]

    def objective_forward(self, trajectories, rewards):
        advantages, result = group_advantages(rewards)
        self.model.eval()
        records, total_loss = [], 0.0
        with torch.no_grad():
            for trajectory, advantage in zip(trajectories, advantages):
                denominator = sum(len(s["action_ids"]) for s in trajectory)
                for segment in trajectory:
                    with self.segment_temperature(segment["temperature"]):
                        inputs = self.segment_inputs(segment, advantage)
                        policy = self.segment_logps(self.model, inputs)
                        old = policy.detach().clone()
                        with reference_scope(self.model):
                            reference = self.segment_logps(self.model, inputs)
                        back = self.segment_logps(self.model, inputs)
                        torch.testing.assert_close(policy, reference, rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(policy, back, rtol=1e-5, atol=1e-5)
                        inputs.update(old_per_token_logps=old, ref_per_token_logps=reference)
                        # This is TRL's implementation, not a copied objective.
                        loss = super()._compute_loss(self.model, inputs)
                        ratio = (policy-old).exp()
                        kl = ((reference-policy).exp()-(reference-policy)-1).mean()
                        values = [loss.item(), policy.sum().item(), old.sum().item(), reference.sum().item(), kl.item()]
                        if not all(math.isfinite(x) for x in values) or not torch.isfinite(ratio).all():
                            raise ValueError("nonfinite GRPO objective")
                        weight = len(segment["action_ids"])/denominator/len(trajectories)
                        total_loss += loss.item()*weight
                        records.append({"trajectory_id": segment["capture"]["trajectory_id"],
                            "generation_idx": segment["capture"]["generation_idx"], "temperature": segment["temperature"],
                            "policy_logp": policy.sum().item(), "reference_logp": reference.sum().item(), "old_logp": old.sum().item(),
                            "policy_ratio_min": ratio.min().item(), "policy_ratio_max": ratio.max().item(),
                            "policy_loss": loss.item()-self.beta*kl.item(), "kl": kl.item(), "total_loss": loss.item(),
                            "trajectory_normalized_weight": weight, "sampled_tokens": len(segment["action_ids"]),
                            "default_ref_output_equal": True, "switch_back_equal": True})
                        del policy, old, reference, back, inputs, loss, ratio
        result.update(segments=records, total_loss=total_loss, grpo_loss_finite=math.isfinite(total_loss),
                      full_grpo_objective_forward_pass=True, active_adapter_returns_to_default=self.model.active_adapter == "default",
                      stock_trl_objective="GRPOTrainer._compute_loss", loss_normalization="mean_trajectory(mean_sampled_tokens)",
                      backward_called=False, optimizer_steps=0)
        return result

    def training_update(self, trajectories, rewards, max_grad_norm=1.0):
        """Apply one real GRPO optimizer update to a fully captured group.

        ``old_per_token_logps`` are snapshotted before any backward call.  The
        frozen ``ref`` adapter and base are then audited before and after the
        update.  Environment observations occur only in ``state_ids``;
        ``completion_mask`` covers sampled policy action IDs exclusively.
        """
        if not trajectories or len(trajectories) != len(rewards):
            raise ValueError("one reward is required for every trajectory")
        advantages, group_audit = group_advantages(rewards)
        before = parameter_groups(self.model)
        policy_before = parameter_hash(before["default"])
        reference_before = parameter_hash(before["ref"])
        base_before = parameter_hash(before["base"])
        ownership_audit(self.model, self.optimizer)

        snapshots = []
        self.model.eval()
        with torch.no_grad():
            for trajectory, advantage in zip(trajectories, advantages):
                denominator = sum(len(segment["action_ids"]) for segment in trajectory)
                if denominator <= 0:
                    raise ValueError("trajectory contains no sampled policy tokens")
                for segment in trajectory:
                    with self.segment_temperature(segment["temperature"]):
                        inputs = self.segment_inputs(segment, advantage)
                        old = self.segment_logps(self.model, inputs).detach()
                        with reference_scope(self.model):
                            reference = self.segment_logps(self.model, inputs).detach()
                    snapshots.append((
                        segment,
                        advantage,
                        old,
                        reference,
                        len(segment["action_ids"]) / denominator / len(trajectories),
                    ))

        enforce_default_ownership(self.model)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        records, scalar_loss = [], 0.0
        for segment, advantage, old, reference, weight in snapshots:
            with self.segment_temperature(segment["temperature"]):
                inputs = self.segment_inputs(segment, advantage)
                policy = self.segment_logps(self.model, inputs)
                inputs.update(old_per_token_logps=old, ref_per_token_logps=reference)
                loss = super()._compute_loss(self.model, inputs)
                weighted_loss = loss * weight
                if not torch.isfinite(weighted_loss):
                    raise FloatingPointError("non-finite weighted GRPO loss")
                weighted_loss.backward()
                scalar_loss += float(weighted_loss.detach())
                delta = reference - policy.detach()
                kl = (delta.exp() - delta - 1).mean()
                records.append({
                    "trajectory_id": segment["capture"]["trajectory_id"],
                    "generation_idx": segment["capture"]["generation_idx"],
                    "sampled_tokens": len(segment["action_ids"]),
                    "trajectory_normalized_weight": weight,
                    "total_loss": float(loss.detach()),
                    "policy_loss": float(loss.detach()) - self.beta * float(kl),
                    "kl": float(kl),
                    "old_logp": float(old.sum()),
                    "reference_logp": float(reference.sum()),
                })

        trainable = [parameter for _, parameter in parameter_groups(self.model)["default"]]
        if not trainable or not any(parameter.grad is not None for parameter in trainable):
            raise RuntimeError("GRPO backward produced no policy gradients")
        if any(
            parameter.grad is not None
            for kind in ("ref", "base")
            for _, parameter in parameter_groups(self.model)[kind]
        ):
            raise RuntimeError("GRPO gradients leaked into reference or base parameters")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(max_grad_norm))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite GRPO gradient norm")
        self.optimizer.step()
        scheduler = getattr(self, "lr_scheduler", None)
        if scheduler is not None:
            scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        after = parameter_groups(self.model)
        policy_after = parameter_hash(after["default"])
        if policy_after == policy_before:
            raise RuntimeError("GRPO optimizer step did not change policy parameters")
        if parameter_hash(after["ref"]) != reference_before:
            raise RuntimeError("reference adapter changed during GRPO update")
        if parameter_hash(after["base"]) != base_before:
            raise RuntimeError("frozen base changed during GRPO update")
        if not self.optimizer.state:
            raise RuntimeError("optimizer state is empty after GRPO update")
        ownership_audit(self.model, self.optimizer)
        return {
            **group_audit,
            "segments": records,
            "loss": scalar_loss,
            "policy_loss": sum(row["policy_loss"] * row["trajectory_normalized_weight"] for row in records),
            "kl": sum(row["kl"] * row["trajectory_normalized_weight"] for row in records),
            "grad_norm": float(grad_norm),
            "backward_called": True,
            "optimizer_steps": 1,
            "optimizer_state_nonempty": True,
            "parameter_updated": True,
            "policy_before_hash": policy_before,
            "policy_after_hash": policy_after,
            "reference_hash": reference_before,
            "reference_unchanged": True,
            "base_hash": base_before,
            "base_unchanged": True,
            "environment_tokens_in_policy_loss": 0,
            "old_policy_snapshot_timing": "same group, before backward and optimizer.step",
        }
