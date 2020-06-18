import torch

from .abstract import Loss


class MAPO(Loss):
    """Model-Aware Policy Optimization."""

    gamma: float
    grad_estimator: str
    model_samples: int

    def verify_model(self, model):
        """Verify model for Model-Aware DPG."""
        obs = torch.randn(self.observation_space.shape)[None]
        act = torch.randn(self.action_space.shape)[None]
        if self.grad_estimator == "SF":
            sample, logp = model(obs, act.requires_grad_())
            assert sample.grad_fn is None
            assert logp is not None
            logp.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad log_prob must exist for SF estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))
        if self.grad_estimator == "PD":
            sample, _ = model(obs.requires_grad_(), act.requires_grad_())
            sample.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad w.r.t. state and action must exist for PD estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))


class SPAML(Loss):
    """Soft Policy-iteration-Aware Model Learning."""

    gamma: float
    grad_estimator: str
    lambda_: float

    def __init__(self, models, actor, critics):
        pass
