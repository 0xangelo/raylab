"""Policy for MAPO using PyTorch."""
from ray.rllib.utils import override

from raylab.agents.sac import SACTorchPolicy
from raylab.policy import EnvFnMixin
from raylab.policy import ModelTrainingMixin
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.policy.losses import DAPO
from raylab.policy.losses import MAPO
from raylab.policy.losses import SPAML
from raylab.pytorch.optim import build_optimizer


class MAPOTorchPolicy(ModelTrainingMixin, EnvFnMixin, SACTorchPolicy):
    """Model-Aware Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method
    dist_class = WrapStochasticPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.loss_model = SPAML(
            self.module.models, self.module.actor, self.module.critics
        )
        self.loss_model.gamma = self.config["gamma"]
        self.loss_model.grad_estimator = self.config["losses"]["grad_estimator"]
        self.loss_model.lambda_ = self.config["losses"]["lambda"]

        self.loss_actor = MAPO(
            self.module.models, self.module.actor, self.module.critics
        )
        self.loss_actor.gamma = self.config["gamma"]
        self.loss_actor.model_samples = self.config["losses"]["model_samples"]
        self.loss_actor.grad_estimator = self.config["losses"]["grad_estimator"]

    @override(EnvFnMixin)
    def _set_reward_hook(self):
        self.loss_model.set_reward_fn(self.reward_fn)
        self.loss_actor.set_reward_fn(self.reward_fn)

    @override(EnvFnMixin)
    def _set_termination_hook(self):
        self.loss_model.set_termination_fn(self.termination_fn)
        self.loss_actor.set_termination_fn(self.termination_fn)

    @override(EnvFnMixin)
    def _set_dynamics_hook(self):
        self.loss_actor = DAPO(self.dynamics_fn, self.module.actor, self.module.critics)
        self.loss_actor.gamma = self.config["gamma"]
        self.loss_actor.dynamics_samples = self.config["losses"]["model_samples"]
        self.loss_actor.grad_estimator = self.config["losses"]["grad_estimator"]

        if self.reward_fn:
            self.loss_actor.set_reward_fn(self.reward_fn)
        if self.termination_fn:
            self.loss_actor.set_termination_fn(self.termination_fn)

    @staticmethod
    def get_default_config():
        """Return the default configuration for MAPO."""
        # pylint:disable=cyclic-import,protected-access
        from raylab.agents.mapo import MAPOTrainer

        return MAPOTrainer._default_config

    @override(SACTorchPolicy)
    def make_optimizers(self):
        optimizers = super().make_optimizers()
        config = self.config["torch_optimizer"]["models"]
        optimizers["models"] = build_optimizer(self.module.models, config)
        return optimizers

    def compile(self):
        super().compile()
        self.loss_model.compile()
        self.loss_actor.compile()

    @override(ModelTrainingMixin)
    def optimize_model(self, *args, **kwargs):
        # pylint:disable=signature-differs
        self.loss_model.alpha = self.module.alpha().item()
        return super().optimize_model(*args, **kwargs)
