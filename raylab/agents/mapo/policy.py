"""Policy for MAPO using PyTorch."""
from ray.rllib.utils import override

from raylab.agents.sac import SACTorchPolicy
from raylab.policy import EnvFnMixin
from raylab.policy import ModelTrainingMixin
from raylab.policy.losses import MAPO
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.losses import SPAML
from raylab.pytorch.optim import build_optimizer


class MAPOTorchPolicy(ModelTrainingMixin, EnvFnMixin, SACTorchPolicy):
    """Model-Aware Policy Optimization policy in PyTorch to use with RLlib."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._setup_model_loss()

    @property
    def options(self):
        """Return the default configuration for MAPO."""
        # pylint:disable=cyclic-import
        from raylab.agents.mapo import MAPOTrainer

        return MAPOTrainer.options

    @property
    def model_training_loss(self):
        return self.loss_paml

    @property
    def model_warmup_loss(self):
        return self.loss_mle

    def compile(self):
        super().compile()
        self.loss_paml.compile()
        self.loss_mle.compile()
        self.loss_actor.compile()

    @override(ModelTrainingMixin)
    def optimize_model(self, *args, **kwargs):
        # pylint:disable=signature-differs
        self.loss_paml.alpha = self.module.alpha().item()
        return super().optimize_model(*args, **kwargs)

    def _setup_model_loss(self):
        self.loss_paml = SPAML(
            self.module.models, self.module.actor, self.module.critics
        )
        self.loss_paml.gamma = self.config["gamma"]
        self.loss_paml.manhattan = self.config["losses"]["manhattan"]
        self.loss_paml.grad_estimator = self.config["losses"]["grad_estimator"]
        self.loss_paml.lambda_ = self.config["losses"]["lambda"]

        self.loss_mle = ModelEnsembleMLE(self.module.models)

    @override(SACTorchPolicy)
    def _setup_actor_loss(self):
        self.loss_actor = MAPO(
            self.module.models, self.module.actor, self.module.critics
        )
        self.loss_actor.gamma = self.config["gamma"]
        self.loss_actor.model_samples = self.config["losses"]["model_samples"]
        self.loss_actor.grad_estimator = self.config["losses"]["grad_estimator"]

    @override(SACTorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["torch_optimizer"]["models"]
        optimizers["models"] = build_optimizer(self.module.models, config)
        return optimizers

    @override(EnvFnMixin)
    def _set_reward_hook(self):
        self.loss_paml.set_reward_fn(self.reward_fn)
        self.loss_actor.set_reward_fn(self.reward_fn)

    @override(EnvFnMixin)
    def _set_termination_hook(self):
        self.loss_paml.set_termination_fn(self.termination_fn)
        self.loss_actor.set_termination_fn(self.termination_fn)

    @override(EnvFnMixin)
    def _set_dynamics_hook(self):
        raise ValueError(
            """Model-Aware Policy Optimization shouldn't use ground-truth \
            dynamics. Refer to DAPO instead."""
        )
