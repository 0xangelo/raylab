# pylint:disable=missing-module-docstring
from raylab.agents.sac import SACTorchPolicy
from raylab.policy import EnvFnMixin
from raylab.policy.losses import DAPO


class DAPOTorchPolicy(EnvFnMixin, SACTorchPolicy):
    """Dynamics-Aware Policy Optimization policy in PyTorch to use with RLlib.

    Effectively substitutes the actor loss in SAC for a dynamics-aware 1-step
    approximation of the expected Q-value.
    """

    # pylint:disable=abstract-method,attribute-defined-outside-init

    @property
    def options(self):
        """Return the default configuration for MAPO."""
        # pylint:disable=cyclic-import
        from raylab.agents.mapo.dapo import DAPOTrainer

        return DAPOTrainer.options

    def _setup_actor_loss(self):
        # Can only be defined once we have access to the env dynamics
        self.loss_actor = None

    def _set_reward_hook(self):
        if self.loss_actor:
            self.loss_actor.set_reward_fn(self.reward_fn)

    def _set_termination_hook(self):
        if self.loss_actor:
            self.loss_actor.set_termination_fn(self.termination_fn)

    def _set_dynamics_hook(self):
        self.loss_actor = DAPO(self.dynamics_fn, self.module.actor, self.module.critics)
        self.loss_actor.gamma = self.config["gamma"]
        self.loss_actor.dynamics_samples = self.config["losses"]["model_samples"]
        self.loss_actor.grad_estimator = self.config["losses"]["grad_estimator"]

        if self.reward_fn:
            self.loss_actor.set_reward_fn(self.reward_fn)
        if self.termination_fn:
            self.loss_actor.set_termination_fn(self.termination_fn)
