# pylint:disable=missing-module-docstring
from raylab.agents.mapo import MAPOTorchPolicy
from raylab.policy.losses import DynaSoftCDQLearning


class MAPOPlusTorchPolicy(MAPOTorchPolicy):
    """MAPO PyTorch policy with Dyna Q-Learning."""

    # pylint:disable=too-many-ancestors

    @property
    def options(self):
        from raylab.agents.mapo.plus import MAPOPlusTrainer

        return MAPOPlusTrainer.options

    def _setup_critic_loss(self):
        self.loss_critic = DynaSoftCDQLearning(
            self.module.critics,
            self.module.models,
            self.module.target_critics,
            self.module.actor,
        )
        self.loss_critic.gamma = self.config["gamma"]
        self.loss_critic.seed(self.config["seed"])
