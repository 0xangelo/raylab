"""Policy for Dyna-SAC in PyTorch."""
from raylab.agents.sac import SACTorchPolicy
from raylab.policy.losses import DynaSoftCDQLearning
from raylab.policy.model_based import ModelTrainingMixin


class DynaSACTorchPolicy(ModelTrainingMixin, SACTorchPolicy):
    """Model-based policy por Dyna-SAC."""

    # pylint:disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_critic = DynaSoftCDQLearning(
            self.module.critics,
            self.module.models,
            self.module.target_critics,
            self.module.actor,
        )
        self.loss_critic.gamma = self.config["gamma"]
        self.loss_critic.seed(self.config["seed"])

    def get_default_config(self):
        # pylint:disable=cyclic-import,protected-access
        from raylab.agents.sac.dyna import DynaSACTrainer

        return DynaSACTrainer._default_config
