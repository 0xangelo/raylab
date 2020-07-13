"""Policy for Dyna-SAC in PyTorch."""
from raylab.agents.sac import SACTorchPolicy
from raylab.policy.losses import DynaSoftCDQLearning
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.model_based import EnvFnMixin
from raylab.policy.model_based import ModelTrainingMixin
from raylab.pytorch.optim.utils import build_optimizer


class DynaSACTorchPolicy(ModelTrainingMixin, EnvFnMixin, SACTorchPolicy):
    """Model-based policy por Dyna-SAC."""

    # pylint:disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_model_loss()
        self.optimizers["models"] = build_optimizer(
            self.module.models, config=self.config["torch_optimizer"]["models"]
        )

    def _setup_model_loss(self):
        self.loss_model = ModelEnsembleMLE(self.module.models)

    def _setup_critic_loss(self):
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

    def _set_reward_hook(self):
        self.loss_critic.set_reward_fn(self.reward_fn)

    def _set_termination_hook(self):
        self.loss_critic.set_termination_fn(self.termination_fn)
