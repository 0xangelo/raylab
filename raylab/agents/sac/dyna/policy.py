"""Policy for Dyna-SAC in PyTorch."""
from raylab.agents.sac import SACTorchPolicy
from raylab.policy.losses import DynaSoftCDQLearning
from raylab.policy.losses import ModelEnsembleMLE
from raylab.policy.model_based import EnvFnMixin
from raylab.policy.model_based import ModelTrainingMixin
from raylab.pytorch.optim.utils import build_optimizer


class DynaSACTorchPolicy(ModelTrainingMixin, EnvFnMixin, SACTorchPolicy):
    """Model-based policy por Dyna-SAC."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_model_loss()

    @property
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.sac.dyna import DynaSACTrainer

        return DynaSACTrainer.options

    @property
    def model_training_loss(self):
        return self.loss_model

    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        optimizers.update(
            models=build_optimizer(
                self.module.models, config=self.config["torch_optimizer"]["models"]
            )
        )
        return optimizers

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

    def _set_reward_hook(self):
        self.loss_critic.set_reward_fn(self.reward_fn)

    def _set_termination_hook(self):
        self.loss_critic.set_termination_fn(self.termination_fn)
