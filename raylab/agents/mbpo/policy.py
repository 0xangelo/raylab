"""Policy for MBPO using PyTorch."""
from raylab.agents.sac import SACTorchPolicy
from raylab.policy import EnvFnMixin
from raylab.policy import ModelSamplingMixin
from raylab.policy import ModelTrainingMixin
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.policy.losses import ModelEnsembleMLE
from raylab.torch.optim import build_optimizer


class MBPOTorchPolicy(
    EnvFnMixin, ModelTrainingMixin, ModelSamplingMixin, SACTorchPolicy
):
    """Model-Based Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=too-many-ancestors
    dist_class = WrapStochasticPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        models = self.module.models
        self.loss_model = ModelEnsembleMLE(models)

    @property
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.mbpo import MBPOTrainer

        return MBPOTrainer.options

    @property
    def model_training_loss(self):
        return self.loss_model

    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["torch_optimizer"]
        optimizers["models"] = build_optimizer(self.module.models, config["models"])
        return optimizers
