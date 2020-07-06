"""Policy for MBPO using PyTorch."""
from ray.rllib.utils import override

from raylab.agents.sac import SACTorchPolicy
from raylab.policy import EnvFnMixin
from raylab.policy import ModelSamplingMixin
from raylab.policy import ModelTrainingMixin
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.policy.losses import ModelEnsembleMLE
from raylab.pytorch.optim import build_optimizer


class MBPOTorchPolicy(
    EnvFnMixin, ModelTrainingMixin, ModelSamplingMixin, SACTorchPolicy
):
    """Model-Based Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method,too-many-ancestors
    dist_class = WrapStochasticPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        models = self.module.models
        self.loss_model = ModelEnsembleMLE(models)

    @staticmethod
    @override(SACTorchPolicy)
    def get_default_config():
        """Return the default config for MBPO."""
        # pylint:disable=cyclic-import,protected-access
        from raylab.agents.mbpo import MBPOTrainer

        return MBPOTrainer._default_config

    @override(SACTorchPolicy)
    def make_optimizers(self):
        config = self.config["torch_optimizer"]
        components = "models actor critics alpha".split()

        return {
            name: build_optimizer(getattr(self.module, name), config[name])
            for name in components
        }
