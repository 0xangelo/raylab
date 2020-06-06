"""Policy for MBPO using PyTorch."""
import collections

from ray.rllib.utils import override

from raylab.agents.sac import SACTorchPolicy
from raylab.losses import ModelEnsembleMLE
from raylab.policy import EnvFnMixin
from raylab.policy import ModelSamplingMixin
from raylab.policy import ModelTrainingMixin
from raylab.pytorch.optim import build_optimizer


class MBPOTorchPolicy(
    EnvFnMixin, ModelTrainingMixin, ModelSamplingMixin, SACTorchPolicy
):
    """Model-Based Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method,too-many-ancestors

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        models = self.module.models
        self.loss_model = ModelEnsembleMLE(models)

    @staticmethod
    @override(SACTorchPolicy)
    def get_default_config():
        """Return the default config for MBPO."""
        # pylint:disable=cyclic-import
        from raylab.agents.mbpo import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SACTorchPolicy)
    def make_optimizer(self):
        config = self.config["torch_optimizer"]
        components = "models actor critics alpha".split()

        optims = {k: build_optimizer(self.module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)
