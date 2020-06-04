"""Policy for MBPO using PyTorch."""
import collections

import numpy as np
import torch
from ray.rllib.utils import override

from raylab.agents.sac import SACTorchPolicy
from raylab.losses import ModelEnsembleMLE
from raylab.policy import ModelBasedMixin
from raylab.pytorch.optim import build_optimizer


class MBPOTorchPolicy(ModelBasedMixin, SACTorchPolicy):
    """Model-Based Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method

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

    @torch.no_grad()
    @override(SACTorchPolicy)
    def extra_grad_info(self, component):
        if component == "models":
            grad_norms = [
                torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")).item()
                for m in self.module.models
            ]
            return {"grad_norm(models)": np.mean(grad_norms)}
        return super().extra_grad_info(component)
