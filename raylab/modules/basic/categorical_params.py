# pylint: disable=missing-docstring
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import initialize_


class CategoricalParams(nn.Module):
    """Neural network module mapping inputs to Categorical parameters.

    This module is initialized to be close to a discrete uniform distribution.
    """

    def __init__(self, in_features, n_categories):
        super().__init__()
        self.logits_module = nn.Linear(in_features, n_categories)
        self.apply(initialize_("orthogonal", gain=0.01))

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        logits = self.logits_module(inputs)
        return {"logits": logits - logits.logsumexp(dim=-1, keepdim=True)}
