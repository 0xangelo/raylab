"""Neural network modules for Stochastic Value Gradients."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class ParallelDynamicsModel(nn.Module):
    """
    Neural network module mapping inputs to distribution parameters through parallel
    subnetworks for each output dimension.
    """

    def __init__(self, *logits_modules):
        super().__init__()
        self.logits_modules = logits_modules
        self.loc_modules = [nn.Linear(m.out_features, 1) for m in self.logits_modules]
        self.scale_diag = nn.Parameter(torch.zeros(len(self.logits_modules)))

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        logits = [m(inputs) for m in self.logits_modules]
        loc = torch.cat([m(i) for m, i in zip(self.loc_modules, logits)], dim=-1)
        return loc, self.scale_diag
