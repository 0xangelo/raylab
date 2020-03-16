# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class Lambda(nn.Module):
    """Neural network module that stores and applies a function on inputs."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.func(inputs)

    @classmethod
    def as_script_module(cls, *args, **kwargs):
        func = args[0]
        assert isinstance(
            func, torch.jit.ScriptFunction
        ), "Function must be converted to TorchScript"
        return torch.jit.script(cls(*args, **kwargs))
