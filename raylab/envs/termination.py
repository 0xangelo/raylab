"""Registry of environment termination functions in PyTorch."""
import torch.nn as nn
from ray.rllib.utils import override
from ray.tune.registry import _global_registry
from ray.tune.registry import ENV_CREATOR

TERMINATIONS = {}


def get_termination_fn(env_id, env_config=None):
    """Return the termination funtion for the given environment name and configuration.

    Only returns for environments which have been registered with Tune.
    """
    assert env_id in TERMINATIONS, f"{env_id} environment termination not registered."
    assert _global_registry.contains(
        ENV_CREATOR, env_id
    ), f"{env_id} environment not registered with Tune."

    env_config = env_config or {}
    termination_fn = TERMINATIONS[env_id](env_config)
    if env_config.get("time_aware", False):
        termination_fn = TimeAwareTerminationFn(termination_fn)
    return termination_fn


def register(*ids):
    """Register termination function class for environments with given ids."""

    def librarian(cls):
        TERMINATIONS.update({i: cls for i in ids})
        return cls

    return librarian


class TerminationFn(nn.Module):
    """
    Module that computes an environment's termination function for batches of inputs.
    """

    def __init__(self, _):
        super().__init__()

    @override(nn.Module)
    def forward(self, state, action, next_state):  # pylint:disable=arguments-differ
        raise NotImplementedError


class TimeAwareTerminationFn(TerminationFn):
    """Wraps a termination function and removes time dimension before forwarding."""

    def __init__(self, termination_fn):
        super().__init__(None)
        self.termination_fn = termination_fn

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        timeout = next_state[..., -1] >= 1.0
        env_done = self.termination_fn(state[..., :-1], action, next_state[..., :-1])
        return timeout | env_done
