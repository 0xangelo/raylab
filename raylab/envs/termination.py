"""Registry of environment termination functions in PyTorch."""
import torch
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

    @override(nn.Module)
    def forward(self, state, action, next_state):  # pylint:disable=arguments-differ
        raise NotImplementedError


class TimeAwareTerminationFn(TerminationFn):
    """Wraps a termination function and removes time dimension before forwarding."""

    def __init__(self, termination_fn):
        super().__init__()
        self.termination_fn = termination_fn

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        timeout = next_state[..., -1] >= 1.0
        env_done = self.termination_fn(state[..., :-1], action, next_state[..., :-1])
        return timeout | env_done


@register(
    "HalfCheetah-v2",
    "HalfCheetah-v3",
    "IndustrialBenchmark-v0",
    "IBOperationalCost-v0",
    "IBMisCalibration-v0",
    "IBFatigue-v0",
    "ReacherBulletEnv-v0",
)
class NoTermination(TerminationFn):
    """Termination function for continuing environments.

    This is usually the case for environments which need an external time limit
    enforcement. In that situation, wrap this with TimeAwareTerminationFn.
    """

    def __init__(self, _):
        super().__init__()

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        return torch.zeros(next_state.shape[:-1]).bool()


@register(
    "CartPoleSwingUp-v0",
    "CartPoleSwingUp-v1",
    "TorchCartPoleSwingUp-v0",
    "TorchCartPoleSwingUp-v1",
)
class CartPoleSwingUpTermination(TerminationFn):
    """CartPoleSwingUp's termination function."""

    def __init__(self, _):
        from gym_cartpole_swingup.envs.cartpole_swingup import CartPoleSwingUpParams

        super().__init__()
        params = CartPoleSwingUpParams()
        self.x_threshold = params.x_threshold

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        return next_state[..., 0].abs() > self.x_threshold


@register("MountainCarContinuous-v0")
class MountainCarContinuousTermination(TerminationFn):
    """MountainCarContinuous' reward function."""

    def __init__(self, config):
        super().__init__()
        goal_position = 0.45
        goal_velocity = config.get("goal_velocity", 0.0)
        self.goal = torch.as_tensor([goal_position, goal_velocity])

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        return (next_state >= self.goal).all(dim=-1)


@register("HVAC")
class HVACTermination(TerminationFn):
    """HVAC's termination function."""

    def __init__(self, _):
        super().__init__()

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        return next_state[..., -1] >= 1.0


@register("Reservoir")
class ReservoirTermination(TerminationFn):
    """Reservoir's termination function."""

    def __init__(self, _):
        super().__init__()

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        return next_state[..., -1] >= 1.0


@register("Navigation")
class NavigationTermination(TerminationFn):
    """Navigation's termination function."""

    def __init__(self, config):
        from .environments.navigation import DEFAULT_CONFIG

        super().__init__()
        config = {**DEFAULT_CONFIG, **config}
        self.end = torch.as_tensor(config["end"]).float()

    @override(TerminationFn)
    def forward(self, state, action, next_state):
        hit_goal = ((next_state[..., :2] - self.end).abs() <= 1e-1).all(dim=-1)
        timeout = next_state[..., -1] >= 1
        return hit_goal | timeout
