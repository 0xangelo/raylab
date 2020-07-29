"""Registry of environment reward functions in PyTorch."""
import math
from typing import Callable
from typing import Optional

import torch
import torch.nn as nn

from raylab.tune.registry import _raylab_registry
from raylab.tune.registry import RAYLAB_REWARD

from .utils import get_env_parameters
from .utils import has_env_creator

# For testing purposes
REWARDS = {}


def register(*ids: str) -> Callable[[type], type]:
    """Register reward function class for environments with given ids."""

    def librarian(cls):
        for id_ in ids:
            REWARDS[id_] = cls
            _raylab_registry.register(RAYLAB_REWARD, id_, cls)

        return cls

    return librarian


def has_reward_fn(env_id: str) -> bool:
    """Whether the environment id has a reward function in the global registry."""
    return _raylab_registry.contains(RAYLAB_REWARD, env_id)


def get_reward_fn(env_id: str, env_config: Optional[dict] = None) -> "RewardFn":
    """Return the reward funtion for the given environment name and configuration.

    Only returns reward functions for environments which have been registered with Tune.
    """
    assert has_env_creator(env_id), f"{env_id} environment not registered with Tune."
    assert has_reward_fn(env_id), f"{env_id} environment reward not registered."

    env_config = env_config or {}
    reward_fn = _raylab_registry.get(RAYLAB_REWARD, env_id)(env_config)
    if env_config.get("time_aware", False):
        reward_fn = TimeAwareRewardFn(reward_fn)
    return reward_fn


class RewardFn(nn.Module):
    """Module that computes an environment's reward function for batches of inputs."""

    def __init__(self, _):
        super().__init__()

    def forward(self, state, action, next_state):  # pylint:disable=arguments-differ
        raise NotImplementedError


class TimeAwareRewardFn(RewardFn):
    """Wraps a reward function and removes time dimension before forwarding."""

    def __init__(self, reward_fn):
        super().__init__(None)
        self.reward_fn = reward_fn

    def forward(self, state, action, next_state):
        return self.reward_fn(state[..., :-1], action, next_state[..., :-1])


################################################################################
# Built-ins
################################################################################


@register("CartPoleSwingUp-v0", "TorchCartPoleSwingUp-v0")
class CartPoleSwingUpV0Reward(RewardFn):
    """
    Compute CartPoleSwingUp's reward given a possibly batched transition.
    Assumes all but the last dimension are batch ones.
    """

    def forward(self, state, action, next_state):
        return next_state[..., 2]


@register("CartPoleSwingUp-v1", "TorchCartPoleSwingUp-v1")
class CartPoleSwingUpV1Reward(RewardFn):
    """
    Compute CartPoleSwingUp's reward given a possibly batched transition.
    Assumes all but the last dimension are batch ones.
    """

    def forward(self, state, action, next_state):
        return (1 + next_state[..., 2]) / 2


@register("HalfCheetah-v3")
class HalfCheetahReward(RewardFn):
    """Compute rewards given a possibly batched transition.

    Assumes all but the last dimension are batch ones.
    """

    def __init__(self, config):
        super().__init__(config)
        parameters = get_env_parameters("HalfCheetah-v3")
        for attr in """
        ctrl_cost_weight
        forward_reward_weight
        exclude_current_positions_from_observation
        """.split():
            setattr(self, "_" + attr, config.get(attr, parameters[attr].default))

        assert (
            self._exclude_current_positions_from_observation is False
        ), "Need x position for HalfCheetah-v3 reward function"
        self.delta_t = 0.05

    def forward(self, state, action, next_state):
        x_position_before = state[..., 0]
        x_position_after = next_state[..., 0]
        x_velocity = (x_position_after - x_position_before) / self.delta_t

        control_cost = self._ctrl_cost_weight * (action ** 2).sum(dim=-1)

        forward_reward = self._forward_reward_weight * x_velocity

        return forward_reward - control_cost


@register("HVAC")
class HVACReward(RewardFn):
    """Compute HVAC's reward function."""

    def __init__(self, config):
        super().__init__(config)
        from .environments.hvac import DEFAULT_CONFIG

        config = {**DEFAULT_CONFIG, **config}
        self.air_max = torch.as_tensor(config["AIR_MAX"]).float()
        self.is_room = torch.as_tensor(config["IS_ROOM"])
        self.cost_air = torch.as_tensor(config["COST_AIR"]).float()
        self.temp_low = torch.as_tensor(config["TEMP_LOW"]).float()
        self.temp_up = torch.as_tensor(config["TEMP_UP"]).float()
        self.penalty = torch.as_tensor(config["PENALTY"]).float()

    def forward(self, state, action, next_state):
        air = action * self.air_max
        temp = next_state[..., :-1]

        reward = -(
            self.is_room
            * (
                air * self.cost_air
                + ((temp < self.temp_low) | (temp > self.temp_up)) * self.penalty
                + 10.0 * torch.abs((self.temp_up + self.temp_low) / 2.0 - temp)
            )
        ).sum(dim=-1)

        return reward


@register("IndustrialBenchmark-v0")
class IndustrialBenchmarkReward(RewardFn):
    """IndustrialBenchmarks's reward function."""

    def __init__(self, config):
        super().__init__(config)
        self.reward_type = config.get("reward_type", "classic")

    def forward(self, state, action, next_state):
        con_coeff, fat_coeff = 1, 3
        consumption, fatigue = next_state[..., 4], next_state[..., 5]
        reward = -(con_coeff * consumption + fat_coeff * fatigue)

        if self.reward_type == "delta":
            old_consumption, old_fatigue = state[..., 4], state[..., 5]
            old_reward = -(con_coeff * old_consumption + fat_coeff * old_fatigue)
            reward -= old_reward

        return reward / 100


@register("Navigation")
class NavigationReward(RewardFn):
    """Navigation's reward function."""

    def __init__(self, config):
        super().__init__(config)
        from .environments.navigation import DEFAULT_CONFIG

        config = {**DEFAULT_CONFIG, **config}
        self._end = torch.as_tensor(config["end"]).float()

    def forward(self, state, action, next_state):
        next_state = next_state[..., :2]
        goal = self._end
        return -torch.norm(next_state - goal, p=2, dim=-1)


@register("Reacher-v2")
class ReacherReward(RewardFn):
    """Reacher-v3's reward function."""

    def forward(self, state, action, next_state):
        dist = state[..., -3:]
        reward_dist = -torch.norm(dist, dim=-1)
        reward_ctrl = -torch.sum(action ** 2, dim=-1)
        return reward_dist + reward_ctrl


@register("Reservoir")
class ReservoirReward(RewardFn):
    """Reservoir's reward function."""

    def __init__(self, config):
        super().__init__(config)
        from .environments.reservoir import DEFAULT_CONFIG

        config = {**DEFAULT_CONFIG, **config}
        self.lower_bound = torch.as_tensor(config["LOWER_BOUND"])
        self.upper_bound = torch.as_tensor(config["UPPER_BOUND"])

        self.low_penalty = torch.as_tensor(config["LOW_PENALTY"])
        self.high_penalty = torch.as_tensor(config["HIGH_PENALTY"])

    def forward(self, state, action, next_state):
        rlevel = next_state[..., :-1]

        mean_capacity_deviation = -0.01 * torch.abs(
            rlevel - (self.lower_bound + self.upper_bound) / 2
        )
        overflow_penalty = self.high_penalty * torch.max(
            torch.zeros_like(rlevel), rlevel - self.upper_bound
        )
        underflow_penalty = self.low_penalty * torch.max(
            torch.zeros_like(rlevel), self.lower_bound - rlevel
        )

        penalty = mean_capacity_deviation + overflow_penalty + underflow_penalty
        return penalty.sum(dim=-1)


@register("MountainCarContinuous-v0")
class MountainCarContinuousReward(RewardFn):
    """MountainCarContinuous' reward function."""

    def __init__(self, config):
        super().__init__(config)
        goal_position = 0.45
        goal_velocity = config.get("goal_velocity", 0.0)
        self.goal = torch.as_tensor([goal_position, goal_velocity])

    def forward(self, state, action, next_state):
        done = (next_state >= self.goal).all(-1)
        shape = state.shape[:-1]
        reward = torch.where(done, torch.empty(shape).fill_(200), torch.zeros(shape))
        reward -= torch.pow(action, 2).squeeze(-1) * 0.1
        return reward


@register("ReacherBulletEnv-v0")
class ReacherBulletEnvReward(RewardFn):
    """ReacherBulletEnv-v0's reward function."""

    def forward(self, state, action, next_state):
        to_target_vec_old = state[..., 2:4]
        to_target_vec_new = next_state[..., 2:4]
        potential_old = -100 * to_target_vec_old.norm(p=2, dim=-1)
        potential_new = -100 * to_target_vec_new.norm(p=2, dim=-1)

        theta_dot = next_state[..., -3]
        gamma = next_state[..., -2]
        gamma_dot = next_state[..., -1]

        electricity_cost = -0.10 * (
            (action[..., 0] * theta_dot).abs() + (action[..., 1] * gamma_dot).abs()
        ) - 0.01 * (action[..., 0].abs() + action[..., 1].abs())

        stuck_joint_cost = torch.where(
            (gamma.abs() - 1).abs() < 0.01,
            torch.empty_like(electricity_cost).fill_(-0.1),
            torch.zeros_like(electricity_cost),
        )

        rewards = (potential_new - potential_old) + electricity_cost + stuck_joint_cost
        return rewards


@register("Pendulum-v0")
class PendulumReward(RewardFn):
    """Pendulum-v0's reward function."""

    def __init__(self, config):
        super().__init__(config)
        self.max_torque = 2.0

    def forward(self, state, action, next_state):
        # pylint:disable=invalid-name
        cos_th, sin_th, thdot = state[..., 0], state[..., 1], state[..., 2]
        th = torch.atan2(sin_th, cos_th)
        u = action[..., 0]

        # angle_normalize
        th = ((th + math.pi) % (2 * math.pi)) - math.pi

        u = torch.clamp(u, -self.max_torque, self.max_torque)
        costs = (th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        return -costs


@register("Pusher-v2")
class PusherReward(RewardFn):
    """Pusher-v2's reward function."""

    vec_size: int = 3

    def forward(self, state, action, next_state):
        idx, vec_size = 14, self.vec_size
        tips_arm = state[..., idx : idx + vec_size]
        idx += vec_size
        obj = state[..., idx : idx + vec_size]
        idx += vec_size
        goal = state[..., idx : idx + vec_size]
        vec_1 = obj - tips_arm
        vec_2 = obj - goal

        reward_near = -torch.norm(vec_1, dim=-1)
        reward_dist = -torch.norm(vec_2, dim=-1)
        reward_ctrl = -torch.square(action).sum(dim=-1)
        return reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near


@register("Walker2d-v3")
class Walker2DReward(RewardFn):
    """Walker2d-v3's reward function."""

    def __init__(self, config):
        super().__init__(config)
        parameters = get_env_parameters("Walker2d-v3")
        for attr in """
        ctrl_cost_weight
        forward_reward_weight
        healthy_reward
        terminate_when_unhealthy
        healthy_z_range
        healthy_angle_range
        exclude_current_positions_from_observation
        """.split():
            setattr(self, "_" + attr, config.get(attr, parameters[attr].default))

        assert (
            not self._exclude_current_positions_from_observation
        ), "Need x position for Walkered-v3 reward function"
        self.delta_t = 0.008

    def forward(self, state, action, next_state):
        x_velocity = (next_state[..., 0] - state[..., 0]) / self.delta_t
        ctrl_cost = self._control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity
        if self._terminate_when_unhealthy:
            healthy_reward = torch.empty_like(forward_reward).fill_(
                self._healthy_reward
            )
        else:
            healthy_reward = torch.where(
                self._is_healthy(next_state),
                torch.empty_like(forward_reward).fill_(self._healthy_reward),
                torch.zeros_like(forward_reward),
            )

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost

        return rewards - costs

    def _control_cost(self, action):
        control_cost = self._ctrl_cost_weight * torch.sum(torch.square(action), dim=-1)
        return control_cost

    def _is_healthy(self, state):
        # pylint:disable=invalid-name
        z, angle = state[..., 1], state[..., 2]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = (min_z < z) & (z < max_z)
        healthy_angle = (min_angle < angle) & (angle < max_angle)
        is_healthy = healthy_z & healthy_angle

        return is_healthy


@register("Swimmer-v3")
class SwimmerReward(RewardFn):
    """Swimmer-v3's reward function."""

    def __init__(self, config):
        super().__init__(config)
        parameters = get_env_parameters("Swimmer-v3")
        for attr in """
        ctrl_cost_weight
        forward_reward_weight
        exclude_current_positions_from_observation
        """.split():
            setattr(self, "_" + attr, config.get(attr, parameters[attr].default))

        assert (
            not self._exclude_current_positions_from_observation
        ), "Need x position for Swimmer-v3 reward function"
        self.delta_t = 0.04

    def forward(self, state, action, next_state):
        x_velocity = (next_state[..., 0] - state[..., 0]) / self.delta_t
        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self._control_cost(action)

        return forward_reward - ctrl_cost

    def _control_cost(self, action):
        control_cost = self._ctrl_cost_weight * torch.sum(torch.square(action), dim=-1)
        return control_cost
