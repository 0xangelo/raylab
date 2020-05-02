"""Registry of environment reward functions to be used by algorithms."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override
from ray.tune.registry import ENV_CREATOR, _global_registry

REWARDS = {}


def get_reward_fn(env_id, env_config=None):
    """Return the reward funtion for the given environment name and configuration.

    Only returns reward functions for environments which have been registered with Tune.
    """
    assert env_id in REWARDS, f"{env_id} environment reward not registered."
    assert _global_registry.contains(
        ENV_CREATOR, env_id
    ), f"{env_id} environment not registered with Tune."

    env_config = env_config or {}
    reward_fn = REWARDS[env_id](env_config)
    if env_config.get("time_aware", False):
        reward_fn = TimeAwareRewardFn(reward_fn)
    return reward_fn


def register(*ids):
    """Register reward function class for environments with given ids."""

    def librarian(cls):
        REWARDS.update({i: cls for i in ids})
        return cls

    return librarian


class RewardFn(nn.Module):
    """Module that computes an environment's reward funtion."""

    def __init__(self, _):
        super().__init__()

    @override(nn.Module)
    def forward(self, state, action, next_state):  # pylint:disable=arguments-differ
        raise NotImplementedError


class TimeAwareRewardFn(RewardFn):
    """Wraps a reward function and removes time dimension before forwarding."""

    def __init__(self, reward_fn):
        super().__init__(None)
        self.reward_fn = reward_fn

    @override(RewardFn)
    def forward(self, state, action, next_state):
        return self.reward_fn(state[..., :-1], action, next_state[..., :-1])


@register("CartPoleSwingUp-v0", "TorchCartPoleSwingUp-v0")
class CartPoleSwingUpV0Reward(RewardFn):
    """
    Compute CartPoleSwingUp's reward given a possibly batched transition.
    Assumes all but the last dimension are batch ones.
    """

    @override(RewardFn)
    def forward(self, state, action, next_state):
        return next_state[..., 2]


@register("CartPoleSwingUp-v1", "TorchCartPoleSwingUp-v1")
class CartPoleSwingUpV1Reward(RewardFn):
    """
    Compute CartPoleSwingUp's reward given a possibly batched transition.
    Assumes all but the last dimension are batch ones.
    """

    @override(RewardFn)
    def forward(self, state, action, next_state):
        return (1 + next_state[..., 2]) / 2


@register("HalfCheetah-v3")
class HalfCheetahReward(RewardFn):
    """Compute rewards given a possibly batched transition.

    Assumes all but the last dimension are batch ones.
    """

    def __init__(self, config):
        super().__init__(config)
        assert (
            config.get("exclude_current_positions_from_observation", True) is False
        ), "Need x position for HalfCheetah-v3 reward function"
        self.delta_t = 0.05
        self._ctrl_cost_weight = config.get("ctrl_cost_weight", 0.1)
        self._forward_reward_weight = config.get("forward_reward_weight", 1.0)

    @override(RewardFn)
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
        from .hvac import DEFAULT_CONFIG

        config = {**DEFAULT_CONFIG, **config}
        self.air_max = torch.as_tensor(config["AIR_MAX"]).float()
        self.is_room = torch.as_tensor(config["IS_ROOM"])
        self.cost_air = torch.as_tensor(config["COST_AIR"]).float()
        self.temp_low = torch.as_tensor(config["TEMP_LOW"]).float()
        self.temp_up = torch.as_tensor(config["TEMP_UP"]).float()
        self.penalty = torch.as_tensor(config["PENALTY"]).float()

    @override(RewardFn)
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


@register("IndustrialBenchmark")
class IndustrialBenchmarkReward(RewardFn):
    """IndustrialBenchmarks's reward function."""

    def __init__(self, config):
        super().__init__(config)
        from .industrial_benchmark.ids import IDS

        self.reward_type = config.get("reward_type", "classic")
        self.crf = IDS.CRF
        self.crc = IDS.CRC

    @override(RewardFn)
    def forward(self, state, action, next_state):
        fat_coeff, con_coeff = self.crf, self.crc
        fatigue_ = next_state[..., 4]
        consumption = next_state[..., 5]
        reward = -(fat_coeff * fatigue_ + con_coeff * consumption)
        if self.reward_type == "delta":
            old_fat = state[..., 4]
            old_con = state[..., 5]
            reward = reward + fat_coeff * old_fat + con_coeff * old_con
        return reward / 100


@register("Navigation")
class NavigationReward(RewardFn):
    """Navigation's reward function."""

    def __init__(self, config):
        super().__init__(config)
        from .navigation import DEFAULT_CONFIG

        config = {**DEFAULT_CONFIG, **config}
        self._end = torch.as_tensor(config["end"]).float()

    @override(RewardFn)
    def forward(self, state, action, next_state):
        next_state = next_state[..., :2]
        goal = self._end
        return -torch.norm(next_state - goal, p=2, dim=-1)


@register("Reacher-v2")
class ReacherReward(RewardFn):
    """Reacher-v3's reward function."""

    @override(RewardFn)
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
        from .reservoir import DEFAULT_CONFIG

        config = {**DEFAULT_CONFIG, **config}
        self.lower_bound = torch.as_tensor(config["LOWER_BOUND"])
        self.upper_bound = torch.as_tensor(config["UPPER_BOUND"])

        self.low_penalty = torch.as_tensor(config["LOW_PENALTY"])
        self.high_penalty = torch.as_tensor(config["HIGH_PENALTY"])

    @override(RewardFn)
    def forward(self, state, action, next_state):
        rlevel = next_state[..., :-1]

        penalty = torch.where(
            (rlevel >= self.lower_bound) & (rlevel <= self.upper_bound),
            torch.zeros_like(rlevel),
            torch.where(
                rlevel < self.lower_bound,
                self.low_penalty * (self.lower_bound - rlevel),
                self.high_penalty * (rlevel - self.upper_bound),
            ),
        )

        return penalty.sum(dim=-1)


@register("MountainCarContinuous-v0")
class MountainCarContinuousReward(RewardFn):
    """MountainCarContinuous' reward function."""

    def __init__(self, config):
        super().__init__(config)
        goal_position = 0.45
        goal_velocity = config.get("goal_velocity", 0.0)
        self.goal = torch.as_tensor([goal_position, goal_velocity])

    @override(RewardFn)
    def forward(self, state, action, next_state):
        done = (next_state >= self.goal).all(-1)
        shape = state.shape[:-1]
        reward = torch.where(done, torch.empty(shape).fill_(200), torch.zeros(shape))
        reward -= torch.pow(action, 2).squeeze(-1) * 0.1
        return reward


@register("ReacherBulletEnv-v0")
class ReacherBulletEnvReward(RewardFn):
    """ReacherBulletEnv-v0's reward function."""

    @override(RewardFn)
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
