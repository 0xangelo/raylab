# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class RewardFn(nn.Module):
    """Module encapsulating an environment's reward funtion."""

    def __init__(self, obs_space, action_space, reward_fn, torch_script=True):
        super().__init__()
        if torch_script:
            obs = torch.as_tensor(obs_space.sample())[None]
            action = torch.as_tensor(action_space.sample())[None]
            new_obs = torch.as_tensor(obs_space.sample())[None]
            reward_fn = torch.jit.trace(reward_fn, (obs, action, new_obs))
        self.reward_fn = reward_fn

    @override(nn.Module)
    def forward(self, obs, action, new_obs):  # pylint:disable=arguments-differ
        return self.reward_fn(obs, action, new_obs)
