# pylint:disable=missing-docstring,invalid-name
from typing import List
from typing import Tuple
from typing import Union

import gym
import numpy as np
import torch
from torch import Tensor


DEFAULT_CONFIG = {
    "MAX_RES_CAP": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    "UPPER_BOUND": [62.5, 62.5, 62.5, 62.5, 62.5, 62.5, 62.5, 62.5],
    "LOWER_BOUND": [37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5],
    "RAIN_SHAPE": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    "RAIN_SCALE": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    "DOWNSTREAM": [
        [False, True, False, False, False, False, False, False],
        [False, False, True, False, False, False, False, False],
        [False, False, False, True, False, False, False, False],
        [False, False, False, False, True, False, False, False],
        [False, False, False, False, False, True, False, False],
        [False, False, False, False, False, False, True, False],
        [False, False, False, False, False, False, False, True],
        [False, False, False, False, False, False, False, False],
    ],
    "SINK_RES": [False, False, False, False, False, False, False, True],
    "MAX_WATER_EVAP_FRAC_PER_TIME_UNIT": 0.05,
    "LOW_PENALTY": [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
    "HIGH_PENALTY": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "init": {"rlevel": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]},
    "horizon": 40,
}


class ReservoirEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        self._config = self._convert_to_tensor({**DEFAULT_CONFIG, **(config or {})})

        self._num_reservoirs = len(self._config["init"]["rlevel"])

        self.action_space = gym.spaces.Box(
            low=np.array([0.0] * self._num_reservoirs, dtype=np.float32),
            high=np.array([1.0] * self._num_reservoirs, dtype=np.float32),
        )

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * (self._num_reservoirs + 1), dtype=np.float32),
            high=np.array([np.inf] * (self._num_reservoirs) + [1.0], dtype=np.float32),
        )

        self._state = None
        self.reset()

        self._horizon = self._config["horizon"]

    @staticmethod
    def _convert_to_tensor(config: dict) -> dict:
        conf_ = config.copy()
        for key in [
            "RAIN_SHAPE",
            "RAIN_SCALE",
            "DOWNSTREAM",
            "MAX_RES_CAP",
            "LOWER_BOUND",
            "UPPER_BOUND",
            "LOW_PENALTY",
            "HIGH_PENALTY",
        ]:
            conf_[key] = torch.as_tensor(conf_[key])
        return conf_

    def reset(self):
        self._state = np.array(self._config["init"]["rlevel"] + [0.0])
        return self._state

    @torch.no_grad()
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        state, action = map(torch.as_tensor, (self._state, action))
        next_state, _ = self.dynamics_fn(state, action)
        reward = self.reward_fn(state, action, next_state).item()
        self._state = next_state.numpy()
        return self._state, reward, self._terminal(), {}

    def dynamics_fn(
        self, state: Tensor, action: Tensor, sample_shape: List[int] = ()
    ) -> Tuple[Tensor, Tensor]:
        # pylint:disable=missing-docstring
        state, time = self._unpack_state(state)
        rain, logp = self._rainfall(state, sample_shape)
        action = torch.as_tensor(action) * state
        next_state = self._rlevel(state, action, rain)
        time = self._step_time(time)
        time = time.expand_as(next_state[..., -1:])
        return torch.cat([next_state, time], dim=-1), logp

    def _rlevel(self, rlevel: Tensor, action: Tensor, rain: Tensor) -> Tensor:
        MIN_RES_CAP = torch.zeros(self._num_reservoirs)
        outflow = torch.as_tensor(action)
        rlevel = rlevel + rain - self._evaporated(rlevel)
        rlevel = (
            rlevel
            + self._inflow(rlevel, action)
            - outflow
            - self._overflow(rlevel, action)
        )
        return torch.max(MIN_RES_CAP, rlevel)

    def _rainfall(
        self, rlevel: Tensor, sample_shape: List[int] = ()
    ) -> Tuple[Tensor, Tensor]:
        concentration = self._config["RAIN_SHAPE"].expand_as(rlevel)
        rate = 1.0 / self._config["RAIN_SCALE"].expand_as(rlevel)
        dist = torch.distributions.Independent(
            torch.distributions.Gamma(concentration, rate), reinterpreted_batch_ndims=1
        )
        sample = dist.rsample(sample_shape)
        logp = dist.log_prob(sample.detach())
        return sample, logp

    def _inflow(self, rlevel: Tensor, action: Tensor) -> Tensor:
        DOWNSTREAM = self._config["DOWNSTREAM"].float()
        overflow = self._overflow(rlevel, action)
        outflow = action
        return (overflow + outflow).matmul(DOWNSTREAM.t())

    def _overflow(self, rlevel: Tensor, action: Tensor) -> Tensor:
        MIN_RES_CAP = torch.zeros(self._num_reservoirs)
        MAX_RES_CAP = self._config["MAX_RES_CAP"]
        outflow = torch.as_tensor(action)
        return torch.max(MIN_RES_CAP, rlevel - outflow - MAX_RES_CAP)

    def _evaporated(self, rlevel: Tensor) -> Tensor:
        # EVAP_PER_TIME_UNIT = self._config["MAX_WATER_EVAP_FRAC_PER_TIME_UNIT"]
        MAX_RES_CAP = self._config["MAX_RES_CAP"]
        # return (
        #     EVAP_PER_TIME_UNIT
        #     * torch.log(1.0 + rlevel)
        #     * (rlevel ** 2)
        #     / (MAX_RES_CAP ** 2)
        # )
        return 1 / 2 * torch.sin(rlevel / MAX_RES_CAP) * rlevel

    def _step_time(self, time: Tensor) -> Tensor:
        timestep = torch.round(self._horizon * time)
        return torch.clamp((timestep + 1) / self._horizon, 0, 1)

    def reward_fn(self, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        # pylint:disable=unused-argument,missing-docstring
        rlevel, _ = self._unpack_state(next_state)

        LOWER_BOUND = self._config["LOWER_BOUND"]
        UPPER_BOUND = self._config["UPPER_BOUND"]

        LOW_PENALTY = self._config["LOW_PENALTY"]
        HIGH_PENALTY = self._config["HIGH_PENALTY"]

        # 0.01 * abs(rlevel - (lower_bound + upper_bound) / 2) + high_penalty * max(
        #     0.0, rlevel - upper_bound
        # ) + low_penalty * max(0.0, lower_bound - rlevel)

        mean_capacity_deviation = -0.01 * torch.abs(
            rlevel - (LOWER_BOUND + UPPER_BOUND) / 2
        )
        overflow_penalty = HIGH_PENALTY * torch.max(
            torch.zeros_like(rlevel), rlevel - UPPER_BOUND
        )
        underflow_penalty = LOW_PENALTY * torch.max(
            torch.zeros_like(rlevel), LOWER_BOUND - rlevel
        )

        penalty = mean_capacity_deviation + overflow_penalty + underflow_penalty
        return penalty.sum(dim=-1)

    def _terminal(self):
        _, time = self._unpack_state(self._state)
        return time.item() >= 1.0

    def termination_fn(
        self, state: Tensor, action: Tensor, next_state: Tensor
    ) -> Tensor:
        # pylint:disable=unused-argument,missing-docstring
        _, time = self._unpack_state(next_state)
        return time[..., 0] >= 1.0

    @staticmethod
    def _unpack_state(state: Union[np.ndarray, Tensor]) -> Tuple[Tensor, Tensor]:
        obs = torch.as_tensor(state[..., :-1], dtype=torch.float32)
        time = torch.as_tensor(state[..., -1:], dtype=torch.float32)
        return obs, time

    def render(self, mode="human"):
        pass
