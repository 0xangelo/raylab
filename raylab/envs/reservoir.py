# pylint: disable=missing-docstring,invalid-name
import gym
import torch
import numpy as np


DEFAULT_CONFIG = {
    "MAX_RES_CAP": [100.0, 100.0, 200.0, 300.0, 400.0, 500.0, 800.0, 1000.0],
    "UPPER_BOUND": [80.0, 80.0, 180.0, 280.0, 380.0, 480.0, 780.0, 980.0],
    "LOWER_BOUND": [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
    "RAIN_SHAPE": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "RAIN_SCALE": [5.0, 3.0, 9.0, 7.0, 15.0, 13.0, 25.0, 30.0],
    "DOWNSTREAM": [
        [False, False, False, False, False, True, False, False],
        [False, False, True, False, False, False, False, False],
        [False, False, False, False, True, False, False, False],
        [False, False, False, False, False, False, False, True],
        [False, False, False, False, False, False, True, False],
        [False, False, False, False, False, False, True, False],
        [False, False, False, False, False, False, False, True],
        [False, False, False, False, False, False, False, False],
    ],
    "SINK_RES": [False, False, False, False, False, False, False, True],
    "MAX_WATER_EVAP_FRAC_PER_TIME_UNIT": 0.05,
    "LOW_PENALTY": [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0],
    "HIGH_PENALTY": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
    "init": {"rlevel": [75.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]},
    "horizon": 40,
}


class ReservoirEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        self._config = {**DEFAULT_CONFIG, **(config or {})}

        self._num_reservoirs = len(self._config["init"]["rlevel"])

        self.action_space = gym.spaces.Box(
            low=np.array([0.0] * self._num_reservoirs, dtype=np.float32),
            high=np.array([1.0] * self._num_reservoirs, dtype=np.float32),
        )

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * (self._num_reservoirs + 1), dtype=np.float32),
            high=np.array(self._config["MAX_RES_CAP"] + [1.0], dtype=np.float32),
        )

        self._state = None
        self.reset()

        self._horizon = self._config["horizon"]

    def reset(self):
        self._state = np.array(self._config["init"]["rlevel"] + [0.0])
        return self._state

    @property
    def rlevel(self):
        obs, _ = self._unpack_state(self._state)
        return torch.as_tensor(obs, dtype=torch.float32)

    @torch.no_grad()
    def step(self, action):
        state, action = map(torch.as_tensor, (self._state, action))
        next_state, _ = self.transition_fn(state, action)
        reward = self.reward_fn(state, action, next_state).item()
        self._state = next_state.numpy()
        return self._state, reward, self._terminal(), {}

    def transition_fn(self, state, action, sample_shape=()):
        # pylint: disable=missing-docstring
        state, time = self._unpack_state(state)
        rain, logp = self._rainfall(sample_shape)
        action = torch.as_tensor(action) * state
        next_state = self._rlevel(action, rain)
        time = torch.clamp(time + 1 / self._horizon, 0.0, 1.0).detach()
        time = time.expand_as(next_state[..., -1:])
        return torch.cat([next_state, time], dim=-1), logp

    def _rainfall(self, sample_shape=()):
        concentration = torch.as_tensor(self._config["RAIN_SHAPE"])
        rate = 1.0 / torch.as_tensor(self._config["RAIN_SCALE"])
        dist = torch.distributions.Gamma(concentration, rate)
        sample = dist.rsample(sample_shape)
        logp = dist.log_prob(sample.detach())
        return sample, logp

    def _evaporated(self):
        EVAP_PER_TIME_UNIT = self._config["MAX_WATER_EVAP_FRAC_PER_TIME_UNIT"]
        MAX_RES_CAP = torch.as_tensor(self._config["MAX_RES_CAP"])
        return (
            EVAP_PER_TIME_UNIT
            * torch.log(1.0 + self.rlevel)
            * (self.rlevel ** 2)
            / (MAX_RES_CAP ** 2)
        )

    def _overflow(self, action):
        MIN_RES_CAP = torch.zeros(self._num_reservoirs)
        MAX_RES_CAP = torch.as_tensor(self._config["MAX_RES_CAP"])
        outflow = torch.as_tensor(action)
        return torch.max(MIN_RES_CAP, self.rlevel - outflow - MAX_RES_CAP)

    def _inflow(self, action):
        DOWNSTREAM = torch.as_tensor(self._config["DOWNSTREAM"], dtype=torch.float32)
        overflow = self._overflow(action)
        outflow = torch.as_tensor(action)
        return torch.matmul(DOWNSTREAM.T, overflow + outflow)

    def _rlevel(self, action, rain):
        MIN_RES_CAP = torch.zeros(self._num_reservoirs)
        outflow = torch.as_tensor(action)
        rlevel = self.rlevel
        rlevel += rain - self._evaporated()
        rlevel += self._inflow(action) - outflow - self._overflow(action)
        return torch.max(MIN_RES_CAP, rlevel)

    def reward_fn(self, state, action, next_state):
        # pylint: disable=unused-argument,missing-docstring
        rlevel, _ = self._unpack_state(next_state)

        LOWER_BOUND = torch.as_tensor(self._config["LOWER_BOUND"])
        UPPER_BOUND = torch.as_tensor(self._config["UPPER_BOUND"])

        LOW_PENALTY = torch.as_tensor(self._config["LOW_PENALTY"])
        HIGH_PENALTY = torch.as_tensor(self._config["HIGH_PENALTY"])

        penalty = torch.where(
            (rlevel >= LOWER_BOUND) & (rlevel <= UPPER_BOUND),
            torch.zeros_like(rlevel),
            torch.where(
                rlevel < LOWER_BOUND,
                LOW_PENALTY * (LOWER_BOUND - rlevel),
                HIGH_PENALTY * (rlevel - UPPER_BOUND),
            ),
        )

        return penalty.sum(dim=-1)

    def _terminal(self):
        _, time = self._unpack_state(self._state)
        return np.allclose(time.numpy(), 1.0)

    def render(self, mode="human"):
        pass

    @staticmethod
    def _unpack_state(state):
        obs = torch.as_tensor(state[..., :-1], dtype=torch.float32)
        time = torch.as_tensor(state[..., -1], dtype=torch.float32)
        return obs, time
