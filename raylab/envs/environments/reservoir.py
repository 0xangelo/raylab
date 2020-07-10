# pylint:disable=missing-docstring,invalid-name
import gym
import numpy as np
import torch


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
            high=np.array([np.inf] * (self._num_reservoirs) + [1.0], dtype=np.float32),
        )

        self._state = None
        self.reset()

        self._horizon = self._config["horizon"]

    def reset(self):
        self._state = np.array(self._config["init"]["rlevel"] + [0.0])
        return self._state

    @torch.no_grad()
    def step(self, action):
        state, action = map(torch.as_tensor, (self._state, action))
        next_state, _ = self.transition_fn(state, action)
        reward = self.reward_fn(state, action, next_state).item()
        self._state = next_state.numpy()
        return self._state, reward, self._terminal(), {}

    def transition_fn(self, state, action, sample_shape=()):
        # pylint:disable=missing-docstring
        state, time = self._unpack_state(state)
        rain, logp = self._rainfall(state, sample_shape)
        action = torch.as_tensor(action) * state
        next_state = self._rlevel(state, action, rain)
        time = self._step_time(time)
        time = time.expand_as(next_state[..., -1:])
        return torch.cat([next_state, time], dim=-1), logp

    def _rlevel(self, rlevel, action, rain):
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

    def _rainfall(self, rlevel, sample_shape=()):
        concentration = torch.as_tensor(self._config["RAIN_SHAPE"]).expand_as(rlevel)
        rate = 1.0 / torch.as_tensor(self._config["RAIN_SCALE"]).expand_as(rlevel)
        dist = torch.distributions.Independent(
            torch.distributions.Gamma(concentration, rate), reinterpreted_batch_ndims=1
        )
        sample = dist.rsample(sample_shape)
        logp = dist.log_prob(sample.detach())
        return sample, logp

    def _inflow(self, rlevel, action):
        DOWNSTREAM = torch.as_tensor(self._config["DOWNSTREAM"], dtype=torch.float32)
        overflow = self._overflow(rlevel, action)
        outflow = action
        return (overflow + outflow).matmul(DOWNSTREAM.t())

    def _overflow(self, rlevel, action):
        MIN_RES_CAP = torch.zeros(self._num_reservoirs)
        MAX_RES_CAP = torch.as_tensor(self._config["MAX_RES_CAP"])
        outflow = torch.as_tensor(action)
        return torch.max(MIN_RES_CAP, rlevel - outflow - MAX_RES_CAP)

    def _evaporated(self, rlevel):
        EVAP_PER_TIME_UNIT = self._config["MAX_WATER_EVAP_FRAC_PER_TIME_UNIT"]
        MAX_RES_CAP = torch.as_tensor(self._config["MAX_RES_CAP"])
        return (
            EVAP_PER_TIME_UNIT
            * torch.log(1.0 + rlevel)
            * (rlevel ** 2)
            / (MAX_RES_CAP ** 2)
        )

    def _step_time(self, time):
        timestep = torch.round(self._horizon * time)
        return torch.clamp((timestep + 1) / self._horizon, 0, 1)

    def reward_fn(self, state, action, next_state):
        # pylint:disable=unused-argument,missing-docstring
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
        return time.item() >= 1.0

    @staticmethod
    def _unpack_state(state):
        obs = torch.as_tensor(state[..., :-1], dtype=torch.float32)
        time = torch.as_tensor(state[..., -1:], dtype=torch.float32)
        return obs, time

    def render(self, mode="human"):
        pass
