# pylint:disable=missing-docstring,invalid-name
import gym
import numpy as np
import torch


DEFAULT_CONFIG = {
    "ADJ": [[False, True, True], [False, False, True], [False, False, False]],
    "ADJ_OUTSIDE": [True, True, False],
    "ADJ_HALL": [True, False, True],
    "R_OUTSIDE": [4.0, 4.0, 4.0],
    "R_HALL": [2.0, 2.0, 2.0],
    "R_WALL": [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]],
    "IS_ROOM": [True, True, True],
    "CAP": [80.0, 80.0, 80.0],
    "CAP_AIR": 1.006,
    "COST_AIR": 1.0,
    "TIME_DELTA": 1.0,
    "TEMP_AIR": 40.0,
    "TEMP_UP": [23.5, 23.5, 23.5],
    "TEMP_LOW": [20.0, 20.0, 20.0],
    "PENALTY": 20000.0,
    "AIR_MAX": [10.0, 10.0, 10.0],
    "TEMP_OUTSIDE_MEAN": [6.0, 6.0, 6.0],
    "TEMP_OUTSIDE_VARIANCE": [1.0, 1.0, 1.0],
    "TEMP_HALL_MEAN": [10.0, 10.0, 10.0],
    "TEMP_HALL_VARIANCE": [1.0, 1.0, 1.0],
    "init": {"temp": [10.0, 10.0, 10.0]},
    "horizon": 40,
}


class HVACEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        self._config = {**DEFAULT_CONFIG, **(config or {})}

        self._num_rooms = len(self._config["init"]["temp"])

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * self._num_rooms + [0.0], dtype=np.float32),
            high=np.array([np.inf] * self._num_rooms + [1.0], dtype=np.float32),
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0] * self._num_rooms, dtype=np.float32),
            high=np.array([1.0] * self._num_rooms, dtype=np.float32),
        )

        self._horizon = self._config["horizon"]
        self._state = None
        self.reset()

    def reset(self):
        self._state = np.array(self._config["init"]["temp"] + [0.0])
        return self._state

    @property
    def temp(self):
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
        # pylint:disable=missing-docstring
        state, time = self._unpack_state(state)

        AIR_MAX = torch.as_tensor(self._config["AIR_MAX"])
        action = torch.as_tensor(action) * AIR_MAX

        temp_hall, logp_temp_hall = self._temp_hall(sample_shape)
        temp_outside, logp_temp_outside = self._temp_outside(sample_shape)

        next_state = self._temp(action, temp_outside, temp_hall)
        logp = logp_temp_hall + logp_temp_outside
        time = self._step_time(time)
        time = time.expand_as(next_state[..., -1:])
        return torch.cat([next_state, time], dim=-1), logp

    def _temp_hall(self, sample_shape=()):
        TEMP_HALL_MEAN = torch.as_tensor(self._config["TEMP_HALL_MEAN"])
        TEMP_HALL_VARIANCE = torch.sqrt(
            torch.as_tensor(self._config["TEMP_HALL_VARIANCE"])
        )
        dist = torch.distributions.Normal(TEMP_HALL_MEAN, TEMP_HALL_VARIANCE)
        sample = dist.rsample(sample_shape)
        logp = dist.log_prob(sample.detach())
        return sample, logp

    def _temp_outside(self, sample_shape=()):
        TEMP_OUTSIDE_MEAN = torch.as_tensor(self._config["TEMP_OUTSIDE_MEAN"])
        TEMP_OUTSIDE_VARIANCE = torch.sqrt(
            torch.as_tensor(self._config["TEMP_OUTSIDE_VARIANCE"])
        )
        dist = torch.distributions.Normal(TEMP_OUTSIDE_MEAN, TEMP_OUTSIDE_VARIANCE)
        sample = dist.rsample(sample_shape)
        logp = dist.log_prob(sample.detach())
        return sample, logp

    def _temp(self, action, temp_outside, temp_hall):  # pylint:disable=too-many-locals
        air = action

        TIME_DELTA = torch.as_tensor(self._config["TIME_DELTA"])

        CAP = torch.as_tensor(self._config["CAP"])
        CAP_AIR = torch.as_tensor(self._config["CAP_AIR"])
        TEMP_AIR = torch.as_tensor(self._config["TEMP_AIR"])

        IS_ROOM = torch.as_tensor(self._config["IS_ROOM"])

        ADJ = torch.as_tensor(self._config["ADJ"])
        ADJ_OUTSIDE = torch.as_tensor(self._config["ADJ_OUTSIDE"])
        ADJ_HALL = torch.as_tensor(self._config["ADJ_HALL"])

        R_OUTSIDE = torch.as_tensor(self._config["R_OUTSIDE"])
        R_HALL = torch.as_tensor(self._config["R_HALL"])
        R_WALL = torch.as_tensor(self._config["R_WALL"])

        temp = self.temp
        temp_ = temp + TIME_DELTA / CAP * (
            air * CAP_AIR * (TEMP_AIR - temp) * IS_ROOM
            + ((ADJ | ADJ.T) * (temp[np.newaxis] - temp[np.newaxis].T) / R_WALL).sum(
                dim=-1
            )
            + ADJ_OUTSIDE * (temp_outside - temp) / R_OUTSIDE
            + ADJ_HALL * (temp_hall - temp) / R_HALL
        )

        return temp_

    def _step_time(self, time):
        timestep = torch.round(self._horizon * time)
        return torch.clamp((timestep + 1) / self._horizon, 0, 1)

    def reward_fn(self, state, action, next_state):
        # pylint:disable=unused-argument,missing-docstring
        AIR_MAX = torch.as_tensor(self._config["AIR_MAX"])
        air = torch.as_tensor(action) * AIR_MAX

        temp, _ = self._unpack_state(next_state)

        IS_ROOM = torch.as_tensor(self._config["IS_ROOM"])
        COST_AIR = torch.as_tensor(self._config["COST_AIR"])
        TEMP_LOW = torch.as_tensor(self._config["TEMP_LOW"])
        TEMP_UP = torch.as_tensor(self._config["TEMP_UP"])
        PENALTY = torch.as_tensor(self._config["PENALTY"])

        reward = -(
            IS_ROOM
            * (
                air * COST_AIR
                + ((temp < TEMP_LOW) | (temp > TEMP_UP)) * PENALTY
                + 10.0 * torch.abs((TEMP_UP + TEMP_LOW) / 2.0 - temp)
            )
        ).sum(dim=-1)

        return reward

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
