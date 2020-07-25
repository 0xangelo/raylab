# pylint:disable=missing-module-docstring
import gym
import numpy as np

from .utils import assert_flat_box_space
from .utils import ignore_rightmost_variables


class AddRelativeTimestep(gym.ObservationWrapper):
    """
    Adds the relative timestep (normalized to the range [0, 1]) to the
    observations of environments with observation spaces of type gym.spaces.Box.
    """

    def __init__(self, env=None):
        assert_flat_box_space(env.observation_space, self)
        super().__init__(env)

        _env = env
        while hasattr(_env, "env"):
            if isinstance(_env, gym.wrappers.TimeLimit):
                break
            _env = _env.env
        self._env = _env

        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, 0.0),
            high=np.append(self.observation_space.high, 1.0),
            dtype=self.observation_space.dtype,
        )

        if hasattr(self.env, "reward_fn"):
            self.reward_fn = ignore_rightmost_variables(self.env.reward_fn, 1)

        if hasattr(self.env, "termination_fn"):

            def termination_fn(state, action, next_state):
                env_done = self.env.termination_fn(
                    state[..., :-1], action, next_state[..., :-1]
                )
                timeout = next_state[..., -1] >= 1.0
                return timeout | env_done

            self.termination_fn = termination_fn

    def observation(self, observation):
        # pylint:disable=protected-access
        relative_time = self._env._elapsed_steps / self._env._max_episode_steps
        return np.append(observation, relative_time)
