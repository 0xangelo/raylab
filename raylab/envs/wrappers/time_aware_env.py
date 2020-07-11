# pylint:disable=missing-docstring
# pylint: enable=missing-docstring
import gym
import numpy as np


class AddRelativeTimestep(gym.ObservationWrapper):
    """
    Adds the relative timestep (normalized to the range [0, 1]) to the
    observations of environments with observation spaces of type gym.spaces.Box.
    """

    def __init__(self, env=None):
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

            def reward_fn(state, action, next_state):
                return self.env.reward_fn(state[..., :-1], action, next_state[..., :-1])

            self.reward_fn = reward_fn

    def observation(self, observation):
        # pylint:disable=protected-access
        return np.append(
            observation, (self._env._elapsed_steps / self._env._max_episode_steps)
        )
