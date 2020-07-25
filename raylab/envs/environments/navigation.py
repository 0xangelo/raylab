# pylint:disable=missing-module-docstring
import gym
import numpy as np
import torch
from torch import Tensor


DEFAULT_CONFIG = {
    "start": [0.0, 1.0],
    "end": [8.0, 9.0],
    "action_lower_bound": [-1.0, -1.0],
    "action_upper_bound": [1.0, 1.0],
    "deceleration_zones": {"center": [[0.0, 0.0]], "decay": [2.0]},
    "noise": {"loc": [0.0, 0.0], "scale_tril": [[0.3, 0.0], [0.0, 0.3]]},
    "horizon": 20,
    "init_dist": True,
}


class NavigationEnv(gym.Env):
    """NavigationEnv implements a gym environment for the Navigation
    domain.

    The agent must navigate from a start position to and end position.
    Its actions represent displacements in the 2D plane. Gaussian noise
    is added to the final position as to incorporate uncertainty in the
    transition. Additionally, the effect of an action might be decreased
    by a scalar factor dependent on the proximity of deceleration zones.

    Please refer to the AAAI paper for further details:

    Bueno, T.P., de Barros, L.N., Mauá, D.D. and Sanner, S., 2019, July.
    Deep Reactive Policies for Planning in Stochastic Nonlinear Domains.
    In Proceedings of the AAAI Conference on Artificial Intelligence.
    """

    # pylint:disable=too-many-instance-attributes

    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        self._config = {**DEFAULT_CONFIG, **(config or {})}

        self._start = np.array(self._config["start"], dtype=np.float32)
        self._end = np.array(self._config["end"], dtype=np.float32)

        self._min_x = self._start[0] - 1.0
        self._max_x = self._end[0] + 1.0
        self._min_y = self._start[1] - 1.0
        self._max_y = self._end[1] + 1.0

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0], dtype=np.float32),
        )
        self.action_space = gym.spaces.Box(
            low=np.array(self._config["action_lower_bound"], dtype=np.float32),
            high=np.array(self._config["action_upper_bound"], dtype=np.float32),
        )

        self._deceleration_zones = self._config["deceleration_zones"]
        if self._deceleration_zones:
            self._deceleration_decay = np.array(
                self._deceleration_zones["decay"], dtype=np.float32
            )
            self._deceleration_center = np.array(
                self._deceleration_zones["center"], dtype=np.float32
            )

        self._noise = self._config["noise"]
        self._horizon = self._config["horizon"]
        self._state = None

    def reset(self):
        if self._config["init_dist"]:
            xpos = np.random.uniform(self._min_x, self._max_x)
            ypos = np.random.uniform(self._min_y, self._max_y)
            time = 0.0
            self._state = np.array([xpos, ypos, time], dtype=np.float32)
        else:
            self._state = np.append(self._start, np.array(0.0, dtype=np.float32))

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
        position, time = self._unpack_state(state)
        deceleration = 1.0
        if self._deceleration_zones:
            deceleration = self._deceleration(position)

        position = position + (deceleration * action)
        position, logp = self._sample_noise(position, sample_shape)
        time = self._step_time(time)
        time = time.expand_as(position[..., -1:])
        return torch.cat([position, time], dim=-1), logp

    def _deceleration(self, position: Tensor) -> Tensor:
        # Decay is a tensor of shape (n)
        decay = torch.from_numpy(self._deceleration_decay)
        # Center is a matrix with rows as center positions (n, d)
        center = torch.from_numpy(self._deceleration_center)
        # Consider position as a batched row vector (*, d)
        # Unsqueeze position to form a (*, 1, d) matrix
        position = position.unsqueeze(-2)
        # Broadcasting will reshape position and center to (*, n, d)
        # Leverage broadcasting rules to compute batched distances
        # distance is a vector with distances to each center (*, n)
        distance = self.distance(position, center)
        # Compute decelerations for each zone (*, n)
        unreduced_decel = 2 / (1.0 + torch.exp(-decay * distance)) - 1.0
        # Reduce by computing the product of decelerations (*, 1)
        deceleration = torch.prod(unreduced_decel, dim=-1, keepdim=True)
        return deceleration

    def _sample_noise(self, position, sample_shape):
        loc = position + torch.as_tensor(self._noise["loc"])
        scale_tril = torch.as_tensor(self._noise["scale_tril"])
        dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        sample = dist.rsample(sample_shape=sample_shape)
        return sample, dist.log_prob(sample.detach())

    def _step_time(self, time):
        timestep = torch.round(self._horizon * time)
        return torch.clamp((timestep + 1) / self._horizon, 0, 1)

    def reward_fn(self, state: Tensor, action: Tensor, next_state: Tensor) -> Tensor:
        # pylint:disable=unused-argument,missing-docstring
        position, _ = self._unpack_state(next_state)
        goal = torch.from_numpy(self._end)
        return torch.norm(position - goal, dim=-1).neg()

    def _terminal(self):
        position, time = self._unpack_state(self._state)
        return np.allclose(position, self._end, atol=1e-1) or time.item() >= 1.0

    def termination_fn(
        self, state: Tensor, action: Tensor, next_state: Tensor
    ) -> Tensor:
        # pylint:disable=unused-argument,missing-docstring
        position, time = self._unpack_state(next_state)
        goal = torch.from_numpy(self._end)
        hit_goal = self.distance(position, goal) <= 1e-1
        timeout = time[..., 0] >= 1.0
        return hit_goal | timeout

    @staticmethod
    def distance(pos1: Tensor, pos2: Tensor) -> Tensor:
        """Batched distance computation between positions.

        Returns:
            Tensor with distances for each batch dimension
        """
        return torch.norm(pos1 - pos2, dim=-1)

    @staticmethod
    def _unpack_state(state):
        position, time = state[..., :-1], state[..., -1:]
        return position, time

    def render(self, mode="human"):
        pass
