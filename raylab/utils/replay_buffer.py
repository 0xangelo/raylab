"""Custom Replay Buffers based on RLlibs's implementation."""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from gym.spaces import Space
from ray.rllib import SampleBatch


@dataclass
class ReplayField:
    """Storage specification for ReplayBuffer data."""

    name: str
    shape: tuple = ()
    dtype: np.dtype = np.float32


class NumpyReplayBuffer:
    """Replay buffer as a dict of ndarrays.

    Returns a SampleBatch object when queried for samples.

    Args:
        obs_space: observation space
        action_space: action space
        size: max number of transitions to store in the buffer.
            When the bufferoverflows the old memories are dropped.

    Attributes:
        fields (:obj:`tuple` of :obj:`ReplayField`): storage fields
            specification
        compute_stats: Whether to track mean and stddev for normalizing
            observations
    """

    # pylint:disable=too-many-instance-attributes
    compute_stats: bool = False

    def __init__(self, obs_space: Space, action_space: Space, size: int):
        self._maxsize = size
        self.fields = (
            ReplayField(
                SampleBatch.CUR_OBS, shape=obs_space.shape, dtype=obs_space.dtype
            ),
            ReplayField(
                SampleBatch.ACTIONS, shape=action_space.shape, dtype=action_space.dtype
            ),
            ReplayField(SampleBatch.REWARDS, shape=(), dtype=np.float32),
            ReplayField(
                SampleBatch.NEXT_OBS, shape=obs_space.shape, dtype=obs_space.dtype
            ),
            ReplayField(SampleBatch.DONES, shape=(), dtype=np.bool),
        )
        self._storage = {}
        self._build_buffers(*self.fields)
        self._next_idx = 0
        self._curr_size = 0
        self._rng = np.random.default_rng()
        self._obs_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __len__(self) -> int:
        return self._curr_size

    def add_fields(self, *fields: ReplayField):
        """Add fields to the replay buffer and build the corresponding storage."""
        new_names = {f.name for f in fields}
        assert len(new_names) == len(fields), "Field names must be unique"

        conflicts = new_names.intersection({f.name for f in self.fields})
        assert not conflicts, f"{conflicts} are already in buffer"

        self.fields = self.fields + fields
        self._build_buffers(*fields)

    def _build_buffers(self, *fields: ReplayField):
        storage = self._storage
        size = self._maxsize
        for field in fields:
            storage[field.name] = np.empty((size,) + field.shape, dtype=field.dtype)

    def __getitem__(
        self, index: Union[int, np.ndarray, slice]
    ) -> Dict[str, np.ndarray]:
        batch = {f.name: self._storage[f.name][index] for f in self.fields}
        for key in SampleBatch.CUR_OBS, SampleBatch.NEXT_OBS:
            batch[key] = self.normalize(batch[key])
        return batch

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using the stored mean and stddev."""
        obs = np.asarray(obs)
        if not self.compute_stats:
            return obs

        if not self._obs_stats:
            self.update_obs_stats()

        mean, std = self._obs_stats
        return (obs - mean) / std

    def update_obs_stats(self):
        """Compute mean and standard deviation for observation normalization.

        Subsequent batches sampled from this buffer will use these statistics to
        normalize the current and next observation fields.
        """
        if len(self) == 0:
            self._obs_stats = (0, 1)
        else:
            cur_obs = self._storage[SampleBatch.CUR_OBS][: len(self)]
            mean = np.mean(cur_obs, axis=0)
            std = np.std(cur_obs, axis=0)
            std[std < 1e-12] = 1.0
            self._obs_stats = (mean, std)

    def seed(self, seed: int = None):
        """Seed the random number generator for sampling minibatches."""
        self._rng = np.random.default_rng(seed)

    def add(self, samples: SampleBatch):
        """Add a SampleBatch to storage.

        Optimized to avoid several queries for large sample batches.

        Args:
            samples: The sample batch
        """
        if samples.count >= self._maxsize:
            samples = samples.slice(samples.count - self._maxsize, None)
            end_idx = 0
            assign = [(slice(0, self._maxsize), samples)]
        else:
            start_idx = self._next_idx
            end_idx = (self._next_idx + samples.count) % self._maxsize
            if end_idx < start_idx:
                tailcount = self._maxsize - start_idx
                assign = [
                    (slice(start_idx, None), samples.slice(0, tailcount)),
                    (slice(end_idx), samples.slice(tailcount, None)),
                ]
            else:
                assign = [(slice(start_idx, end_idx), samples)]

        for field in self.fields:
            for slc, smp in assign:
                self._storage[field.name][slc] = smp[field.name]

        self._next_idx = end_idx
        self._curr_size = min(self._curr_size + samples.count, self._maxsize)
        self._obs_stats = None

    def sample(self, batch_size: int) -> SampleBatch:
        """Transition batch uniformly sampled with replacement."""
        return SampleBatch(self[self.sample_idxes(batch_size)])

    def sample_idxes(self, batch_size: int) -> np.ndarray:
        """Get random transition indexes uniformly sampled with replacement."""
        return self._rng.integers(self._curr_size, size=batch_size)

    def all_samples(self) -> SampleBatch:
        """All stored transitions."""
        return SampleBatch(self[: len(self)])

    def state_dict(self) -> dict:
        return {"obs_stats": self._obs_stats}

    def load_state_dict(self, state: dict):
        self._obs_stats = state["obs_stats"]
