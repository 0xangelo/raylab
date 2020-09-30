"""Custom Replay Buffers based on RLlibs's implementation."""
from dataclasses import dataclass
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from gym.spaces import Space
from ray.rllib import SampleBatch


@dataclass
class ReplayField:
    """Storage specification for ReplayBuffer data."""

    name: str
    shape: tuple = ()
    dtype: np.dtype = np.float32


class Storage:
    """Replay storage as a dict of ndarrays.

    Args:
        size: Maximum length of arrays to store.

    Attributes:
        maxsize: Maximum length of stored arrays
        next_idx: Next starting index for new arrays
        fields (:obj:`tuple` of :obj:`ReplayField`): storage fields
            specification
    """

    def __init__(self, size: int):
        self.maxsize = size
        self.fields = ()
        self.buffer = {}
        self.next_idx = 0
        self.curr_size = 0

    def __len__(self) -> int:
        return self.curr_size

    def __getitem__(
        self, index: Union[int, np.ndarray, slice]
    ) -> Dict[str, np.ndarray]:
        return {f.name: self.buffer[f.name][index] for f in self.fields}

    def column(self, name: str) -> np.ndarray:
        """Returns a full array of one of the buffers."""
        return self.buffer[name][: len(self)]

    def add_fields(self, *fields: ReplayField):
        """Add fields and build the corresponding buffers."""
        new_names = {f.name for f in fields}
        assert len(new_names) == len(fields), "Field names must be unique"

        conflicts = new_names.intersection({f.name for f in self.fields})
        assert not conflicts, f"{conflicts} are already in buffer"

        self.fields = self.fields + fields
        self._build_buffers(*fields)

    def _build_buffers(self, *fields: ReplayField):
        size = self.maxsize
        for field in fields:
            self.buffer[field.name] = np.empty((size,) + field.shape, dtype=field.dtype)

    def add(self, arrs: Mapping[str, np.ndarray]):
        """Add a collection of arrays to storage.

        Args:
            arrs: A mapping from strings to arrays. Must have all the keys
                specificied by this instance's field names.
        """
        keys = [f.name for f in self.fields]
        count = len(arrs[keys[0]])
        assert all(len(arrs[k]) == count for k in keys)

        if count >= self.maxsize:
            for key in keys:
                self.buffer[key] = arrs[key][-self.maxsize :]
            self.next_idx = 0
        else:
            start = self.next_idx
            end = (start + count) % self.maxsize
            if start <= end:
                for key in keys:
                    self.buffer[key][start:end] = arrs[key]
            else:
                tailcount = self.maxsize - start
                for key in keys:
                    self.buffer[key][start:] = arrs[key][:tailcount]
                    self.buffer[key][:end] = arrs[key][-tailcount:]

        self.curr_size = min(self.curr_size + count, self.maxsize)


class NumpyReplayBuffer:
    """Replay buffer as a collection of ndarrays.

    Returns a SampleBatch object when queried for samples.

    Args:
        obs_space: observation space
        action_space: action space
        size: max number of transitions to store in the buffer.
            When the bufferoverflows the old memories are dropped.

    Attributes:
        compute_stats: Whether to track mean and stddev for normalizing
            observations
    """

    compute_stats: bool = False

    def __init__(self, obs_space: Space, action_space: Space, size: int):
        self._storage = Storage(size)
        self.add_fields(
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
        self._rng = np.random.default_rng()
        self._obs_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __len__(self) -> int:
        return len(self._storage)

    def add_fields(self, *fields: ReplayField):
        """Add fields and build the corresponding storage."""
        self._storage.add_fields(*fields)

    @property
    def fields(self):
        """All registered replay fields in this buffer."""
        return self._storage.fields

    def __getitem__(
        self, index: Union[int, np.ndarray, slice]
    ) -> Dict[str, np.ndarray]:
        batch = self._storage[index]
        for key in SampleBatch.CUR_OBS, SampleBatch.NEXT_OBS:
            batch[key] = self.normalize(batch[key])
        return batch

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using the stored mean and stddev."""
        if not self.compute_stats:
            return obs

        if not self._obs_stats:
            self.update_obs_stats()

        mean, std = self._obs_stats
        return (obs - mean) / (std + 1e-6)

    def update_obs_stats(self):
        """Compute mean and standard deviation for observation normalization.

        Subsequent batches sampled from this buffer will use these statistics to
        normalize the current and next observation fields.
        """
        if len(self) == 0:
            self._obs_stats = (0, 1)
        else:
            cur_obs = self._storage.column(SampleBatch.CUR_OBS)
            mean = np.mean(cur_obs, axis=0)
            std = np.std(cur_obs, axis=0)
            std = np.where(std < 1e-12, np.ones_like(std), std)
            self._obs_stats = (mean, std)

    def seed(self, seed: int = None):
        """Seed the random number generator for sampling minibatches."""
        self._rng = np.random.default_rng(seed)

    def add(self, samples: SampleBatch):
        """Add a SampleBatch to storage.

        Args:
            samples: The sample batch
        """
        self._storage.add(samples)
        self._obs_stats = None

    def sample(self, batch_size: int) -> SampleBatch:
        """Transition batch uniformly sampled with replacement."""
        return SampleBatch(self[self.sample_idxes(batch_size)])

    def sample_idxes(self, batch_size: int) -> np.ndarray:
        """Get random transition indexes uniformly sampled with replacement."""
        return self._rng.integers(len(self), size=batch_size)

    def all_samples(self) -> SampleBatch:
        """All stored transitions."""
        return SampleBatch(self[: len(self)])
