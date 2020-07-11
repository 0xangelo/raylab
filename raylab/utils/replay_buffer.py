"""Custom Replay Buffers based on RLlibs's implementation."""
import random
import sys
from dataclasses import dataclass

import numpy as np
from gym.spaces import Space
from ray.rllib import SampleBatch
from ray.rllib.optimizers.replay_buffer import ReplayBuffer as _ReplayBuffer
from ray.rllib.utils import override
from ray.rllib.utils.compression import unpack_if_needed


@dataclass
class ReplayField:
    """Storage specification for ReplayBuffer data."""

    name: str
    shape: tuple = ()
    dtype: np.dtype = np.float32


class ListReplayBuffer(_ReplayBuffer):
    """Replay buffer as a list of tuples.

    Returns a SampleBatch object when queried for samples.

    Args:
        size: max number of transitions to store in the buffer. When the buffer
            overflows, the old memories are dropped.

    Attributes:
        fields (:obj:`tuple` of :obj:`ReplayField`): storage fields
            specification
    """

    def __init__(self, size: int):
        super().__init__(size)
        self.fields = (
            ReplayField(SampleBatch.CUR_OBS),
            ReplayField(SampleBatch.ACTIONS),
            ReplayField(SampleBatch.REWARDS),
            ReplayField(SampleBatch.NEXT_OBS),
            ReplayField(SampleBatch.DONES),
        )

    def add_fields(self, *fields: ReplayField):
        """Add fields to the replay buffer and build the corresponding storage."""
        new_names = {f.name for f in fields}
        assert len(new_names) == len(fields), "Field names must be unique"

        conflicts = new_names.intersection({f.name for f in self.fields})
        assert not conflicts, f"{conflicts} are already in buffer"

        self.fields = self.fields + fields

    @override(_ReplayBuffer)
    def add(self, row: dict):  # pylint:disable=arguments-differ
        """Add a row from a SampleBatch to storage.

        Args:
            row: sample batch row as returned by SampleBatch.rows
        """
        data = tuple(row[f.name] for f in self.fields)
        self._num_added += 1

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            self._est_size_bytes += sum(sys.getsizeof(d) for d in data)
        else:
            self._storage[self._next_idx] = data
        if self._next_idx + 1 >= self._maxsize:
            self._eviction_started = True
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if self._eviction_started:
            self._evicted_hit_stats.push(self._hit_count[self._next_idx])
            self._hit_count[self._next_idx] = 0

    @override(_ReplayBuffer)
    def _encode_sample(self, idxes):
        sample = []
        for i in idxes:
            sample.append(self._storage[i])
            self._hit_count[i] += 1

        obses_t, actions, rewards, obses_tp1, dones, *extras = zip(*sample)

        obses_t = [np.array(unpack_if_needed(o), copy=False) for o in obses_t]
        actions = [np.array(a, copy=False) for a in actions]
        obses_tp1 = [np.array(unpack_if_needed(o), copy=False) for o in obses_tp1]

        return tuple(
            map(np.array, [obses_t, actions, rewards, obses_tp1, dones] + extras)
        )

    @override(_ReplayBuffer)
    def sample(self, batch_size: int) -> SampleBatch:
        idxes = random.choices(range(len(self._storage)), k=batch_size)
        return self.sample_with_idxes(idxes)

    @override(_ReplayBuffer)
    def sample_with_idxes(self, idxes: np.ndarray) -> SampleBatch:
        self._num_sampled += len(idxes)
        data = self._encode_sample(idxes)
        return SampleBatch(dict(zip([f.name for f in self.fields], data)))

    def all_samples(self) -> SampleBatch:
        """All transitions stored in buffer."""
        return self.sample_with_idxes(range(len(self)))


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
    """

    def __init__(self, obs_space: Space, action_space: Space, size: int):
        # pylint:disable=too-many-arguments
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

    def __len__(self) -> int:
        return self._curr_size

    def _build_buffers(self, *fields: ReplayField):
        storage = self._storage
        size = self._maxsize
        for field in fields:
            storage[field.name] = np.empty((size,) + field.shape, dtype=field.dtype)

    def seed(self, seed: int = None):
        """Seed the random number generator for sampling minibatches."""
        self._rng = np.random.default_rng(seed)

    def add_fields(self, *fields: ReplayField):
        """Add fields to the replay buffer and build the corresponding storage."""
        new_names = {f.name for f in fields}
        assert len(new_names) == len(fields), "Field names must be unique"

        conflicts = new_names.intersection({f.name for f in self.fields})
        assert not conflicts, f"{conflicts} are already in buffer"

        self.fields = self.fields + fields
        self._build_buffers(*fields)

    def add(self, row: dict):  # pylint:disable=arguments-differ
        """Add a row from a SampleBatch to storage.

        Args:
            row: sample batch row as returned by SampleBatch.rows().
                Must have the same keys as the field names in the buffer.
        """
        for field in self.fields:
            self._storage[field.name][self._next_idx] = row[field.name]

        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._curr_size += 1 if self._curr_size < self._maxsize else 0

    def sample(self, batch_size: int) -> SampleBatch:
        """Transition batch uniformly sampled with replacement."""
        return self.sample_with_idxes(self.sample_idxes(batch_size))

    def sample_idxes(self, batch_size: int) -> np.ndarray:
        """Get random transition indexes uniformly sampled with replacement."""
        return self._rng.integers(self._curr_size, size=batch_size)

    def sample_with_idxes(self, idxes: np.ndarray) -> SampleBatch:
        """Transition batch corresponding with the given indexes."""
        batch = {k: self._storage[k][idxes] for k in (f.name for f in self.fields)}
        return SampleBatch(batch)

    def all_samples(self) -> SampleBatch:
        """All stored transitions."""
        return SampleBatch(
            {k: self._storage[k][: len(self)] for k in (f.name for f in self.fields)}
        )
