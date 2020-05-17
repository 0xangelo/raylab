"""Custom Replay Buffer subclassing RLlibs's implementation."""
import sys
import random

import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib import SampleBatch
from ray.rllib.optimizers.replay_buffer import ReplayBuffer as _ReplayBuffer
from ray.rllib.utils.compression import unpack_if_needed


class ReplayBuffer(_ReplayBuffer):
    """Replay buffer that returns a SampleBatch object when queried for samples.

    Arguments:
        size (int): max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        extra_keys (tuple): extra keys to store from sample batches, optional.
    """

    def __init__(self, size, extra_keys=()):
        super().__init__(size)
        self._batch_keys = (
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.REWARDS,
            SampleBatch.NEXT_OBS,
            SampleBatch.DONES,
        ) + tuple(extra_keys)

    @override(_ReplayBuffer)
    def add(self, row):  # pylint: disable=arguments-differ
        """Add a row from a SampleBatch to storage.

        Arguments:
            row (dict): sample batch row as returned by SampleBatch.rows
        """
        data = tuple(row[key] for key in self._batch_keys)
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
    def sample(self, batch_size):
        idxes = random.choices(range(len(self._storage)), k=batch_size)
        data = self._encode_sample(idxes)
        return SampleBatch(dict(zip(self._batch_keys, data)))
