"""Global registry of rewards, terminations, and raylab modules.

Mimics Tune's registry in `ray.tune.registry`. Such as registry is needed so
that users can create custom reward and termination functions and use them in
a distributed setting.
"""
import ray
import ray.cloudpickle as pickle
from ray.experimental.internal_kv import _internal_kv_get
from ray.experimental.internal_kv import _internal_kv_initialized
from ray.experimental.internal_kv import _internal_kv_put
from ray.tune.registry import _make_key

RAYLAB_REWARD = "raylab_reward"
RAYLAB_TERMINATION = "raylab_termination"
RAYLAB_MODULE = "raylab_module"
KNOWN_CATEGORIES = {
    RAYLAB_REWARD,
    RAYLAB_TERMINATION,
    RAYLAB_MODULE,
}


class _Registry:
    # pylint:disable=missing-docstring
    def __init__(self):
        self._to_flush = {}

    def register(self, category, key, value):
        if category not in KNOWN_CATEGORIES:
            from ray.tune import TuneError

            raise TuneError(
                "Unknown category {} not among {}".format(category, KNOWN_CATEGORIES)
            )
        self._to_flush[(category, key)] = pickle.dumps(value)
        if _internal_kv_initialized():
            self.flush_values()

    def contains(self, category, key):
        if _internal_kv_initialized():
            value = _internal_kv_get(_make_key(category, key))
            return value is not None

        return (category, key) in self._to_flush

    def get(self, category, key):
        if _internal_kv_initialized():
            value = _internal_kv_get(_make_key(category, key))
            if value is None:
                raise ValueError(
                    "Registry value for {}/{} doesn't exist.".format(category, key)
                )
            return pickle.loads(value)

        return pickle.loads(self._to_flush[(category, key)])

    def flush_values(self):
        for (category, key), value in self._to_flush.items():
            _internal_kv_put(_make_key(category, key), value, overwrite=True)
        self._to_flush.clear()


_raylab_registry = _Registry()
# pylint:disable=protected-access
ray.worker._post_init_hooks.append(_raylab_registry.flush_values)
