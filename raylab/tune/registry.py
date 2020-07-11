"""Global registry of rewards, terminations, and raylab modules.

Mimics Tune's registry in `ray.tune.registry`. Such as registry is needed so
that users can create custom reward and termination functions and use them in
a distributed setting.
"""
import ray
import ray.cloudpickle as pickle
from ray.experimental.internal_kv import _internal_kv_initialized
from ray.tune.registry import _Registry as _TuneRegistry

RAYLAB_REWARD = "raylab_reward"
RAYLAB_TERMINATION = "raylab_termination"
RAYLAB_MODULE = "raylab_module"
KNOWN_CATEGORIES = {
    RAYLAB_REWARD,
    RAYLAB_TERMINATION,
    RAYLAB_MODULE,
}


class _RaylabRegistry(_TuneRegistry):
    # pylint:disable=missing-docstring
    def register(self, category, key, value):
        if category not in KNOWN_CATEGORIES:
            from ray.tune import TuneError

            raise TuneError(
                "Unknown category {} not among {}".format(category, KNOWN_CATEGORIES)
            )
        self._to_flush[(category, key)] = pickle.dumps(value)
        if _internal_kv_initialized():
            self.flush_values()


_raylab_registry = _RaylabRegistry()
# pylint:disable=protected-access
ray.worker._post_init_hooks.append(_raylab_registry.flush_values)
