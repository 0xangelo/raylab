"""Primitives for all Trainers."""
from ray.rllib.agents.trainer import Trainer as _Trainer
from ray.rllib.utils.annotations import override
from ray.tune.registry import _global_registry, ENV_CREATOR, register_env

from raylab.envs.utils import wrap_if_needed


class Trainer(_Trainer):
    """Base Trainer for all algorithms. This should not be instantiated."""

    # pylint: disable=abstract-method,no-member
    _allow_unknown_subkeys = _Trainer._allow_unknown_subkeys + ["module"]

    @override(_Trainer)
    def _register_if_needed(self, env_object):
        if isinstance(env_object, str) and not _global_registry.contains(
            ENV_CREATOR, env_object
        ):
            import gym

            register_env(env_object, wrap_if_needed(lambda _: gym.make(env_object)))
        return super()._register_if_needed(env_object)

    def _iteration_done(self):
        return self.optimizer.num_steps_sampled - self.global_vars["timestep"] >= max(
            self.config["timesteps_per_iteration"], 1
        )

    def _log_metrics(self, learner_stats):
        res = self.collect_metrics()
        timesteps = self.optimizer.num_steps_sampled - self.global_vars["timestep"]
        res.update(
            timesteps_this_iter=timesteps,
            info=dict(learner=learner_stats, **res.get("info", {})),
        )
        if self._iteration == 0 and self.config["evaluation_interval"]:
            res.update(self.evaluation_metrics)
        return res
