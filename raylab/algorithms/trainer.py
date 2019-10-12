"""Primitives for all Trainers."""
from ray.rllib.agents.trainer import Trainer as _Trainer


class Trainer(_Trainer):
    """Base Trainer for all algorithms. This should not be instantiated."""

    # pylint: disable=abstract-method,no-member
    _allow_unknown_subkeys = _Trainer._allow_unknown_subkeys + ["module"]

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
        return res
