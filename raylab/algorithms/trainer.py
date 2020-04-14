"""Primitives for all Trainers."""
from ray.rllib.agents.trainer import Trainer as _Trainer
from ray.rllib.utils.annotations import override

_Trainer._allow_unknown_subkeys += ["module"]
_Trainer._override_all_subkeys_if_type_changes += ["module"]


class Trainer(_Trainer):
    """Base Trainer for all algorithms. This should not be instantiated."""

    # pylint: disable=abstract-method,no-member

    @override(_Trainer)
    def _setup(self, config):
        super()._setup(config)
        # Evaluate first, before any optimization is done
        if self.config.get("evaluation_interval"):
            self.evaluation_metrics = self._evaluate()

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
