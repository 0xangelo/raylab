"""Primitives for all Trainers."""
from ray.rllib.agents.trainer import Trainer as _Trainer
from ray.rllib.utils.annotations import override

_Trainer._allow_unknown_subkeys += ["module", "torch_optimizer"]
_Trainer._override_all_subkeys_if_type_changes += ["module"]


class Trainer(_Trainer):
    """Base Trainer for all algorithms. This should not be instantiated."""

    # pylint: disable=abstract-method,no-member

    @override(_Trainer)
    def train(self):
        # Evaluate first, before any optimization is done
        if self.config.get("evaluation_interval"):
            # pylint:disable=attribute-defined-outside-init
            self.evaluation_metrics = self._evaluate()

        result = super().train()

        # Update global_vars after training so that the info is saved if checkpointing
        if self._has_policy_optimizer():
            self.global_vars["timestep"] = self.optimizer.num_steps_sampled
        return result

    @override(_Trainer)
    def __getstate__(self):
        state = super().__getstate__()
        state["global_vars"] = self.global_vars
        return state

    @override(_Trainer)
    def __setstate__(self, state):
        self.global_vars = state["global_vars"]
        super().__setstate__(state)
        if self._has_policy_optimizer():
            self.optimizer.workers.local_worker().set_global_vars(self.global_vars)
            for worker in self.optimizer.workers.remote_workers():
                worker.set_global_vars.remote(self.global_vars)

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
