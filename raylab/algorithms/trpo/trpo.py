"""Trainer and configuration for TRPO."""
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.utils.annotations import override

from raylab.algorithms import Trainer, with_common_config
from .trpo_policy import TRPOTorchPolicy

DEFAULT_CONFIG = with_common_config(
    {
        # # Fraction of samples to use for KL computation
        # "kl_frac": 1.0,
        # Trust region constraint
        "delta": 0.01,
        # Number of iterations to fit value function
        "val_iters": 80,
        # Whether to use Generalized Advantage Estimation
        "use_gae": True,
        # Whether to use a line search to calculate policy update.
        # Effectively turns TRPO into Natural PG when turned off.
        "line_search": True,
    }
)


class TRPOTrainer(Trainer):
    """Single agent trainer for TRPO."""

    _name = "TRPO"
    _default_config = DEFAULT_CONFIG
    _policy = TRPOTorchPolicy

    # pylint:disable=attribute-defined-outside-init

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=config["num_workers"]
        )
        self.optimizer = SyncSamplesOptimizer(
            self.workers, train_batch_size=config["train_batch_size"]
        )

    @override(Trainer)
    def _train(self):
        while not self._iteration_done():
            fetches = self.optimizer.step()

        return self._log_metrics(fetches)
