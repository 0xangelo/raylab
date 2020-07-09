"""Trainer and configuration for TRPO."""
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.utils import override

from raylab.agents import Trainer
from raylab.agents import trainer

from .policy import TRPOTorchPolicy

LINESEARCH_OPTIONS = {
    "accept_ratio": 0.1,
    "backtrack_ratio": 0.8,
    "max_backtracks": 15,
    "atol": 1e-7,
}


@trainer.config("delta", 0.01, info="Trust region constraint")
@trainer.config(
    "fvp_samples",
    10,
    info="Number of actions to sample per state for Fisher vector-product calculation",
)
@trainer.config("lambda", 0.97, info=r"For GAE(\gamma, \lambda)")
@trainer.config("val_iters", 80, info="Number of iterations to fit value function")
@trainer.config(
    "torch_optimizer",
    {"type": "Adam", "lr": 1e-3},
    override=True,
    info="Options for the critic's optimizer",
)
@trainer.config("use_gae", True, info="Whether to use Generalized Advantage Estimation")
@trainer.config("cg_iters", 10, info="Number of iterations for Conjugate Gradient")
@trainer.config(
    "cg_damping",
    1e-3,
    info="Damping factor to avoid singular matrix multiplication in Conjugate Gradient",
)
@trainer.config(
    "line_search",
    True,
    info="""\
    Whether to use a line search to calculate policy update.
    Effectively turns TRPO into Natural PG when turned off.
    """,
)
@trainer.config("line_search_options", LINESEARCH_OPTIONS)
@trainer.config("module/type", "OnPolicyActorCritic-v0")
@trainer.config(
    "exploration_config/type",
    "raylab.utils.exploration.StochasticActor",
    override=True,
)
@Trainer.with_base_specs
class TRPOTrainer(Trainer):
    """Single agent trainer for TRPO."""

    _name = "TRPO"
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
        init_timesteps = self.optimizer.num_steps_sampled
        timesteps_per_iteration = max(self.config["timesteps_per_iteration"], 2)

        while (
            self.optimizer.num_steps_sampled - init_timesteps < timesteps_per_iteration
        ):
            _ = self.optimizer.step()

        res = self.collect_metrics()
        timesteps = self.optimizer.num_steps_sampled - init_timesteps
        res.update(timesteps_this_iter=timesteps, info=res.get("info", {}))
        return res
