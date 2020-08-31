"""Trainer and configuration for TRPO."""
from ray.rllib.agents.trainer_template import default_execution_plan
from ray.rllib.utils import override
from ray.tune import Trainable

from raylab.agents import Trainer
from raylab.agents import trainer

from .policy import TRPOTorchPolicy

LINESEARCH_OPTIONS = {
    "accept_ratio": 0.1,
    "backtrack_ratio": 0.8,
    "max_backtracks": 15,
    "atol": 1e-7,
}


@trainer.configure
@trainer.option("delta", 0.01, help="Trust region constraint")
@trainer.option(
    "fvp_samples",
    10,
    help="Number of actions to sample per state for Fisher vector-product calculation",
)
@trainer.option("lambda", 0.97, help=r"For GAE(\gamma, \lambda)")
@trainer.option("val_iters", 80, help="Number of iterations to fit value function")
@trainer.option(
    "torch_optimizer",
    {"type": "Adam", "lr": 1e-3},
    override=True,
    help="Options for the critic's optimizer",
)
@trainer.option("use_gae", True, help="Whether to use Generalized Advantage Estimation")
@trainer.option("cg_iters", 10, help="Number of iterations for Conjugate Gradient")
@trainer.option(
    "cg_damping",
    1e-3,
    help="Damping factor to avoid singular matrix multiplication in Conjugate Gradient",
)
@trainer.option(
    "line_search",
    True,
    help="""
    Whether to use a line search to calculate policy update.
    Effectively turns TRPO into Natural PG when turned off.
    """,
)
@trainer.option("line_search_options", LINESEARCH_OPTIONS)
@trainer.option("module/type", "OnPolicyActorCritic-v0")
@trainer.option(
    "exploration_config/type",
    "raylab.utils.exploration.StochasticActor",
    override=True,
)
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

        self.execution_plan = default_execution_plan
        self.train_exec_impl = self.execution_plan(self.workers, config)

    @override(Trainable)
    def step(self):
        return next(self.train_exec_impl)
