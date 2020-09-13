"""Trainer and configuration for TRPO."""
from ray.rllib.utils import override

from raylab.agents.simple_trainer import SimpleTrainer
from raylab.options import configure
from raylab.options import option

from .policy import TRPOTorchPolicy

LINESEARCH_OPTIONS = {
    "accept_ratio": 0.1,
    "backtrack_ratio": 0.8,
    "max_backtracks": 15,
    "atol": 1e-7,
}


@configure
@option("delta", 0.01, help="Trust region constraint")
@option(
    "fvp_samples",
    10,
    help="Number of actions to sample per state for Fisher vector-product calculation",
)
@option("lambda", 0.97, help=r"For GAE(\gamma, \lambda)")
@option("val_iters", 80, help="Number of iterations to fit value function")
@option(
    "torch_optimizer",
    {"type": "Adam", "lr": 1e-3},
    override=True,
    help="Options for the critic's optimizer",
)
@option("use_gae", True, help="Whether to use Generalized Advantage Estimation")
@option("cg_iters", 10, help="Number of iterations for Conjugate Gradient")
@option(
    "cg_damping",
    1e-3,
    help="Damping factor to avoid singular matrix multiplication in Conjugate Gradient",
)
@option(
    "line_search",
    True,
    help="""
    Whether to use a line search to calculate policy update.
    Effectively turns TRPO into Natural PG when turned off.
    """,
)
@option("line_search_options", LINESEARCH_OPTIONS)
@option("module/type", "TRPO")
@option(
    "exploration_config/type",
    "raylab.utils.exploration.StochasticActor",
    override=True,
)
class TRPOTrainer(SimpleTrainer):
    """Single agent trainer for TRPO."""

    # pylint:disable=abstract-method
    _name = "TRPO"

    @staticmethod
    @override(SimpleTrainer)
    def validate_config(config: dict):
        assert not config[
            "learning_starts"
        ], "No point in having a warmup for an on-policy algorithm."

    @override(SimpleTrainer)
    def get_policy_class(self, _):
        return TRPOTorchPolicy
