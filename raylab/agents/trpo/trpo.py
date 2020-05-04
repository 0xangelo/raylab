"""Trainer and configuration for TRPO."""
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.utils.annotations import override

from raylab.agents import Trainer, with_common_config
from .trpo_policy import TRPOTorchPolicy

DEFAULT_CONFIG = with_common_config(
    {
        # Trust region constraint
        "delta": 0.01,
        # Number of actions to sample per state for Fisher vector product approximation
        "fvp_samples": 10,
        # For GAE(\gamma, \lambda)
        "lambda": 0.97,
        # Number of iterations to fit value function
        "val_iters": 80,
        # Options for critic optimizer
        "torch_optimizer": {"type": "Adam", "lr": 1e-3},
        # Whether to use Generalized Advantage Estimation
        "use_gae": True,
        # Configuration for Conjugate Gradient
        "cg_iters": 10,
        "cg_damping": 1e-3,
        # Whether to use a line search to calculate policy update.
        # Effectively turns TRPO into Natural PG when turned off.
        "line_search": True,
        "line_search_options": {
            "accept_ratio": 0.1,
            "backtrack_ratio": 0.8,
            "max_backtracks": 15,
            "atol": 1e-7,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and value function. No layers means the component is
        # linear in states or actions.
        "module": {"type": "OnPolicyActorCritic", "torch_script": True},
        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "raylab.utils.exploration.StochasticActor",
        },
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"explore": True},
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
            _ = self.optimizer.step()

        res = self.collect_metrics()
        timesteps = self.optimizer.num_steps_sampled - self.global_vars["timestep"]
        res.update(timesteps_this_iter=timesteps, info=res.get("info", {}))
        return res
