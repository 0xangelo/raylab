"""Trainer and configuration for SVG(1)."""
from ray.rllib.agents.trainer import with_base_config
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats

from .svg_base import SVG_BASE_CONFIG, SVGBaseTrainer
from .svg_one_policy import SVGOneTorchPolicy


DEFAULT_CONFIG = with_base_config(
    SVG_BASE_CONFIG,
    {
        # === Optimization ===
        # PyTorch optimizer to use
        "torch_optimizer": "Adam",
        # Optimizer options for each component.
        # Valid keys include: 'model', 'value', and 'policy'
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "critic": {"lr": 1e-3},
            "actor": {"lr": 1e-3},
        },
        # === Regularization ===
        # Options for adaptive KL coefficient. See raylab.utils.adaptive_kl
        "kl_schedule": {},
        # Whether to penalize KL divergence with the current policy or past policies
        # that generated the replay pool.
        "replay_kl": True,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {"name": "SVGModule", "torch_script": False},
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
            # Options for parameter noise exploration
            # Until this many timesteps have elapsed, the agent's policy will be
            # ignored & it will instead take uniform random actions. Can be used in
            # conjunction with learning_starts (which controls when the first
            # optimization step happens) to decrease dependence of exploration &
            # optimization on initial policy parameters. Note that this will be
            # disabled when the action noise scale is set to 0 (e.g during evaluation).
            "pure_exploration_steps": 1000,
        },
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"explore": False},
    },
)


class SVGOneTrainer(SVGBaseTrainer):
    """Single agent trainer for SVG(1)."""

    # pylint: disable=attribute-defined-outside-init

    _name = "SVG(1)"
    _default_config = DEFAULT_CONFIG
    _policy = SVGOneTorchPolicy

    @override(SVGBaseTrainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        while not self._iteration_done():
            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            if not self.config["replay_kl"]:
                policy.update_old_policy()
            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                learner_stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count
            learner_stats.update(policy.update_kl_coeff(samples))

        return self._log_metrics(learner_stats)
