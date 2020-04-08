"""Trainer and configuration for SVG(inf)."""
from ray.rllib.agents.trainer import with_base_config
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats

from .svg_base import SVG_BASE_CONFIG, SVGBaseTrainer
from .svg_inf_policy import SVGInfTorchPolicy


DEFAULT_CONFIG = with_base_config(
    SVG_BASE_CONFIG,
    {
        # === Optimization ===
        # Name of Pytorch optimizer class for dynamics model and value function
        "off_policy_optimizer": "Adam",
        # Keyword arguments to be passed to the off-policy optimizer
        "off_policy_optimizer_options": {"lr": 1e-3},
        # Name of Pytorch optimizer class for paremetrized policy
        "on_policy_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "on_policy_optimizer_options": {"lr": 3e-4},
        # Model and Value function updates per step in the environment
        "updates_per_step": 1.0,
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
    },
)


class SVGInfTrainer(SVGBaseTrainer):
    """Single agent trainer for SVG(inf)."""

    # pylint: disable=attribute-defined-outside-init

    _name = "SVG(inf)"
    _default_config = DEFAULT_CONFIG
    _policy = SVGInfTorchPolicy

    @override(SVGBaseTrainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        samples = worker.sample()
        self.optimizer.num_steps_sampled += samples.count
        for row in samples.rows():
            self.replay.add(row)

        with policy.learning_off_policy():
            for _ in range(int(samples.count * self.config["updates_per_step"])):
                batch = self.replay.sample(self.config["train_batch_size"])
                off_policy_stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count

        on_policy_stats = get_learner_stats(policy.learn_on_batch(samples))

        learner_stats = {**off_policy_stats, **on_policy_stats}
        return self._log_metrics(learner_stats)
