"""Trainer and configuration for SVG(1) with maximum entropy."""
from ray.rllib.agents.trainer import with_base_config
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats

from .svg_base import SVG_BASE_CONFIG, SVGBaseTrainer
from .soft_svg_policy import SoftSVGTorchPolicy


DEFAULT_CONFIG = with_base_config(
    SVG_BASE_CONFIG,
    {
        # === Entropy ===
        # Target entropy to optimize the temperature parameter towards
        # If "auto", will use the heuristic provided in the SAC paper:
        # H = -dim(A), where A is the action space
        "target_entropy": None,
        # === Optimization ===
        # PyTorch optimizers to use
        "torch_optimizer": {
            "model": {"type": "Adam", "lr": 1e-3},
            "actor": {"type": "Adam", "lr": 1e-3},
            "critic": {"type": "Adam", "lr": 1e-3},
            "alpha": {"type": "Adam", "lr": 1e-3},
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "type": "MaxEntModelBased",
            "torch_script": True,
            "critic": {"target_vf": True},
        },
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


class SoftSVGTrainer(SVGBaseTrainer):
    """Single agent trainer for SoftSVG."""

    _name = "SoftSVG"
    _default_config = DEFAULT_CONFIG
    _policy = SoftSVGTorchPolicy

    @override(SVGBaseTrainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        while not self._iteration_done():
            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                learner_stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count

        return self._log_metrics(learner_stats)
