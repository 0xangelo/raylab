"""Trainer and configuration for SVG(1) with maximum entropy."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.off_policy import OffPolicyTrainer
from raylab.agents.off_policy import with_base_config
from raylab.utils.replay_buffer import ReplayField

from .policy import SoftSVGTorchPolicy


DEFAULT_CONFIG = with_base_config(
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
        # Weight of the fitted V loss in the joint model-value loss
        "vf_loss_coeff": 1.0,
        # Clip importance sampling weights by this value
        "max_is_ratio": 5.0,
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {"type": "MaxEntModelBased", "critic": {"target_vf": True}},
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
            "pure_exploration_steps": 1000,
        },
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"explore": False},
    }
)


class SoftSVGTrainer(OffPolicyTrainer):
    """Single agent trainer for SoftSVG."""

    # pylint: disable=attribute-defined-outside-init
    _name = "SoftSVG"
    _default_config = DEFAULT_CONFIG
    _policy = SoftSVGTorchPolicy

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        self.get_policy().set_reward_from_config(config["env"], config["env_config"])

    @override(OffPolicyTrainer)
    def build_replay_buffer(self, config):
        super().build_replay_buffer(config)
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))
