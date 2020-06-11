"""Trainer and configuration for SVG(1)."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.off_policy import OffPolicyTrainer
from raylab.agents.off_policy import with_base_config
from raylab.utils.replay_buffer import ReplayField

from .policy import SVGOneTorchPolicy


DEFAULT_CONFIG = with_base_config(
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
        # Weight of the fitted V loss in the joint model-value loss
        "vf_loss_coeff": 1.0,
        # Clip gradient norms by this value
        "max_grad_norm": 10.0,
        # Clip importance sampling weights by this value
        "max_is_ratio": 5.0,
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Regularization ===
        # Options for adaptive KL coefficient. See raylab.utils.adaptive_kl
        "kl_schedule": {"initial_coeff": 0},
        # Whether to penalize KL divergence with the current policy or past policies
        # that generated the replay pool.
        "replay_kl": True,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {"type": "SVGModule"},
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


class SVGOneTrainer(OffPolicyTrainer):
    """Single agent trainer for SVG(1)."""

    # pylint: disable=attribute-defined-outside-init
    _name = "SVG(1)"
    _default_config = DEFAULT_CONFIG
    _policy = SVGOneTorchPolicy

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        self.get_policy().set_reward_from_config(config["env"], config["env_config"])

    @override(OffPolicyTrainer)
    def build_replay_buffer(self, config):
        super().build_replay_buffer(config)
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))

    @override(OffPolicyTrainer)
    def _before_replay_steps(self, policy):
        if not self.config["replay_kl"]:
            policy.update_old_policy()
