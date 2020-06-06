"""Trainer and configuration for MAPO."""
from ray.rllib.utils import override

from raylab.agents.off_policy import OffPolicyTrainer
from raylab.agents.off_policy import with_base_config

from .policy import MAPOTorchPolicy


DEFAULT_CONFIG = with_base_config(
    {
        # === MAPO model training ===
        # Type of model-training to use. Possible types include
        # DAML: policy gradient-aware model learning
        # MLE: maximum likelihood estimation
        "model_loss": "DAML",
        # Type of the used p-norm of the distance between gradients.
        # Can be float('inf') for infinity norm.
        "norm_type": 2,
        # Number of initial next states to sample from the model when calculating the
        # model-aware deterministic policy gradient
        "num_model_samples": 4,
        # Gradient estimator for model-aware dpg. Possible types include
        # SF: score function
        # PD: pathwise derivative
        "grad_estimator": "SF",
        # KL regularization to avoid degenerate solutions (needs to be tuned)
        "mle_interpolation": 0.0,
        # === Debugging ===
        # Whether to use the environment's true model to sample states
        "true_model": False,
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 500000,
        # === Optimization ===
        # PyTorch optimizers to use
        "torch_optimizer": {
            "model": {"type": "Adam", "lr": 1e-3},
            "actor": {"type": "Adam", "lr": 1e-3},
            "critics": {"type": "Adam", "lr": 1e-3},
        },
        # Clip gradient norms for each component by the following values.
        "max_grad_norm": {
            "model": float("inf"),
            "actor": float("inf"),
            "critics": float("inf"),
        },
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # Wait until this many steps have been sampled before starting optimization.
        "learning_starts": 0,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {"type": "MAPOModule", "torch_script": False},
        # === Rollout Worker ===
        "num_workers": 0,
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
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
            "type": "raylab.utils.exploration.ParameterNoise",
            # Options for parameter noise exploration
            "param_noise_spec": {
                "initial_stddev": 0.1,
                "desired_action_stddev": 0.2,
                "adaptation_coeff": 1.01,
            },
            "pure_exploration_steps": 1000,
        },
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"explore": False},
    }
)


class MAPOTrainer(OffPolicyTrainer):
    """Single agent trainer for Model-Aware Policy Optimization."""

    # pylint: disable=attribute-defined-outside-init

    _name = "MAPO"
    _default_config = DEFAULT_CONFIG
    _policy = MAPOTorchPolicy

    @override(OffPolicyTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        policy = self.get_policy()
        policy.set_reward_from_config(config["env"], config["env_config"])
        if config["true_model"]:
            self.set_transition_kernel()

    def set_transition_kernel(self):
        """Make policies use the real transition kernel."""
        self.workers.foreach_worker(
            lambda w: w.foreach_trainable_policy(
                lambda p, _: p.set_transition_kernel(w.env.transition_fn)
            )
        )
