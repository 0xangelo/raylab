"""Trainer and configuration for MAPO."""
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.utils.annotations import override

from raylab.agents import Trainer, with_common_config
from raylab.utils.replay_buffer import ReplayBuffer

from .mapo_policy import MAPOTorchPolicy


DEFAULT_CONFIG = with_common_config(
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
    }
)


class MAPOTrainer(Trainer):
    """Single agent trainer for Model-Aware Policy Optimization."""

    # pylint: disable=attribute-defined-outside-init

    _name = "MAPO"
    _default_config = DEFAULT_CONFIG
    _policy = MAPOTorchPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)

        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
        )

        if config["true_model"]:
            self.set_transition_kernel()

        # Dummy optimizer to log stats
        self.optimizer = PolicyOptimizer(self.workers)
        self.replay = ReplayBuffer(config["buffer_size"])

    @override(Trainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        self.sample_until_learning_starts(worker)

        while not self._iteration_done():
            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count

        return self._log_metrics(stats)

    def sample_until_learning_starts(self, worker):
        """
        Sample enough transtions so that 'learning_starts' steps are collected before
        the next policy update.
        """
        learning_starts = self.config["learning_starts"]
        samples_count = self.config["rollout_fragment_length"]
        while self.optimizer.num_steps_sampled < learning_starts - samples_count:
            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            self.global_vars["timestep"] += samples.count
            for row in samples.rows():
                self.replay.add(row)

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."

    def set_transition_kernel(self):
        """Make policies use the real transition kernel."""
        self.workers.foreach_worker(
            lambda w: w.foreach_trainable_policy(
                lambda p, _: p.set_transition_kernel(w.env.transition_fn)
            )
        )
