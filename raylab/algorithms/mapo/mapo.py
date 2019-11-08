"""Trainer and configuration for MAPO."""
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.utils.annotations import override

from raylab.algorithms import Trainer, with_common_config
from raylab.algorithms.mixins import ExplorationPhaseMixin, ParameterNoiseMixin
from raylab.utils.replay_buffer import ReplayBuffer

from .mapo_policy import MAPOTorchPolicy


DEFAULT_CONFIG = with_common_config(
    {
        # === MAPO model training ===
        # Type of model-training to use. Possible types include
        # decision_aware: policy gradient-aware model learning
        # mle: maximum likelihood estimation
        "model_loss": "decision_aware",
        # Type of the used p-norm of the distance between gradients.
        # Can be float('inf') for infinity norm.
        "norm_type": 2,
        # Number of initial next states to sample from the model when calculating the
        # model-aware deterministic policy gradient
        "num_model_samples": 4,
        # Length of the rollouts from each next state sampled
        "model_rollout_len": 1,
        # Gradient estimator for model-aware dpg. Possible types include:
        # score_function, pathwise_derivative
        "grad_estimator": "score_function",
        # === Debugging ===
        # Whether to use the environment's true model to sample states
        "true_model": False,
        # Degrade the true model using a constant bias, i.e., by adding a constant
        # vector to the model's output
        "model_bias": None,
        # Degrade the true model using zero-mean gaussian noise
        "model_noise_sigma": None,
        # === SQUASHING EXPLORATION PROBLEM ===
        # Maximum l1 norm of the policy's output vector before the squashing function
        "beta": 1.2,
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # Add gaussian noise to the action when calculating the Deterministic
        # Policy Gradient
        "target_policy_smoothing": True,
        # Additive Gaussian i.i.d. noise to add to actions inputs to target Q function
        "target_gaussian_sigma": 0.3,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 500000,
        # === Optimization ===
        # PyTorch optimizer to use for policy
        "policy_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
        # PyTorch optimizer to use for critic
        "critic_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
        # PyTorch optimizer to use for model
        "model_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
            },
            "critic": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
            "model": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
                "input_dependent_scale": False,
            },
        },
        # === Rollout Worker ===
        "num_workers": 0,
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        # === Exploration ===
        # Whether to act greedly or exploratory, mostly for evaluation purposes
        "greedy": False,
        # Which type of exploration to use. Possible types include
        # None: use the greedy policy to act
        # parameter_noise: use parameter space noise
        # gaussian: use i.i.d gaussian action space noise independently for each
        #     action dimension
        "exploration": None,
        # Options for parameter noise exploration
        "param_noise_spec": {
            "initial_stddev": 0.1,
            "desired_action_stddev": 0.2,
            "adaptation_coeff": 1.01,
        },
        # Additive Gaussian i.i.d. noise to add to actions before squashing
        "exploration_gaussian_sigma": 0.3,
        # Until this many timesteps have elapsed, the agent's policy will be
        # ignored & it will instead take uniform random actions. Can be used in
        # conjunction with learning_starts (which controls when the first
        # optimization step happens) to decrease dependence of exploration &
        # optimization on initial policy parameters. Note that this will be
        # disabled when the action noise scale is set to 0 (e.g during evaluation).
        "pure_exploration_steps": 1000,
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"greedy": True, "pure_exploration_steps": 0},
    }
)


class MAPOTrainer(ExplorationPhaseMixin, ParameterNoiseMixin, Trainer):
    """Single agent trainer for Model-Aware Policy Optimization."""

    # pylint: disable=attribute-defined-outside-init

    _name = "MAPO"
    _default_config = DEFAULT_CONFIG
    _policy = MAPOTorchPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        self._set_parameter_noise_callbacks(config)

        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
        )
        self.workers.foreach_worker(
            lambda w: w.foreach_trainable_policy(
                lambda p, _: (
                    p.set_reward_fn(w.env.reward_fn),
                    p.set_transition_fn(w.env.transition_fn)
                    if config["true_model"]
                    else None,
                )
            )
        )
        # Dummy optimizer to log stats
        self.optimizer = PolicyOptimizer(self.workers)
        self.replay = ReplayBuffer(config["buffer_size"])

    @override(Trainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        while not self._iteration_done():
            self.update_exploration_phase()

            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count

        return self._log_metrics(stats)

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["sample_batch_size"] >= 1
        ), "At least one sample must be collected."
        assert (
            config["batch_mode"] == "complete_episodes"
        ), "Must sample complete episodes."
