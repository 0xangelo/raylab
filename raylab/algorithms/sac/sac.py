"""
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor.
"""
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers import PolicyOptimizer

from raylab.utils.replay_buffer import ReplayBuffer
from raylab.algorithms import Trainer, with_common_config
from raylab.algorithms.mixins import ExplorationPhaseMixin
from .sac_policy import SACTorchPolicy


DEFAULT_CONFIG = with_common_config(
    {
        # === Entropy ===
        # Target entropy to optimize the temperature parameter towards
        # If None, will use the heuristic provided in the SAC paper:
        # H = -dim(A), where A is the action space
        "target_entropy": None,
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": True,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 500000,
        # === Optimization ===
        # PyTorch optimizer to use for policy
        "policy_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
        # PyTorch optimizer to use for critic
        "critic_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
        # PyTorch optimizer to use for entropy coefficient
        "alpha_optimizer": {"name": "Adam", "options": {"lr": 1e-3}},
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
                "input_dependent_scale": True,
            },
            "critic": {
                "units": (400, 300),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
        },
        # === Rollout Worker ===
        "num_workers": 0,
        # === Exploration ===
        # Whether to sample only the mean action, mostly for evaluation purposes
        "mean_action_only": False,
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
        "evaluation_config": {"mean_action_only": True, "pure_exploration_steps": 0},
    }
)


class SACTrainer(ExplorationPhaseMixin, Trainer):
    """Single agent trainer for SAC."""

    # pylint: disable=attribute-defined-outside-init

    _name = "SAC"
    _default_config = DEFAULT_CONFIG
    _policy = SACTorchPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
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
