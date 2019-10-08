"""
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor.
"""
import time

from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.agents.trainer import Trainer, with_common_config

from raylab.utils.replay_buffer import ReplayBuffer
from raylab.algorithms.sac.sac_policy import SACTorchPolicy


DEFAULT_CONFIG = with_common_config(
    {
        # === Entropy ===
        # Target entropy to optimize the temperature parameter towards
        # If None, will use the heuristic provided in the SAC paper:
        # H = -dim(A), where A is the action space
        "target_entropy": None,
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": False,
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
                "input_dependent_scale": False,
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


class SACTrainer(Trainer):
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

        start = time.time()
        steps_sampled = 0
        while True:
            self.update_exploration_phase()

            samples = worker.sample()
            steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count

            if (
                time.time() - start >= self.config["min_iter_time_s"]
                and steps_sampled >= self.config["timesteps_per_iteration"]
            ):
                break

        self.optimizer.num_steps_sampled += steps_sampled

        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=steps_sampled,
            info=dict(learner=stats, **res.get("info", {})),
        )
        return res

    # === New Methods ===

    def update_exploration_phase(self):
        """Signal to policies if training is still in the pure exploration phase."""
        global_timestep = self.optimizer.num_steps_sampled
        pure_expl_steps = self.config["pure_exploration_steps"]
        if pure_expl_steps:
            only_explore = global_timestep < pure_expl_steps
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, _: p.set_pure_exploration_phase(only_explore)
            )

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] == 0, "No point in using additional workers."
