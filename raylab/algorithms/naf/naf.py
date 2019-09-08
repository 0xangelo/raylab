"""Continuous Q-Learning with Normalized Advantage Functions."""
import time

from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.agents.trainer import Trainer, with_common_config

from raylab.utils.replay_buffer import ReplayBuffer
from raylab.algorithms.naf.naf_policy import NAFTorchPolicy


DEFAULT_CONFIG = with_common_config(
    {
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 500000,
        # === Time Limits ===
        # Whether to ignore horizon termination and bootstrap from final observation.
        # This is used to set targets for the action value function.
        "timeout_bootstrap": True,
        # === Network ===
        # Size and activation of the fully connected network computing the logits
        # for the normalized advantage function. No layers means the Q function is
        # linear in states and actions.
        "module": {"layers": [400, 300], "activation": "elu", "ortho_init_gain": 0.2},
        # === Optimization ===
        # Name of Pytorch optimizer class
        "torch_optimizer": "Adam",
        # Keyword arguments to be passed to the PyTorch optimizer
        "torch_optimizer_options": {"lr": 1e-3},
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Exploration ===
        # Which type of exploration to use. Possible types include
        # None: use the greedy policy to act
        # parameter_noise: use parameter space noise
        # diag_gaussian: use i.i.d gaussian action space noise independently for each
        #     action dimension
        # full_gaussian: use gaussian action space noise where the precision matrix is
        #     given by the advantage function P matrix
        "exploration": None,
        # Scaling term of the lower triangular matrix for the multivariate gaussian
        # action distribution
        "scale_tril_coeff": 1.0,
        # Gaussian stddev for diagonal gaussian action space noise
        "diag_gaussian_stddev": 0.1,
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
        "evaluation_config": {"exploration": None},
    }
)


class NAFTrainer(Trainer):
    """Single agent trainer for NAF."""

    # pylint: disable=attribute-defined-outside-init

    _name = "NAF"
    _default_config = DEFAULT_CONFIG
    _policy = NAFTorchPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        policy_cls = self._policy
        self.workers = self._make_workers(
            env_creator, policy_cls, config, num_workers=0
        )
        self.replay = ReplayBuffer(config["buffer_size"])
        self.learner_stats = {}
        self.num_steps_sampled = 0
        self.num_steps_trained = 0
        self.episode_history = []

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
                self.replay.add(
                    row[SampleBatch.CUR_OBS],
                    row[SampleBatch.ACTIONS],
                    row[SampleBatch.REWARDS],
                    row[SampleBatch.NEXT_OBS],
                    row[SampleBatch.DONES],
                    weight=None,
                )

            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                self.learner_stats = policy.learn_on_batch(batch)
                self.num_steps_trained += batch.count

            if (
                time.time() - start >= self.config["min_iter_time_s"]
                and steps_sampled >= self.config["timesteps_per_iteration"]
            ):
                break

        self.num_steps_sampled += steps_sampled

        return {**self.collect_metrics(), "timesteps_this_iter": steps_sampled}

    @override(Trainer)
    def collect_metrics(self):  # pylint: disable=arguments-differ
        episodes, _ = collect_episodes(
            local_worker=self.workers.local_worker(),
            timeout_seconds=self.config["collect_metrics_timeout"],
        )
        orig_episodes = list(episodes)

        min_history = self.config["metrics_smoothing_episodes"]
        missing = min_history - len(episodes)
        if missing > 0:
            episodes.extend(self.episode_history[-missing:])
            assert len(episodes) <= min_history
        self.episode_history.extend(orig_episodes)
        self.episode_history = self.episode_history[-min_history:]
        res = summarize_episodes(episodes, orig_episodes)
        res.update(
            info={
                "num_steps_trained": self.num_steps_trained,
                "num_steps_sampled": self.num_steps_sampled,
                "learner": self.learner_stats,
            }
        )
        return res

    # === New Methods ===

    def update_exploration_phase(self):
        global_timestep = self.num_steps_sampled
        pure_expl_steps = self.config["pure_exploration_steps"]
        if pure_expl_steps:
            only_explore = global_timestep < pure_expl_steps
            self.workers.local_worker().foreach_trainable_policy(
                lambda p, _: p.set_pure_exploration_phase(only_explore)
            )

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] >= 0, "No point in using additional workers."
