# pylint: disable=missing-module-docstring
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers import PolicyOptimizer

from raylab.agents import Trainer, with_common_config
from raylab.utils.dictionaries import deep_merge
from raylab.utils.replay_buffer import ReplayBuffer

BASE_CONFIG = with_common_config(
    {
        # === Replay buffer ===
        # Size of the replay buffer.
        "buffer_size": 500000,
        # === Optimization ===
        # Wait until this many steps have been sampled before starting optimization.
        "learning_starts": 0,
        # === Common config defaults ===
        "num_workers": 0,
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        "train_batch_size": 128,
    }
)


def with_base_config(config):
    """Returns the given config dict merged with the base off-policy configuration."""
    return deep_merge(BASE_CONFIG, config, True)


class GenericOffPolicyTrainer(Trainer):
    """Generic trainer for off-policy agents."""

    # pylint: disable=attribute-defined-outside-init

    _name = ""
    _default_config = None
    _policy = None
    _extra_replay_keys = ()

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
        )

        # Dummy optimizer to log stats since Trainer.collect_metrics is coupled with it
        self.optimizer = PolicyOptimizer(self.workers)
        self.replay = ReplayBuffer(
            config["buffer_size"], extra_keys=self._extra_replay_keys
        )

    @override(Trainer)
    def _train(self):
        self.sample_until_learning_starts()

        worker = self.workers.local_worker()
        policy = worker.get_policy()
        while not self._iteration_done():
            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            for row in samples.rows():
                self.replay.add(row)

            self._before_replay_steps(policy)
            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                stats = get_learner_stats(policy.learn_on_batch(batch))
                self.optimizer.num_steps_trained += batch.count

        return self._log_metrics(stats)

    def sample_until_learning_starts(self):
        """
        Sample enough transtions so that 'learning_starts' steps are collected before
        the next policy update.
        """
        learning_starts = self.config["learning_starts"]
        samples_count = self.config["rollout_fragment_length"]
        worker = self.workers.local_worker()
        while self.optimizer.num_steps_sampled < learning_starts - samples_count:
            samples = worker.sample()
            self.optimizer.num_steps_sampled += samples.count
            self.global_vars["timestep"] += samples.count
            for row in samples.rows():
                self.replay.add(row)

    def _before_replay_steps(self, policy):  # pylint:disable=unused-argument
        pass

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."
