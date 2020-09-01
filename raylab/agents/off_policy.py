# pylint:disable=missing-module-docstring
from typing import Tuple

from ray.rllib import RolloutWorker
from ray.rllib import SampleBatch
from ray.rllib.utils import override
from ray.tune import Trainable

from raylab.agents import trainer
from raylab.agents.trainer import Trainer
from raylab.utils.replay_buffer import NumpyReplayBuffer


@trainer.configure
@trainer.option(
    "policy_improvements",
    default=1,
    help="""Policy improvement steps after each sample call to the rollout worker.

    Example:
        With a 'rollout_fragment_length' of 1 and 'policy_improvement' equal to 10,
        will perform 10 policy updates with minibatch size 'train_batch_size' per
        environment step.
    """,
)
@trainer.option(
    "learning_starts",
    default=0,
    help="Sample this many steps before starting optimization.",
)
@trainer.option("train_batch_size", default=128, override=True)
@trainer.option("rollout_fragment_length", default=1, override=True)
@trainer.option("num_workers", default=0, override=True)
@trainer.option("buffer_size", default=500000, help="Size of the replay buffer")
class OffPolicyTrainer(Trainer):
    """Generic trainer for off-policy agents."""

    # pylint:disable=attribute-defined-outside-init,abstract-method

    @override(Trainer)
    def _init(self, config, env_creator):
        self.validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
        )
        self.build_replay_buffer(config)

    @staticmethod
    def validate_config(config: dict):
        """Assert configuration values are valid."""
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."
        assert (
            config["policy_improvements"] >= 0
        ), "Number of policy improvement steps must be non-negative"

    def build_replay_buffer(self, config: dict):
        """Construct replay buffer to hold samples."""
        policy = self.get_policy()
        self.replay = NumpyReplayBuffer(
            policy.observation_space, policy.action_space, config["buffer_size"]
        )
        self.replay.seed(config["seed"])

    @property
    def worker(self) -> RolloutWorker:
        """The rollout worker."""
        return self.workers.local_worker()

    @override(Trainable)
    def step(self):
        pre_learning_steps = self.sample_until_learning_starts()

        timesteps_this_iter = 0
        while timesteps_this_iter < max(self.config["timesteps_per_iteration"], 1):
            sample_count, info = self.single_iteration()
            timesteps_this_iter += sample_count

        timesteps_this_iter += pre_learning_steps
        return self._log_metrics(
            learner_stats=info, timesteps_this_iter=timesteps_this_iter
        )

    def sample_until_learning_starts(self) -> int:
        """
        Sample enough transtions so that 'learning_starts' steps are collected before
        the next policy update.
        """
        learning_starts = self.config["learning_starts"]
        sample_count = 0
        while self.metrics.num_steps_sampled + sample_count < learning_starts:
            samples = self.worker.sample()
            sample_count += samples.count
            self.add_to_buffer(samples)

        if sample_count:
            self.update_steps_sampled(sample_count)
        return sample_count

    def single_iteration(self) -> Tuple[int, dict]:
        """Run one logical iteration of training.

        Returns:
            A tuple with the number of timesteps collected in the environment
            and the info dict from this iteration.
        """
        info = {}
        samples = self.worker.sample()
        self.update_steps_sampled(samples.count)
        self.add_to_buffer(samples)

        policy = self.worker.get_policy()
        info.update(policy.get_exploration_info())

        self._before_replay_steps(policy)
        for _ in range(int(self.config["policy_improvements"])):
            batch = self.replay.sample(self.config["train_batch_size"])
            info.update(policy.learn_on_batch(batch))
            self.metrics.num_steps_trained += batch.count

        return samples.count, info

    def _before_replay_steps(self, policy):  # pylint:disable=unused-argument
        pass

    def add_to_buffer(self, samples: SampleBatch):
        """Add sample batch to replay buffer"""
        for row in samples.rows():
            self.replay.add(row)

    def update_steps_sampled(self, count: int):
        """Update the number of steps sampled in the environment.

        Updates the standard metrics, global vars, worker vars, and policy
        global timestep (important to keep exploration and schedule in sync).
        """
        self.metrics.num_steps_sampled += count
        self.global_vars["timestep"] += count
        self._broadcast_global_vars()

    def _broadcast_global_vars(self):
        self.workers.foreach_worker(lambda w: w.set_global_vars(self.global_vars))

    def _log_metrics(self, learner_stats: dict, timesteps_this_iter: int) -> dict:
        res = self.collect_metrics()
        res.update(timesteps_this_iter=timesteps_this_iter, learner=learner_stats)
        return res
