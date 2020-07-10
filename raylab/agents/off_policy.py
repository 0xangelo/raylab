# pylint:disable=missing-module-docstring
from ray.rllib.utils import override

from raylab.agents import trainer
from raylab.agents.trainer import Trainer
from raylab.utils.replay_buffer import NumpyReplayBuffer


@trainer.config("train_batch_size", 128, override=True)
@trainer.config("batch_mode", "complete_episodes", override=True)
@trainer.config("rollout_fragment_length", 1, override=True)
@trainer.config("num_workers", 0, override=True)
@trainer.config(
    "learning_starts", 0, info="Sample this many steps before starting optimization."
)
@trainer.config("buffer_size", 500000, info="Size of the replay buffer")
@Trainer.with_base_specs
class OffPolicyTrainer(Trainer):
    """Generic trainer for off-policy agents."""

    # pylint:disable=attribute-defined-outside-init
    _name = ""
    _policy = None

    @override(Trainer)
    def _init(self, config, env_creator):
        self.validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
        )
        self.build_replay_buffer(config)

    @override(Trainer)
    def _train(self):
        pre_learning_steps = self.sample_until_learning_starts()
        timesteps_this_iter = 0

        worker = self.workers.local_worker()
        policy = worker.get_policy()
        stats = {}
        while timesteps_this_iter < max(self.config["timesteps_per_iteration"], 1):
            samples = worker.sample()
            timesteps_this_iter += samples.count
            for row in samples.rows():
                self.replay.add(row)
            stats.update(policy.get_exploration_info())

            self._before_replay_steps(policy)
            for _ in range(samples.count):
                batch = self.replay.sample(self.config["train_batch_size"])
                stats.update(policy.learn_on_batch(batch))
                self.metrics.num_steps_trained += batch.count

        self.metrics.num_steps_sampled += timesteps_this_iter
        return self._log_metrics(stats, timesteps_this_iter + pre_learning_steps)

    def build_replay_buffer(self, config):
        """Construct replay buffer to hold samples."""
        policy = self.get_policy()
        self.replay = NumpyReplayBuffer(
            policy.observation_space, policy.action_space, config["buffer_size"]
        )
        self.replay.seed(config["seed"])

    def sample_until_learning_starts(self):
        """
        Sample enough transtions so that 'learning_starts' steps are collected before
        the next policy update.
        """
        learning_starts = self.config["learning_starts"]
        worker = self.workers.local_worker()
        sample_count = 0
        while self.metrics.num_steps_sampled + sample_count < learning_starts:
            samples = worker.sample()
            sample_count += samples.count
            for row in samples.rows():
                self.replay.add(row)

        if sample_count:
            self.metrics.num_steps_sampled += sample_count
            self.global_vars["timestep"] = self.metrics.num_steps_sampled
            self.workers.foreach_worker(lambda w: w.set_global_vars(self.global_vars))

        return sample_count

    def _before_replay_steps(self, policy):  # pylint:disable=unused-argument
        pass

    def _log_metrics(self, learner_stats, timesteps_this_iter):
        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=timesteps_this_iter,
            info=dict(learner=learner_stats, **res.get("info", {})),
        )
        return res

    @staticmethod
    def validate_config(config):
        """Assert configuration values are valid."""
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."
