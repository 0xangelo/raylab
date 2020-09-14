"""Trainer and configuration for SVG(inf)."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override
from ray.tune import Trainable

from raylab.agents import Trainer
from raylab.agents.model_based import set_policy_with_env_fn
from raylab.options import configure
from raylab.options import option
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.replay_buffer import ReplayField

from .policy import SVGInfTorchPolicy


@configure
@option("buffer_size", 500000, help="Size of the replay buffer")
@option(
    "updates_per_step",
    1.0,
    help="Model and Value function updates per step in the environment",
)
@option("evaluation_config/explore", True)
@option("num_workers", 0, override=True)
@option("rollout_fragment_length", 1, override=True)
@option("batch_mode", "complete_episodes", override=True)
@option("train_batch_size", 128, override=True)
class SVGInfTrainer(Trainer):
    """Single agent trainer for SVG(inf)."""

    # pylint:disable=attribute-defined-outside-init

    _name = "SVG(inf)"
    _policy = SVGInfTorchPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=config["num_workers"]
        )
        set_policy_with_env_fn(self.workers, fn_type="reward")

        policy = self.get_policy()
        self.replay = NumpyReplayBuffer(
            policy.observation_space, policy.action_space, config["buffer_size"]
        )
        self.replay.add_fields(ReplayField(SampleBatch.ACTION_LOGP))
        self.replay.seed(config["seed"])

    @override(Trainable)
    def step(self):
        init_timesteps = self.metrics.num_steps_sampled
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        samples = worker.sample()
        self.metrics.num_steps_sampled += samples.count
        for row in samples.rows():
            self.replay.add(row)
        stats = policy.get_exploration_info()

        with policy.learning_off_policy():
            for _ in range(int(samples.count * self.config["updates_per_step"])):
                batch = self.replay.sample(self.config["train_batch_size"])
                off_policy_stats = policy.learn_on_batch(batch)
                self.metrics.num_steps_trained += batch.count
        stats.update(off_policy_stats)

        stats.update(policy.learn_on_batch(samples))

        timesteps_this_iter = self.metrics.num_steps_sampled - init_timesteps
        return self._log_metrics(stats, timesteps_this_iter)

    def _log_metrics(self, learner_stats, timesteps_this_iter):
        res = self.collect_metrics()
        res.update(timesteps_this_iter=timesteps_this_iter, learner=learner_stats)
        return res
