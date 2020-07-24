"""Trainer and configuration for SVG(inf)."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents import Trainer
from raylab.agents import trainer
from raylab.agents.model_based import set_policy_with_env_fn
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.replay_buffer import ReplayField

from .policy import SVGInfTorchPolicy


@trainer.configure
@trainer.option(
    "vf_loss_coeff",
    1.0,
    help="Weight of the fitted V loss in the joint model-value loss",
)
@trainer.option("max_grad_norm", 10.0, help="Clip gradient norms by this value")
@trainer.option(
    "max_is_ratio", 5.0, help="Clip importance sampling weights by this value"
)
@trainer.option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@trainer.option("torch_optimizer/on_policy", {"type": "Adam", "lr": 1e-3})
@trainer.option("torch_optimizer/off_policy", {"type": "Adam", "lr": 1e-3})
@trainer.option(
    "updates_per_step",
    1.0,
    help="Model and Value function updates per step in the environment",
)
@trainer.option("buffer_size", 500000, help="Size of the replay buffer")
@trainer.option(
    "kl_schedule/",
    help="Options for adaptive KL coefficient. See raylab.utils.adaptive_kl",
    allow_unknown_subkeys=True,
)
@trainer.option("module", {"type": "SVGModule-v0"}, override=True)
@trainer.option(
    "exploration_config/type", "raylab.utils.exploration.StochasticActor", override=True
)
@trainer.option("evaluation_config/explore", True)
@trainer.option("num_workers", 0, override=True)
@trainer.option("rollout_fragment_length", 1, override=True)
@trainer.option("batch_mode", "complete_episodes", override=True)
@trainer.option("train_batch_size", 128, override=True)
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

    @override(Trainer)
    def _train(self):
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
