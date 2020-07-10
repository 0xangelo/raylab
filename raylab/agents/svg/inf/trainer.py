"""Trainer and configuration for SVG(inf)."""
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents import Trainer
from raylab.agents import trainer
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.replay_buffer import ReplayField

from .policy import SVGInfTorchPolicy


@trainer.config(
    "vf_loss_coeff",
    1.0,
    info="Weight of the fitted V loss in the joint model-value loss",
)
@trainer.config("max_grad_norm", 10.0, info="Clip gradient norms by this value")
@trainer.config(
    "max_is_ratio", 5.0, info="Clip importance sampling weights by this value"
)
@trainer.config(
    "polyak",
    0.995,
    info="Interpolation factor in polyak averaging for target networks.",
)
@trainer.config("torch_optimizer/on_policy", {"type": "Adam", "lr": 1e-3})
@trainer.config("torch_optimizer/off_policy", {"type": "Adam", "lr": 1e-3})
@trainer.config(
    "updates_per_step",
    1.0,
    info="Model and Value function updates per step in the environment",
)
@trainer.config("buffer_size", 500000, info="Size of the replay buffer")
@trainer.config(
    "kl_schedule",
    {},
    info="Options for adaptive KL coefficient. See raylab.utils.adaptive_kl",
)
@trainer.config("module", {"type": "SVGModule-v0"}, override=True)
@trainer.config(
    "exploration_config/type", "raylab.utils.exploration.StochasticActor", override=True
)
@trainer.config("evaluation_config/explore", True)
@trainer.config("num_workers", 0, override=True)
@trainer.config("rollout_fragment_length", 1, override=True)
@trainer.config("batch_mode", "complete_episodes", override=True)
@trainer.config("train_batch_size", 128, override=True)
@Trainer.with_base_specs
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
        policy = self.get_policy()
        policy.set_reward_from_config(config["env"], config["env_config"])

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
        res.update(
            timesteps_this_iter=timesteps_this_iter,
            info=dict(learner=learner_stats, **res.get("info", {})),
        )
        return res
