"""Trainer and configuration for SVG(1)."""
from ray.rllib.agents.trainer import with_base_config
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation.metrics import get_learner_stats

from raylab.algorithms.svg import SVG_BASE_CONFIG, SVGBaseTrainer
from raylab.algorithms.svg.svg_one_policy import SVGOneTorchPolicy
from raylab.utils.replay_buffer import ReplayBuffer


DEFAULT_CONFIG = with_base_config(
    SVG_BASE_CONFIG,
    {
        # === Optimization ===
        # PyTorch optimizer to use
        "torch_optimizer": "Adam",
        # Optimizer options for each component.
        # Valid keys include: 'model', 'value', and 'policy'
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "value": {"lr": 1e-3},
            "policy": {"lr": 1e-3},
        },
    },
)


class SVGOneTrainer(SVGBaseTrainer):
    """Single agent trainer for SVG(1)."""

    # pylint: disable=attribute-defined-outside-init

    _name = "SVG(1)"
    _default_config = DEFAULT_CONFIG
    _policy = SVGOneTorchPolicy

    @override(SVGBaseTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        self.replay = ReplayBuffer(
            config["buffer_size"],
            extra_keys=[self._policy.ACTION_LOGP, "loc", "scale_diag"],
        )

    @override(SVGBaseTrainer)
    def _train(self):
        worker = self.workers.local_worker()
        policy = worker.get_policy()

        samples = worker.sample()
        self.optimizer.num_steps_sampled += samples.count
        for row in samples.rows():
            self.replay.add(row)

        for _ in range(samples.count):
            batch = self.replay.sample(self.config["train_batch_size"])
            learner_stats = get_learner_stats(policy.learn_on_batch(batch))
            self.optimizer.num_steps_trained += batch.count

        learner_stats.update(policy.update_kl_coeff(samples))

        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=samples.count,
            info=dict(learner=learner_stats, **res.get("info", {})),
        )
        return res
