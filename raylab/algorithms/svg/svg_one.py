"""Learning Continuous Control Policies by Stochastic Value Gradients.

http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients
"""
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.utils.annotations import override
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.evaluation.metrics import get_learner_stats

from raylab.algorithms.svg.svg_one_policy import SVGOneTorchPolicy
from raylab.utils.replay_buffer import ReplayBuffer


DEFAULT_CONFIG = with_common_config(
    {
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 500000,
        # === Optimization ===
        # PyTorch optimizer to use
        "torch_optimizer": "Adam",
        # Optimizer options for each component.
        # Valid key include: 'model', 'value', and 'policy'
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "value": {"lr": 1e-3},
            "policy": {"lr": 1e-3},
        },
        # Weight of the fitted V loss in the joint model-value loss
        "vf_loss_coeff": 1.0,
        # Clip gradient norms by this value
        "max_grad_norm": 1.0,
        # Clip importance sampling weights by this value
        "max_is_ratio": 5.0,
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Rollout Worker ===
        "num_workers": 0,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "layers": [100, 100],
                "activation": "Tanh",
                "initializer": "xavier_uniform",
                "initializer_options": {"gain": 5 / 3},
                "input_dependent_scale": False,
            },
            "value": {
                "layers": [400, 200],
                "activation": "Tanh",
                "initializer": "xavier_uniform",
                "initializer_options": {"gain": 5 / 3},
            },
            "model": {
                "layers": [40, 40],
                "activation": "Tanh",
                "initializer": "xavier_uniform",
                "initializer_options": {"gain": 5 / 3},
            },
        },
    }
)


class SVGOneTrainer(Trainer):
    """Single agent trainer for SVG(1)."""

    # pylint: disable=attribute-defined-outside-init

    _name = "SVG(1)"
    _default_config = DEFAULT_CONFIG
    _policy = SVGOneTorchPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config(config)
        self.workers = self._make_workers(
            env_creator, self._policy, config, num_workers=0
        )
        self.workers.foreach_worker(
            lambda w: w.foreach_trainable_policy(
                lambda p, _: p.set_reward_fn(w.env.reward_fn)
            )
        )
        # Dummy optimizer to log stats
        self.optimizer = PolicyOptimizer(self.workers)
        self.replay = ReplayBuffer(
            config["buffer_size"], extra_keys=[self._policy.ACTION_LOGP]
        )

    @override(Trainer)
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

        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=samples.count,
            info=dict(learner=learner_stats, **res.get("info", {})),
        )
        return res

    # === New Methods ===

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] == 0, "No point in using additional workers."
