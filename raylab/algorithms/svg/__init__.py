"""Learning Continuous Control Policies by Stochastic Value Gradients.

http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients
"""
import numpy as np
from ray import tune
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.utils.annotations import override

from raylab.utils.replay_buffer import ReplayBuffer
from raylab.algorithms import Trainer, with_common_config


SVG_BASE_CONFIG = with_common_config(
    {
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 500000,
        # === Optimization ===
        # Weight of the fitted V loss in the joint model-value loss
        "vf_loss_coeff": 1.0,
        # Clip gradient norms by this value
        "max_grad_norm": 10.0,
        # Clip importance sampling weights by this value
        "max_is_ratio": 5.0,
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Rollout Worker ===
        "num_workers": 0,
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": 0.0,
            "desired_kl": 0.01,
            "adaptation_coeff": 2.0,
            "threshold": 1.5,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "name": "SVGModule",
            "torch_script": True,
            "mean_action_only": False,
            "actor": {
                "units": [100, 100],
                "activation": "Tanh",
                "initializer_options": {"name": "xavier_uniform"},
                "input_dependent_scale": False,
            },
            "critic": {
                "units": [400, 200],
                "activation": "Tanh",
                "initializer_options": {"name": "xavier_uniform"},
                "target_vf": True,
            },
            "model": {
                "encoder": "svg_paper",
                "residual": True,
                "units": [40, 40],
                "activation": "Tanh",
                "delay_action": True,
                "initializer_options": {"name": "xavier_uniform"},
                "input_dependent_scale": False,
            },
        },
        # === Debugging ===
        # Whether to log detailed information about the actions selected in each episode
        "debug": False,
    }
)


class SVGBaseTrainer(Trainer):
    """Base Trainer to set up SVG algorithms. This should not be instantiated."""

    # pylint: disable=abstract-method,no-member,attribute-defined-outside-init

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
        # Dummy optimizer to log stats since Trainer.collect_metrics is coupled with it
        self.optimizer = PolicyOptimizer(self.workers)
        self.replay = ReplayBuffer(
            config["buffer_size"], extra_keys=[self._policy.ACTION_LOGP]
        )

    # === New Methods ===

    @staticmethod
    def _validate_config(config):
        assert config["num_workers"] == 0, "No point in using additional workers."

        if config["debug"]:
            start_callback = config["callbacks"]["on_episode_start"]

            def on_episode_start(info):
                episode = info["episode"]
                episode.user_data["actions"] = []
                if start_callback:
                    start_callback(info)

            step_callback = config["callbacks"]["on_episode_step"]

            def on_episode_step(info):
                episode = info["episode"]
                if episode.length > 0:
                    episode.user_data["actions"].append(episode.last_action_for())
                if step_callback:
                    step_callback(info)

            end_callback = config["callbacks"]["on_episode_end"]

            def on_episode_end(info):
                eps = info["episode"]
                mean_action = np.mean(eps.user_data["actions"], axis=0)
                for idx, mean in enumerate(mean_action):
                    eps.custom_metrics[f"mean_action-{idx}"] = mean
                if end_callback:
                    end_callback(info)

            config["callbacks"]["on_episode_start"] = tune.function(on_episode_start)
            config["callbacks"]["on_episode_step"] = tune.function(on_episode_step)
            config["callbacks"]["on_episode_end"] = tune.function(on_episode_end)
