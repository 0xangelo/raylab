"""Learning Continuous Control Policies by Stochastic Value Gradients.

http://papers.nips.cc/paper/5796-learning-continuous-control-policies-by-stochastic-value-gradients
"""
from ray.rllib.agents.trainer import with_common_config

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
            "initial_coeff": 0.2,
            "desired_kl": 0.01,
            "adaptation_coeff": 2.0,
            "threshold": 1.5,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "layers": [100, 100],
                "activation": "Tanh",
                "initializer_options": {"name": "xavier_uniform"},
                "input_dependent_scale": False,
            },
            "value": {
                "layers": [400, 200],
                "activation": "Tanh",
                "initializer_options": {"name": "xavier_uniform"},
            },
            "model": {
                "layers": [40, 40],
                "activation": "Tanh",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
        },
    }
)
