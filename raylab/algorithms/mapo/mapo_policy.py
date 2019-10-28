"""Policy for MAPO using PyTorch."""
import collections

import torch.nn as nn
from ray.rllib.utils.annotations import override

import raylab.policy as raypi
import raylab.modules as mods
import raylab.utils.pytorch as torch_util
import raylab.algorithms.mapo.mapo_module as mapom


OptimizerCollection = collections.namedtuple(
    "OptimizerCollection", "policy critic alpha"
)


class MAPOTorchPolicy(
    raypi.AdaptiveParamNoiseMixin,
    raypi.PureExplorationMixin,
    raypi.TargetNetworksMixin,
    raypi.TorchPolicy,
):
    """Model-Aware Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default configuration for MAPO."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.mapo.mapo import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module = nn.ModuleDict()
        module.update(self._make_policy(obs_space, action_space, config))

        def make_critic():
            return self._make_critic(obs_space, action_space, config)

        module.critics = nn.ModuleList([make_critic()])
        module.target_critics = nn.ModuleList([make_critic()])
        if config["clipped_double_q"]:
            module.critics.append(make_critic())
            module.target_critics.append(make_critic())
        module.target_critics.load_state_dict(module.critics.state_dict())

        module.update(self._make_model(obs_space, action_space, config))
        return module

    def _make_policy(self, obs_space, action_space, config):
        policy_config = config["module"]["policy"]

        def _make_modules():
            logits = mods.FullyConnected(
                in_features=obs_space.shape[0],
                units=policy_config["units"],
                activation=policy_config["activation"],
                layer_norm=policy_config.get(
                    "layer_norm", config["exploration"] == "parameter_noise"
                ),
                **policy_config["initializer_options"]
            )
            mu_ = mods.NormalizedLinear(
                in_features=logits.out_features,
                out_features=action_space.shape[0],
                beta=config["beta"],
            )
            squash = mods.TanhSquash(
                self.convert_to_tensor(action_space.low),
                self.convert_to_tensor(action_space.high),
            )
            return logits, mu_, squash

        logits_module, mu_module, squash_module = _make_modules()
        modules = {}
        modules["policy"] = nn.Sequential(logits_module, mu_module, squash_module)

        if config["exploration"] == "gaussian":
            expl_noise = mods.GaussianNoise(config["exploration_gaussian_sigma"])
            modules["sampler"] = nn.Sequential(
                logits_module, mu_module, expl_noise, squash_module
            )
        elif config["exploration"] == "parameter_noise":
            modules["sampler"] = modules["perturbed_policy"] = nn.Sequential(
                *_make_modules()
            )
        else:
            modules["sampler"] = modules["policy"]

        if config["target_policy_smoothing"]:
            modules["target_policy"] = nn.Sequential(
                logits_module,
                mu_module,
                mods.GaussianNoise(config["target_gaussian_sigma"]),
                squash_module,
            )
        else:
            modules["target_policy"] = modules["policy"]
        return modules

    @staticmethod
    def _make_critic(obs_space, action_space, config):
        critic_config = config["module"]["critic"]
        return mods.ActionValueFunction.from_scratch(
            obs_dim=obs_space.shape[0],
            action_dim=action_space.shape[0],
            delay_action=critic_config["delay_action"],
            units=critic_config["units"],
            activation=critic_config["activation"],
            **critic_config["initializer_options"]
        )

    @staticmethod
    def _make_model(obs_space, action_space, config):
        model_config = config["module"]["model"]
        model_module = mapom.DynamicsModel.from_scratch(
            obs_dim=obs_space.shape[0],
            action_dim=action_space.shape[0],
            input_dependent_scale=model_config["input_dependent_scale"],
            delay_action=model_config["delay_action"],
            units=model_config["units"],
            activation=model_config["activation"],
            **model_config["initializer_options"]
        )

        sampler_module = mapom.DynamicsModelRSample(model_module)
        return {"model": model_module, "model_sampler": sampler_module}

    @override(raypi.TorchPolicy)
    def optimizer(self):
        pi_cls = torch_util.get_optimizer_class(self.config["policy_optimizer"]["name"])
        pi_optim = pi_cls(
            self.module.policy.parameters(),
            **self.config["policy_optimizer"]["options"]
        )

        qf_cls = torch_util.get_optimizer_class(self.config["critic_optimizer"]["name"])
        qf_optim = qf_cls(
            self.module.critics.parameters(),
            **self.config["critic_optimizer"]["options"]
        )

        dm_cls = torch_util.get_optimizer_class(self.config["model_optimizer"]["name"])
        dm_optim = dm_cls(
            [self.module.log_alpha], **self.config["model_optimizer"]["options"]
        )

        return OptimizerCollection(policy=pi_optim, critic=qf_optim, model=dm_optim)
