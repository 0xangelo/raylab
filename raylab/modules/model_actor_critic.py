"""Architecture with disjoint model, actor, and critic."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.distributions import DiagMultivariateNormal
from .basic import (
    FullyConnected,
    StateActionEncoder,
    DiagMultivariateNormalParams,
    DistRSample,
    DistLogProb,
    DistReproduce,
)


class ModelActorCritic(nn.ModuleDict):
    """Module containing env model, policy, and value functions."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.update(self._make_model(obs_space, action_space, config))
        self.update(self._make_critic(obs_space, action_space, config))
        self.update(self._make_actor(obs_space, action_space, config))

    def _make_model(self, obs_space, action_space, config):
        model = nn.ModuleDict()
        model_config = config["model"]
        model.params = self._make_model_encoder(obs_space, action_space, model_config)

        dist_samp = DistRSample(
            dist_cls=DiagMultivariateNormal,
            detach_logp=False,
            low=torch.as_tensor(obs_space.low),
            high=torch.as_tensor(obs_space.high),
        )
        dist_logp = DistLogProb(
            dist_cls=DiagMultivariateNormal,
            low=torch.as_tensor(obs_space.low),
            high=torch.as_tensor(obs_space.high),
        )
        dist_repr = DistReproduce(
            dist_cls=DiagMultivariateNormal,
            low=torch.as_tensor(obs_space.low),
            high=torch.as_tensor(obs_space.high),
        )
        if config["torch_script"]:
            params_ = {
                "loc": torch.zeros(1, obs_space.shape[0]),
                "scale_diag": torch.ones(1, obs_space.shape[0]),
            }
            obs_ = torch.randn(1, *obs_space.shape)
            dist_samp = dist_samp.traced(params_)
            dist_logp = dist_logp.traced(params_, obs_)
            dist_repr = dist_repr.traced(params_, obs_)

        if model_config["residual"]:
            model.sampler = ResModelRSample(model.params, dist_samp)
            model.logp = ResModelLogProb(model.params, dist_logp)
            model.reproduce = ResModelReproduce(model.params, dist_repr)
        else:
            model.sampler = ModelRSample(model.params, dist_samp)
            model.logp = ModelLogProb(model.params, dist_logp)
            model.reproduce = ModelReproduce(model.params, dist_repr)

        return {"model": model}

    @staticmethod
    def _make_model_encoder(obs_space, action_space, config):
        return GaussianDynamicsParams(obs_space, action_space, config)

    @staticmethod
    def _make_critic(obs_space, action_space, config):
        # pylint:disable=unused-argument
        modules = {}
        critic_config = config["critic"]

        def make_vf():
            logits_mod = FullyConnected(
                in_features=obs_space.shape[0],
                units=critic_config["units"],
                activation=critic_config["activation"],
                **critic_config["initializer_options"],
            )
            value_mod = nn.Linear(logits_mod.out_features, 1)
            return nn.Sequential(logits_mod, value_mod)

        modules["critic"] = make_vf()
        modules["target_critic"] = make_vf()
        modules["target_critic"].load_state_dict(modules["critic"].state_dict())
        return modules

    @staticmethod
    def _make_actor(obs_space, action_space, config):
        actor = nn.ModuleDict()
        actor_config = config["actor"]

        logits_mod = FullyConnected(
            in_features=obs_space.shape[0],
            units=actor_config["units"],
            activation=actor_config["activation"],
            **actor_config["initializer_options"],
        )
        actor.params = nn.Sequential(
            logits_mod,
            DiagMultivariateNormalParams(
                logits_mod.out_features,
                action_space.shape[0],
                input_dependent_scale=actor_config["input_dependent_scale"],
            ),
        )

        dist_samp = DistRSample(
            dist_cls=DiagMultivariateNormal,
            detach_logp=False,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )
        if config["torch_script"]:
            params_ = {
                "loc": torch.zeros(1, *action_space.shape),
                "scale_diag": torch.ones(1, *action_space.shape),
            }
            dist_samp = dist_samp.traced(params_)
        actor.sampler = nn.Sequential(actor.params, dist_samp)

        return {"actor": actor}


class GaussianDynamicsParams(nn.Module):
    """
    Neural network module mapping inputs to Normal distribution parameters.
    """

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.logits = StateActionEncoder(
            obs_dim=obs_space.shape[0],
            action_dim=action_space.shape[0],
            units=config["units"],
            delay_action=config["delay_action"],
            activation=config["activation"],
            **config["initializer_options"],
        )
        self.params = DiagMultivariateNormalParams(
            self.logits.out_features,
            obs_space.shape[0],
            input_dependent_scale=config["input_dependent_scale"],
        )

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        return self.params(self.logits(obs, actions))


class ModelRSample(nn.Module):
    """Samples next states given an initial state and action."""

    def __init__(self, params_module, logp_module):
        super().__init__()
        self.params_module = params_module
        self.rsample_module = logp_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        return self.rsample_module(self.params_module(obs, actions))


class ModelLogProb(nn.Module):
    """Computes the log-likelihood of transitions."""

    def __init__(self, params_module, logp_module):
        super().__init__()
        self.params_module = params_module
        self.logp_module = logp_module

    @override(nn.Module)
    def forward(self, obs, actions, new_obs):  # pylint:disable=arguments-differ
        return self.logp_module(self.params_module(obs, actions), new_obs)


class ModelReproduce(nn.Module):
    """Reproduces observed transitions."""

    def __init__(self, params_module, resample_module):
        super().__init__()
        self.params_module = params_module
        self.resample_module = resample_module

    @override(nn.Module)
    def forward(self, obs, actions, new_obs):  # pylint:disable=arguments-differ
        self.resample_module(self.params_module(obs, actions), new_obs)


class ResModelRSample(ModelRSample):
    """Samples next states with a residual model."""

    @override(ModelRSample)
    def forward(self, obs, actions):
        res, logp = self.rsample_module(self.params_module(obs, actions))
        return obs + res, logp


class ResModelLogProb(ModelLogProb):
    """Computes the log-likelihood of transitions with a residual model."""

    @override(ModelLogProb)
    def forward(self, obs, actions, new_obs):
        residual = new_obs - obs
        return self.logp_module(self.params_module(obs, actions), residual)


class ResModelReproduce(ModelReproduce):
    """Reproduces observed transitions with a residual model."""

    @override(ModelReproduce)
    def forward(self, obs, actions, new_obs):
        residual = new_obs - obs
        return obs + self.resample_module(self.params_module(obs, actions), residual)
